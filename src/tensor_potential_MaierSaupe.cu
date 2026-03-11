// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "tensor_potential_MaierSaupe.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include "device_utils.cuh"
#include "timing.h"

using namespace std; 

__global__ void d_zero_particle_forces(float*, int, int);
void unstack(int, int*);


void MaierSaupe::CalcForces() {
    MaierSaupe_t_in = int(time(0));

    this->CalcSTensors();

    // First contribution to the force from grad u(rij)
    for ( int j=0 ; j<Dim ; j++ ) {
        // Loop over S-tensor components to convolve with grad_j u(r)
        // Uses the symmetry to halve the calculations
        for ( int k=0 ; k<Dim ; k++ ) {
            for ( int m=k ; m<Dim ; m++ ) {

                // Grab component k, m
                // d_cpx1 = S_km
                d_extractTensorComponent<<<M_Grid, M_Block>>>(d_cpx1, 
                    this->d_S_field, k, m, M, Dim);

                // FFT component k, m
                // d_cpx2 = FFT(S_km)
                cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);
                check_cudaError("FFTing in Maier-Saupe forces");

                // Multiply component j of d_f_k with d_cpx2
                // Result stored in d_cpx1
                d_prepareForceKSpace<<<M_Grid, M_Block>>>(this->d_f_k, d_cpx2, d_cpx1, j, Dim, M);

                // d_cpx2 = IFFT of above convolution
                cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_INVERSE);
                check_cudaError("Inverse FFT in Maier-Saupe forces");

                // Store the result in the temporary tensor field
                d_storeTensorComponent<<<M_Grid, M_Block>>>(this->d_tmp_tensor,
                    d_cpx2, k, m, M, Dim);

                // Use the symmetry to store the m,k component too
                if ( k != m ) {
                    d_storeTensorComponent<<<M_Grid, M_Block>>>(this->d_tmp_tensor,
                        d_cpx2, m, k, M, Dim);
                }
            }// m=k:Dim
        }// k=0:Dim

        d_accumulateMSForce1<<<ns_Grid, ns_Block>>>(::d_f, this->d_MS_pair, this->d_tmp_tensor, this->d_ms_S, 
            d_grid_W, d_grid_inds, gvol, j, grid_per_partic, ns, Dim);

    }// j=0:Dim loop over the dimensions of grad u



    // Second contribution from du/dri

    // First, convole S field with u(r)
    for ( int k=0 ; k<Dim ; k++ ) {
        for ( int m=k; m<Dim ; m++ ) {
            // Grab component k, m
            // d_cpx1 = S_km
            d_extractTensorComponent<<<M_Grid, M_Block>>>(d_cpx1, 
                this->d_S_field, k, m, M, Dim);

            // FFT component k, m
            // d_cpx2 = FFT(S_km)
            cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);
            check_cudaError("FFTing in Maier-Saupe forces");

            // Multiply component j of d_f_k with d_cpx2
            // Result stored in d_cpx1
            d_multiplyComplex<<<M_Grid, M_Block>>>(this->d_u_k, d_cpx2, d_cpx1, M);

            // d_cpx2 = IFFT of above convolution
            cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_INVERSE);
            check_cudaError("Inverse FFT in Maier-Saupe forces");

            // Store the result in the temporary tensor field
            d_storeTensorComponent<<<M_Grid, M_Block>>>(this->d_tmp_tensor,
                d_cpx2, k, m, M, Dim);

            // Use the symmetry to store the m,k component too
            if ( k != m ) {
                d_storeTensorComponent<<<M_Grid, M_Block>>>(this->d_tmp_tensor,
                    d_cpx2, m, k, M, Dim);
            }            
        }// m=k:Dim
    }// k=0:Dim

    d_accumulateMSForce2<<<ns_Grid, ns_Block>>>(::d_f, d_x, this->d_MS_pair, this->d_tmp_tensor, this->d_ms_u, 
            d_grid_W, d_grid_inds, gvol, grid_per_partic, ns, d_L, d_Lh, Dim);

    MaierSaupe_t_out = int(time(0));
    MaierSaupe_tot_time += MaierSaupe_t_out - MaierSaupe_t_in;
}


// Routine to calculate MS potential energy
// Assumes that forces have been called, meaning:
// this->d_tmp_tensor contains (S*u)(r)
// this->d_S_field is already populated with S(r)
float MaierSaupe::CalcEnergy() {

    d_doubleDotTensorFields<<<M_Grid, M_Block>>>(d_tmp, this->d_tmp_tensor, this->d_S_field, M, Dim);

    cudaMemcpy(tmp, d_tmp, M*sizeof(float), cudaMemcpyDeviceToHost);

    this->energy = -integ_trapPBC(tmp);
    return energy;
}


// Calculate the S Tensors for all of the particles
void MaierSaupe::CalcSTensors() {

    // Calculate particle-level S tensors
    d_calcParticleSTensors<<<ns_Grid, ns_Block>>>(this->d_ms_u, this->d_ms_S, d_x,
        this->d_MS_pair, d_L, d_Lh, Dim, ns);
    check_cudaError("Calculate particle-level S tensors");

    // Zero the Dim*Dim*M S tensor field
    int biggerM = M*Dim*Dim;
    int bM_Grid = (int)ceil((float(biggerM) / M_Block));
    d_zero_float_vector<<<bM_Grid, M_Block>>>(this->d_S_field, biggerM);
    check_cudaError("Zeroing S field");

    // Map the particle S to the field S
    d_mapFieldSTensors<<<ns_Grid, ns_Block>>>(this->d_S_field, this->d_MS_pair, this->d_ms_S,
        d_grid_W, d_grid_inds, ns, grid_per_partic, Dim);

    check_cudaError("MapSTensors in Maier-Saupe forces");

    // Record the step so CalculateOrderParameter() can skip a redundant call
    // when it runs in the same time step as CalcForces().
    last_stensors_step = step;
}

void MaierSaupe::DistributeSTensors() {
    // Calculate particle-level S tensors
    d_calcParticleSTensors<<<ns_Grid, ns_Block>>>(this->d_ms_u, this->d_ms_S, d_x,
        this->d_MS_pair, d_L, d_Lh, Dim, ns);
    check_cudaError("Calculate particle-level S tensors");

    // Copy the particle S tensors and partner list to the host
    cudaMemcpy(this->ms_S, this->d_ms_S, Dim*Dim*ns*sizeof(float), cudaMemcpyDeviceToHost);
    check_cudaError("Copy ms_S to host in DistributeSTensors");
    cudaMemcpy(this->MS_pair, this->d_MS_pair, ns*sizeof(int), cudaMemcpyDeviceToHost);
    check_cudaError("Copy MS_pair to host in DistributeSTensors");

    // Propagate the head-particle S tensor to all co-molecular particles.
    // Uses the precomputed molec_to_particles map (built in Allocate()) for an
    // O(ns) pass instead of the previous O(ns^2) nested loop.
    for (int i = 0; i < ns; i++) {
        if (this->MS_pair[i] > 1) {
            int mol = molecID[i];
            for (int j : molec_to_particles.at(mol)) {
                this->MS_pair[j] = 1;
                for (int k = 0; k < Dim*Dim; k++)
                    this->ms_S[j*Dim*Dim + k] = this->ms_S[i*Dim*Dim + k];
            }
        }
    }

    // Copy updated arrays back to the device
    cudaMemcpy(this->d_MS_pair, this->MS_pair, ns*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_ms_S, this->ms_S, Dim*Dim*ns*sizeof(float), cudaMemcpyHostToDevice);
    check_cudaError("Copy ms_S to device in DistributeSTensors");

    // Map the distributed particle S tensors to the field
    d_mapDistributedFieldSTensors<<<ns_Grid, ns_Block>>>(this->d_S_field, this->d_MS_pair, this->d_ms_S,
        d_grid_W, d_grid_inds, ns, grid_per_partic, Dim);
    check_cudaError("MapSTensors in Maier-Saupe forces");
}

MaierSaupe::MaierSaupe(istringstream &iss) : Potential(iss) {
	potential_type = "MaierSaupe";
	type_specific_id = num++;

	readRequiredParameter(iss, filename);
	readRequiredParameter(iss, initial_prefactor);
	readRequiredParameter(iss, sigma_squared);

	final_prefactor = initial_prefactor;
	sigma_squared *= sigma_squared;

	ramp_check_input(iss);

}

void MaierSaupe::Initialize() {
    Initialize_Potential();
    Initialize_TensorPotential();
    Allocate();
}


void MaierSaupe::Allocate() {

    // Allocate memory for this potential
    int ns_alloc = ns + extra_ns_memory;
    
    this->MS_pair = (int*) calloc(ns_alloc, sizeof(int));
    cudaMalloc(&this->d_MS_pair, ns_alloc * sizeof(int));
    mem_use += ns_alloc * sizeof(int);
    device_mem_use += ns_alloc * sizeof(int);

    h_Dim_Dim_tensor = (float*) calloc(Dim*Dim, sizeof(float));
    cudaMalloc(&this->d_Dim_Dim_tensor, Dim * Dim* sizeof(float));
    cout << "Allocating " << Dim*Dim << " bytes for tmp_tensor" << endl;

    int size = Dim * ns;
    this->ms_u = (float*) calloc(size, sizeof(float));
    cudaMalloc(&this->d_ms_u, size * sizeof(float));
    mem_use += size * sizeof(float);
    device_mem_use += size * sizeof(float);

    size = Dim * Dim * ns;
    this->ms_S = (float*) calloc(size, sizeof(float));
    cudaMalloc(&this->d_ms_S, size * sizeof(float));
    mem_use += size * sizeof(float);
    device_mem_use += size * sizeof(float);

    size = Dim * Dim * M;
    this->S_field = ( float* ) calloc(size, sizeof(float));
    cudaMalloc(&this->d_S_field, size * sizeof(float));
    cudaMalloc(&this->d_tmp_tensor, size * sizeof(float));
    mem_use += size * sizeof(float);
    device_mem_use += 2 * size * sizeof(float);

    this->allocated = true;
    // End memory allocation
    check_cudaError("Allocating memory for Maier Saupe");

    // Sentinel: S tensors have not been computed for any step yet.
    last_stensors_step = -1;

    // Set all partners initially to -1 (non-LC sentinel)
    for ( int i=0 ; i<ns ; i++ ) this->MS_pair[i] = -1;

    // Build molecule-ID -> particle-index map once at initialization.
    // DistributeSTensors() uses this for an O(ns) lookup instead of O(ns^2).
    molec_to_particles.clear();
    for (int i = 0; i < ns; i++)
        molec_to_particles[molecID[i]].push_back(i);

    this->read_lc_file(this->filename);


    // Initialize the Gaussian potential
    // Initialize in k-space

    init_device_gaussian<<<M_Grid, M_Block>>>(this->d_u_k, this->d_f_k,
        initial_prefactor, this->sigma_squared, d_L, M, d_Nx, Dim);

    init_device_gaussian<<<M_Grid, M_Block>>>(this->d_master_u_k, this->d_master_f_k,
        1, this->sigma_squared, d_L, M, d_Nx, Dim);

    // Inverse transform into real-space
    cufftExecC2C(fftplan, this->d_u_k, d_cpx1, CUFFT_INVERSE);

    // Store real-space version
    d_complex2real<<<M_Grid, M_Block>>>(d_cpx1, this->d_u, M);

    // Store the potential on the host, too
    cudaMemcpy(this->u, this->d_u, M*sizeof(float), cudaMemcpyDeviceToHost);


}


void MaierSaupe::read_lc_file(string name) {
    FILE *inp;
    inp = fopen(name.c_str(), "r");
    if ( inp == NULL ) 
        die("MaierSaupe input file not found!");

    int id1, id2, di;

    // Reads the number of MaierSaupe pairs
    (void)!fscanf(inp, "%d\n", &nms);

    // Scans the rest of the file for all the MS pairs
    // Note the file is expected to be 1-indexed, so 
    // the -1 below is to shift to 0 indexing.
    for ( int i=0 ; i<nms ; i++ ) {
        (void)!fscanf(inp, "%d %d %d\n", &di, &id1, &id2);

        this->MS_pair[id1-1] = id2-1;
    }

    fclose(inp);

    // Copy the ms list to the device
    cudaMemcpy(this->d_MS_pair, this->MS_pair, ns*sizeof(int), cudaMemcpyHostToDevice);
}


MaierSaupe::MaierSaupe() : TensorPotential() {
    type1 = -1; 
    type2 = -1; 
}

MaierSaupe::~MaierSaupe() {
    if (op_fh) { fclose(op_fh); op_fh = nullptr; }
}

void MaierSaupe::ramp_check_input(istringstream& iss){

    if (iss.fail()){
        die("Error during input script; failed to properly read:\n" + iss.str());
    }

    string convert;
    iss >> convert;

    if (!iss.fail()){
        if (convert == "ramp") {
            ramp = true;
            iss >> final_prefactor;
            if(iss.fail()) die("no final prefactor specified");
            cout << "Ramping prefactor of " <<potential_type<< " style from " << initial_prefactor \
                << " to " << final_prefactor << endl;

            cout << "Estimated per time step change: " << \
                (final_prefactor - initial_prefactor) / (prod_steps)
                << endl;

        }
        else 
            die("Invalid keyword: " + convert);
    }


}

float MaierSaupe::CalculateOrderParameter(){

    // Skip recomputation if CalcSTensors() already ran in this time step
    // (e.g., it was called from CalcForces() just before ReportEnergies()).
    if (last_stensors_step != step) {
        CalcSTensors();
        check_cudaError("Calculate S tensor in CalculateOrderParameter");
    }

    // Average the particle S tensors to the device

    // Zero the Dim*Dim*M S tensor field
    int DD = Dim*Dim;

    d_zero_float_vector<<<1, DD>>>(d_Dim_Dim_tensor, Dim*Dim);
    check_cudaError("Zero d_tmp_tensor in CalculateOrderParameter");

    d_SumAndAverageSTensors<<<ns_Grid, ns_Block>>>(this->d_ms_S, this->d_Dim_Dim_tensor, this->d_MS_pair, Dim, ns);
    check_cudaError("Average STensors in CalculateOrderParameter");

    // Copy Dim*Dim float values from d_tmp_tensor to h_tmp_tensor

    cudaMemcpy(this->h_Dim_Dim_tensor, this->d_Dim_Dim_tensor, Dim*Dim*sizeof(float), cudaMemcpyDeviceToHost);

    check_cudaError("Copy d_tmp_tensor to host in CalculateOrderParameter");


    return CalculateMaxEigenValue(&h_Dim_Dim_tensor[0]) / float(nms);
}

void MaierSaupe::CalculateOrderParameterGridPoints(){


    // Distribute the S tensors to the rest of the particles
    DistributeSTensors();
    check_cudaError("Distribute S tensor in CalculateOrderParameterGridPoints");

    // Zero the Dim*Dim*M S tensor field
    int DDM = Dim*Dim*M;
    cudaMemcpy(this->S_field, this->d_S_field, DDM*sizeof(float), cudaMemcpyDeviceToHost);
    check_cudaError("Copy d_S_field to host in CalculateOrderParameterGridPoints");

    static std::vector<float> per_grid_eigen_value(M, 0);

    for (int i = 0; i < M; i++)
        per_grid_eigen_value[i] = CalculateMaxEigenValue(&S_field[i * Dim*Dim]);

    // Normalize by the physical maximum eigenvalue of the traceless nematic S tensor:
    //   lambda_max = (Dim - 1) / Dim   (e.g. 2/3 in 3D, 1/2 in 2D for perfect alignment)
    // Dividing by the spatial maximum instead would pin every frame's most-ordered
    // grid point to 1.0, making cross-frame and cross-simulation comparisons meaningless.
    const float physical_max = float(Dim - 1) / float(Dim);
    for (size_t i = 0; i < per_grid_eigen_value.size(); ++i)
        per_grid_eigen_value[i] /= physical_max;

    // Use a fixed-size buffer safe for 2D and 3D (VLAs are not standard C++).
    int nn[3] = {0, 0, 0};

    // Define the output filename based on the step number
    std::ostringstream filename;
    filename << "order_parameter_step_" << step << ".csv";

    // Open the file in write mode
    std::ofstream fileout(filename.str());

    // Dim-aware CSV header
    if (Dim == 3)
        fileout << "x,y,z,lambda\n";
    else
        fileout << "x,y,lambda\n";

    for (int i = 0; i < M; i++) {
        unstack(i, nn);
        for (int d = 0; d < Dim; d++)
            fileout << nn[d] << ",";
        fileout << per_grid_eigen_value.at(i) << "\n";
    }

    fileout.close();  // Close the file after writing

}


// Open (or re-open) the binary order-parameter file for this simulation phase.
// Writes the grid header once, then keeps the file handle open so that
// WriteBinaryOP() can append frames efficiently without repeated fopen/fclose.
//
// File format:
//   Header  : Dim (int32), Nx[0..Dim-1] (int32[Dim]), L[0..Dim-1] (float32[Dim]),
//              M (int32) — total number of grid points
//   Per frame: step (int32), lambda[0..M-1] (float32[M]) — normalised eigenvalue
//              at each grid point, in the same flat order as other grid fields
void MaierSaupe::InitBinaryOP() {
    // Close any handle left over from a previous phase (equil → production).
    if (op_fh) { fclose(op_fh); op_fh = nullptr; }

    const char* fname = (equil && equilData) ? "equil_order_parameter.bin"
                                             : "order_parameter.bin";

    // Write the header (create / truncate the file).
    FILE* f = fopen(fname, "wb");
    if (f == NULL) die("InitBinaryOP: failed to open order_parameter.bin");
    fwrite(&Dim, sizeof(int),   1,   f);
    fwrite(Nx,   sizeof(int),   Dim, f);
    fwrite(L,    sizeof(float), Dim, f);
    fwrite(&M,   sizeof(int),   1,   f);
    fclose(f);

    // Re-open in append mode; keep handle alive for all subsequent frames.
    op_fh = fopen(fname, "ab");
    if (op_fh == NULL) die("InitBinaryOP: failed to reopen order_parameter.bin for appending");
}


// Compute the per-grid-point nematic order parameter and append one frame to
// the binary file opened by InitBinaryOP().  Mirrors the logic of
// CalculateOrderParameterGridPoints() but writes binary instead of CSV.
void MaierSaupe::WriteBinaryOP() {
    if (equil && !equilData) return;  // skip if not writing equilibration data

    if (op_fh == NULL)
        die("WriteBinaryOP: order_parameter.bin not open (InitBinaryOP not called?)");

    // Distribute S tensors to all grid points (populates d_S_field).
    DistributeSTensors();
    check_cudaError("DistributeSTensors in WriteBinaryOP");

    // Copy the full Dim*Dim*M S-tensor field to the host buffer.
    int DDM = Dim * Dim * M;
    cudaMemcpy(this->S_field, this->d_S_field, DDM * sizeof(float), cudaMemcpyDeviceToHost);
    check_cudaError("Copy d_S_field to host in WriteBinaryOP");

    // Compute the leading eigenvalue at each grid point and normalise by the
    // physical maximum for a traceless nematic tensor: (Dim-1)/Dim.
    const float physical_max = float(Dim - 1) / float(Dim);
    std::vector<float> eigen_buf(M);
    for (int i = 0; i < M; i++)
        eigen_buf[i] = CalculateMaxEigenValue(&S_field[i * Dim * Dim]) / physical_max;

    // Write step number followed by the M eigenvalue floats.
    fwrite(&step,           sizeof(int),   1, op_fh);
    fwrite(eigen_buf.data(), sizeof(float), M, op_fh);
}


void MaierSaupe::ReportEnergies(int& die_flag){
    static int counter = 0;
	static string reported_energy = "";
	static string reported_order = "";

    string tmp_energy =  " " + to_string(energy);
    string tmp_order =  " " + to_string(CalculateOrderParameter());

    dout << tmp_energy;
    dout << tmp_order;

	reported_energy += tmp_energy;
	reported_order += tmp_order;
	if (std::isnan(energy)) die_flag = 1 ;
    

    if (++counter == num){
        cout << " UMaierSaupe:" + reported_energy;
        cout << " LambdaMaierSaupe:" + reported_order;
        counter=0;
		reported_energy.clear();
		reported_order.clear();
    }

}

int MaierSaupe::num = 0;
