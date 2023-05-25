#include "globals.h"
#include "global_templated_functions.h"
#include "Compute_bennet.h"

using namespace std;

Bennet::~Bennet(){ }

Bennet::Bennet(std::istringstream& iss) : Compute(iss) {
    style = "Bennet";

    // The Bennet compute will estimate the interfacial tension of a system
    // by changing the simulation box cross section while maintaining 
    // the same volume and calculating the change in free energy. 

    readRequiredParameter(iss, areaChange);
    readRequiredParameter(iss, normD);
    set_optional_args(iss);
}

void Bennet::doCompute() {

    // Copy original positions
    d_copyPositions<<<ns_Grid, ns_Block>>>(this->d_fdata, d_x, Dim, ns);
    
    // Regenerate density fields
    prepareDensityFields();
    
    // Calculate original energy of system
    calc_properties(0);
    float U_init = Unb;

    // Calculate original area and volume
    float orig_volume = L[0]*L[1]*L[2];
    float orig_area = 1.0;
    float L_orig[3] ;
    for (int j = 0; j < Dim; j++){
        L_orig[j] = L[j];
        if (j != normD){
            orig_area *= L[j];
        }
    }
    // Calculate new area and length in direction normal to interface
    float new_area = orig_area + areaChange;
    
    L[normD] = orig_volume/new_area;
    //cout << new_area << "\n";
    //cout << L[normD] << "\n"; 
    // Scale box lengths and grid spacing
    float lenChange;
    for (int j = 0; j < Dim; j++) {
        if (j != normD) 
        {  
            L[j] = L_orig[j] + (-2*L_orig[j]+sqrt(pow(2*L_orig[j], 2)+4*areaChange))/2;
            //dx[j] = L[j] / float(Nx[j]);
            lenChange = (L[j] - L_orig[j])/L_orig[j];
        }
        //cout << L[j] << "\n";
        dx[j] = L[j] / float(Nx[j]);

    }
    //cout << dx[normD -1] << "\n";

    // Scale positions

    d_scalePositions<<<ns_Grid, ns_Block>>>(d_x, this->d_fdata,  Dim, ns, lenChange, normD);
    
    // Copy box lengths and grid spacing from host to device
    cudaMemcpy(d_L, L,  Dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dx, dx,  Dim * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate new density fields and weights
    prepareDensityFields();

    // Calculate new energy
    calc_properties(0);
    float U_scaled = Unb;

    // Unscale box lengths and grid spacing
    for (int j = 0; j < Dim; j++) {
        L[j] = L_orig[j];
        dx[j] = L[j]/float(Nx[j]);
    }

    cudaMemcpy(d_L, L, 2 * Dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dx, dx,  Dim * sizeof(float), cudaMemcpyHostToDevice);

    //cout << dx[normD -1] << "\n";

    // Restore original  particle positions
    d_copyPositions<<<ns_Grid, ns_Block>>>(d_x, this->d_fdata, Dim, ns);

    // Recalculate density fields and energy for sanity check
    prepareDensityFields();
    calc_properties(0);

    float U_unscaled = Unb;

    //if(Unbi2 != Unbi)
    if(U_unscaled < U_init*0.99999 || U_unscaled > U_init*1.00001)
    {
        cout << "Original energy: " << U_init << " , new energy: " << U_unscaled << "\n";

        die("Initial and final energy states don't match.");
    }

    // Store the energy difference
    float deltaU = U_scaled - U_unscaled;

    this->fstore1[num_data_pts++] = deltaU;



}   

void Bennet::allocStorage(){
    string perpD = "z";
    string sys_num = "10";
    if (normD == 0)
        perpD = "x";
    else if (normD == 1)
        perpD = "y";

    if (areaChange < 0)
        sys_num = "01";
    
    ////////////////////////////////////////////
    // Set the intermittant storage variables //
    ////////////////////////////////////////////

    number_stored = log_freq / this->compute_freq;  
    this->fstore1 = (float *)malloc(number_stored * sizeof(float));

    cudaMalloc(&this->d_fdata, ns * Dim * sizeof(float));

    check_cudaError("Allocating d_fdata in compute");

    num_data_pts = 0;

    // Initialize output file
    ofstream outfile;
    outfile.open("deltaU_" + sys_num + "_" + perpD +"_areaD_"+ to_string(areaChange) + ".dat", ios::out);
    outfile.close();
}

void Bennet::writeResults(){
    string perpD = "z";
    string sys_num = "10";
    if (normD == 0)
        perpD = "x";
    else if (normD == 1)
        perpD = "y";

    if (areaChange < 0)
        sys_num = "01";
    
    // Append fstore1 to file deltaU.dat  
    ofstream outfile;
    outfile.open("deltaU_" + sys_num + "_" + perpD +"_areaD_"+ to_string(areaChange) + ".dat", ios::app);
    for ( int i=0 ; i<num_data_pts ; i++ ) {
        outfile << fstore1[i] << "\n";
    }
    outfile.close();

    num_data_pts = 0;
}