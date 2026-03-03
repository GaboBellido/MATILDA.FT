// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"

using namespace std;

void get_r(int, float*);
void unstack(int, int*);
void init_binary_output(void);

// Persistent file handles for binary output.  Kept open between frames to
// avoid repeated fopen/fclose overhead on every write_binary() call.
// init_binary_output() opens them; they are closed automatically when the
// process exits (or can be explicitly flushed/closed if needed).
static FILE* s_dens_fh = NULL;
static FILE* s_pos_fh  = NULL;

void write_grid_data(const char* lbl, float* dat) {

    int i, j, * nn;
    nn = new int[Dim];
    FILE* otp;
    float* r = new float [Dim];


    otp = fopen(lbl, "w");

    for (i = 0; i < M; i++) {
        get_r(i, r);
        unstack(i, nn);

        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", r[j]);

        fprintf(otp, "%1.8e \n", dat[i]);

        if (Dim == 2 && nn[0] == Nx[0] - 1)
            fprintf(otp, "\n");
    }

    fclose(otp);

}

void write_kspace_cudaComplex(const char* lbl, cufftComplex* kdt) {
    int i, j, nn[3];
    FILE* otp;
    float kv[3], k2;

    otp = fopen(lbl, "w");

    for (i = 1; i < M; i++) {
        unstack(i, nn);

        k2 = get_k(i, kv, Dim);

        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", kv[j]);

        float cpx_abs = sqrtf(kdt[i].x * kdt[i].x + kdt[i].y * kdt[i].y);
        fprintf(otp, "%1.5e %1.5e %1.5e %1.5e\n", cpx_abs, sqrtf(k2),
            kdt[i].x, kdt[i].y);

        if (Dim == 2 && nn[0] == Nx[0] - 1)
            fprintf(otp, "\n");
    }

    fclose(otp);
}

void write_kspace_data(const char* lbl, complex<float> * kdt) {
    int i, j, nn[3];
    FILE* otp;
    float kv[3], k2;

    otp = fopen(lbl, "w");

    for (i = 1; i < M; i++) {
        unstack(i, nn);

        k2 = get_k(i, kv, Dim);

        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", kv[j]);

        fprintf(otp, "%1.5e %1.5e %1.5e %1.5e\n", abs(kdt[i]), sqrt(k2),
            real(kdt[i]), imag(kdt[i]));

        if (Dim == 2 && nn[0] == Nx[0] - 1)
            fprintf(otp, "\n");
    }

    fclose(otp);
}

void init_binary_output() {
    // Close any previously open persistent handles (equil -> prod transition).
    if (s_dens_fh) { fclose(s_dens_fh); s_dens_fh = NULL; }
    if (s_pos_fh)  { fclose(s_pos_fh);  s_pos_fh  = NULL; }

    const char* dens_name = (equil && equilData) ? "equil_grid_densities.bin"
                                                  : "grid_densities.bin";
    const char* pos_name  = (equil && equilData) ? "equil_positions.bin"
                                                  : "positions.bin";

    // --- Write grid-density header (create/truncate) ---
    FILE* otp = fopen(dens_name, "wb");
    if (otp == NULL)
        die("failed to open grid_densities.bin");

    fwrite(&Dim, sizeof(int), 1, otp);
    fwrite(Nx, sizeof(int), Dim, otp);
    fwrite(L, sizeof(float), Dim, otp);
    fwrite(&ntypes, sizeof(int), 1, otp);
    fclose(otp);

    // --- Write positions header (create/truncate) ---
    otp = fopen(pos_name, "wb");
    if (otp == NULL)
        die("Failed to open positions.bin");

    fwrite(&ns, sizeof(int), 1, otp);
    fwrite(&Dim, sizeof(int), 1, otp);
    fwrite(L, sizeof(float), 3, otp);  // always 3 floats so readers see full L
    fwrite(tp, sizeof(int), ns, otp);
    fwrite(molecID, sizeof(int), ns, otp);
    if ( Charges::do_charges == 1 )
        fwrite(charges, sizeof(float), ns, otp);
    fclose(otp);

    // Re-open both files in append mode and keep handles alive for all
    // subsequent write_binary() calls this phase (equil or production).
    s_dens_fh = fopen(dens_name, "ab");
    if (s_dens_fh == NULL)
        die("Failed to reopen grid_densities.bin for appending");

    s_pos_fh = fopen(pos_name, "ab");
    if (s_pos_fh == NULL)
        die("Failed to reopen positions.bin for appending");
}

void write_binary() {
    if (equil && !equilData)
        return;

    // Persistent handles must have been opened by init_binary_output().
    if (s_dens_fh == NULL || s_pos_fh == NULL)
        die("write_binary: binary output files not open (init_binary_output not called?)");

    fwrite(all_rho,    sizeof(float), M * ntypes, s_dens_fh);
    fwrite(h_ns_float, sizeof(float), ns * Dim,   s_pos_fh);
}

void write_struc_fac() {
    // Guard against divide-by-zero if called before any S(k) has been accumulated.
    if (n_avg_calc == 0) {
        printf("write_struc_fac: n_avg_calc == 0, skipping output.\n");
        return;
    }

    // Declare Local Variables
    FILE* otp;
    int i, j, k, nn[3];
    float kv[3], k2;
    double temp;
    char label [30];

    for (i = 0; i < ntypes; i++) {
        // Open output file

        sprintf(label, "sk%d.dat", i);
        //sprintf(label, "sk%d_%d.dat", i,step);
        otp = fopen(label, "w");
        if (otp == NULL)
            die("Failed to write to sk.dat"); //Check to see that output file actually opened


    
        //fprintf(otp, "Printing type: %d\n", i);
        // SKIP j=o to skip 0 0 0 point
        for (j = 1; j < M; j++) {
            /// Get kspace coordinates
            unstack(j, nn);
            k2 = get_k(j, kv, Dim);
            /// Print kspace coordinates
            for (k = 0; k < Dim; k++) {
                fprintf(otp, "%f ", kv[k]);
            }
            /// calculate the avg (sum) over total calcuations
            temp = avg_sk[i][j] / n_avg_calc;
            /// Write data points. Calculations based on binToSk script on git
            fprintf(otp, "%1.5e %1.5e %1.5e %1.5e\n", abs(temp), sqrt(k2), real(temp), imag(temp));
        }
        fclose(otp);
    }
    
}