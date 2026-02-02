#include <complex>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

#define PI 3.141592653589793238462643383

void unstack(int, int*, int*, int);
void write_header(FILE *ot, int ntypes);

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: avg-grid-densities [input.bin] [output_name] [frame_step] [optional: n skipped frames]" << endl;
        exit(1);
    }

    int Dim, Nx[3], ntypes, M, rt, skip, step;
    float L[3], *all_rho, dx[3];
    vector<float> rho_avg;

    skip = -1;
    step = atoi(argv[3]); // Frame step
    if (argc == 5) {
        skip = atoi(argv[4]);
    }

    FILE *inp;

    inp = fopen(argv[1], "rb");
    if (inp == NULL) {
        cout << "Failed to open " << argv[1] << endl;
        exit(1);
    }

    // Read header
    rt = fread(&Dim, sizeof(int), 1, inp);
    rt = fread(Nx, sizeof(int), Dim, inp);
    rt = fread(L, sizeof(float), Dim, inp);
    rt = fread(&ntypes, sizeof(int), 1, inp);

    cout << "Dim = " << Dim << endl;
    cout << "Nx[0]: " << Nx[0] << endl;
    cout << "ntypes: " << ntypes << endl;

    M = 1;
    for (int i = 0; i < Dim; i++) {
        M *= Nx[i];
        dx[i] = L[i] / float(Nx[i]);
    }
    if (Dim == 2)
        dx[2] = 1.0;

    all_rho = new float[ntypes * M];
    rho_avg.resize(ntypes * M, 0.0f); // Initialize average accumulator

    int nframes = 0, selected_frames = 0;

    while (!feof(inp)) {
        rt = fread(all_rho, sizeof(float), M * ntypes, inp);

        if (rt != M * ntypes) {
            cout << "Successfully processed " << nframes << " frames" << endl;
            break;
        }

        // Process frames based on skip and step
        if (nframes >= skip && (nframes % step == 0)) {
            for (int i = 0; i < M * ntypes; i++) {
                rho_avg[i] += all_rho[i]; // Accumulate density values
            }
            selected_frames++;
        }

        nframes++;
    }
    fclose(inp);

    // Compute averages
    for (int i = 0; i < M * ntypes; i++) {
        rho_avg[i] /= selected_frames;
    }

    // Output results in CSV format
    string output_name_csv = argv[2];
    output_name_csv += ".csv";
    FILE *otp_csv = fopen(output_name_csv.c_str(), "w");
    if (otp_csv == NULL) {
        cout << "Failed to open output file " << output_name_csv << endl;
        exit(1);
    }

    write_header(otp_csv, ntypes);

    for (int i = 0; i < M; i++) {
        int nn[3];
        unstack(i, nn, Nx, Dim);

        for (int j = 0; j < Dim; j++) {
            fprintf(otp_csv, "%1.3e, ", float(nn[j]) * dx[j]);
        }

        for (int j = 0; j < ntypes; j++) {
            if (j == ntypes - 1) {
                fprintf(otp_csv, "%1.3e", rho_avg[j * M + i]);
            } else {
                fprintf(otp_csv, "%1.3e, ", rho_avg[j * M + i]);
            }
        }

        fprintf(otp_csv, "\n");

        if (Dim == 2 && nn[0] == Nx[0] - 1) {
            fprintf(otp_csv, "\n");
        }
    }
    fclose(otp_csv);

    // Output results in binary format
    string output_name_bin = argv[2];
    output_name_bin += ".bin";
    FILE *otp_bin = fopen(output_name_bin.c_str(), "wb");
    if (otp_bin == NULL) {
        cout << "Failed to open output file " << output_name_bin << endl;
        exit(1);
    }

    fwrite(&Dim, sizeof(int), 1, otp_bin);
    fwrite(Nx, sizeof(int), Dim, otp_bin);
    fwrite(L, sizeof(float), Dim, otp_bin);
    fwrite(&ntypes, sizeof(int), 1, otp_bin);
    fwrite(rho_avg.data(), sizeof(float), M * ntypes, otp_bin);
    fclose(otp_bin);

    delete[] all_rho;
    return 0;
}

void write_header(FILE *ot, int ntypes) {
    fprintf(ot, "\"X\", \"Y\", \"Z\",");
    for (int i = 0; i < ntypes; i++) {
        if (i == ntypes - 1) {
            fprintf(ot, " \"rho%d\"", i);
        } else {
            fprintf(ot, " \"rho%d\",", i);
        }
    }
    fprintf(ot, "\n");
}

void unstack(int id, int *nn, int *Nx, int Dim) {
    if (Dim == 1) {
        nn[0] = id;
        return;
    } else if (Dim == 2) {
        nn[1] = id / Nx[0];
        nn[0] = id - nn[1] * Nx[0];
        return;
    } else if (Dim == 3) {
        nn[2] = id / Nx[1] / Nx[0];
        nn[1] = id / Nx[0] - nn[2] * Nx[1];
        nn[0] = id - (nn[1] + nn[2] * Nx[1]) * Nx[0];
    } else {
        cout << "Dim is invalid!" << endl;
        return;
    }
}
