#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace std;
void unstack(int id, int* nn, int* Nx, int Dim);

// Smooth transition function using complementary error function (erfc)
float smooth_density(float dist, float radius, float sigma) {
    return 0.5f * erfc((dist - radius) / sigma);  // Smooth transition from 1 to 0
}

void create_sphere_density(const char* filename, int Dim, int* Nx, float* L, int ntypes, float radius, float sigma) {
    FILE* otp = fopen(filename, "wb");
    if (!otp) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }

    // Write grid metadata
    fwrite(&Dim, sizeof(int), 1, otp);           // Write dimension
    fwrite(Nx, sizeof(int), Dim, otp);           // Write grid points per dimension
    fwrite(L, sizeof(float), Dim, otp);          // Write grid size in each dimension
    fwrite(&ntypes, sizeof(int), 1, otp);        // Write number of types

    // Calculate the number of grid points
    int M = 1;
    for (int i = 0; i < Dim; i++) {
        M *= Nx[i];
    }

    // Create the density grid
    vector<float> all_rho(M * ntypes, 0.0f); // Initialize densities to 0

    // Define the center of the sphere
    float center[3] = {L[0] / 2.0f, L[1] / 2.0f, (Dim == 3 ? L[2] / 2.0f : 0.0f)};

    // Generate sphere density inside the grid
    for (int i = 0; i < M; i++){
        int nn[3] = {0, 0, 0};
        unstack(i, nn, Nx, Dim);
        float pos[3] = {nn[0] * (L[0] / Nx[0]), nn[1] * (L[1] / Nx[1]), (Dim == 3 ? nn[2] * (L[2] / Nx[2]) : 0.0f)};
        float dist = sqrt(pow(pos[0] - center[0], 2) + pow(pos[1] - center[1], 2) + (Dim == 3 ? pow(pos[2] - center[2], 2) : 0));
        
        // Use smooth transition for density based on complementary error function
        all_rho[i] = smooth_density(dist, radius, sigma);
        all_rho[i+M] = 1 - smooth_density(dist, radius, sigma);
    }

    // Write the densities to the binary file
    fwrite(all_rho.data(), sizeof(float), M * ntypes, otp);

    fclose(otp);
}

void create_cylinder_density(const char* filename, int Dim, int* Nx, float* L, int ntypes, float radius, float sigma) {
    FILE* otp = fopen(filename, "wb");
    if (!otp) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }

    // Write grid metadata
    fwrite(&Dim, sizeof(int), 1, otp);           // Write dimension
    fwrite(Nx, sizeof(int), Dim, otp);           // Write grid points per dimension
    fwrite(L, sizeof(float), Dim, otp);          // Write grid size in each dimension
    fwrite(&ntypes, sizeof(int), 1, otp);        // Write number of types

    // Calculate the number of grid points
    int M = 1;
    for (int i = 0; i < Dim; i++) {
        M *= Nx[i];
    }

    // Create the density grid
    vector<float> all_rho(M * ntypes, 0.0f); // Initialize densities to 0

    // Define the center axis of the cylinder
    float center[2] = {L[0] / 2.0f, L[1] / 2.0f};

    // Generate cylinder density inside the grid
    for (int i = 0; i < M; i++){
        int nn[3] = {0, 0, 0};
        unstack(i, nn, Nx, Dim);
        float pos[3] = {nn[0] * (L[0] / Nx[0]), nn[1] * (L[1] / Nx[1]), nn[2] * (L[2] / Nx[2])};
        float dist = sqrt(pow(pos[0] - center[0], 2) + pow(pos[1] - center[1], 2));

        all_rho[i] = smooth_density(dist, radius, sigma);
        
        all_rho[i+M] = 1 - smooth_density(dist, radius, sigma);

    }

    // Write the densities to the binary file
    fwrite(all_rho.data(), sizeof(float), M * ntypes, otp);

    fclose(otp);
}

void unstack(int id, int* nn, int* Nx, int Dim) {
    if (Dim == 1) {
        nn[0] = id;
    } else if (Dim == 2) {
        nn[1] = id / Nx[0];
        nn[0] = id - nn[1] * Nx[0];
    } else if (Dim == 3) {
        nn[2] = id / (Nx[1] * Nx[0]);
        nn[1] = (id / Nx[0]) % Nx[1];
        nn[0] = id % Nx[0];
    } else {
        cout << "Invalid Dimension!" << endl;
    }
}
int main(int argc, char** argv) {
    // Example of grid setup for a 3D sphere or cylinder
    int Dim = 3;
    int Nx[3] = {60, 60, 60};  // 100x100x100 grid points
    float L[3] = {60.0f, 60.0f, 60.0f};  // Grid size in each dimension
    int ntypes = 2;  // Molecule types
    float r = atof(argv[1]);
    float sigma = atof(argv[2]);  // Width of the smooth transition function


    // Make output file names
    char sphere_file[1028];
    sprintf(sphere_file, "sphere_density_radius_%.1f_sigma_%.1f.bin", r, sigma);
    char cylinder_file[1028];
    sprintf(cylinder_file, "cylinder_density_radius_%.1f_sigma_%.1f.bin", r, sigma);

    create_sphere_density(sphere_file, Dim, Nx, L, ntypes, r, sigma);

    // Create a cylinder with radius r_c and height h
    create_cylinder_density(cylinder_file, Dim, Nx, L, ntypes, r, sigma);

    return 0;
}
