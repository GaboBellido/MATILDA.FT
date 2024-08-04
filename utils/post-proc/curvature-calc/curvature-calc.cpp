#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkPolyDataWriter.h>
#include <Eigen/Dense> // Include Eigen header for matrices and vectors

using namespace Eigen; // Use Eigen namespace
using namespace std;

#define PI 3.141592653589793238462643383

void unstack(int id, int* nn, int* Nx, int Dim);
vector<float> read_density_grid(const string& filename, int& Dim, int* Nx, float* L, int& ntypes, int frame_to_read);
void gaussian_smoothing(vector<float>& volume_fraction, int* Nx, float sigma);
double trilinear_interpolate(const vector<float>& volume_fraction, int* Nx, double x, double y, double z);
void calculate_curvature(vtkSmartPointer<vtkPolyData> polyData, vector<double>& gaussian_curvature, vector<double>& mean_curvature, const vector<float>& volume_fraction, int* Nx);
void write_curvature_data(const std::string& filename, vtkSmartPointer<vtkPolyData> polyData, const vector<double>& gaussian_curvature, const vector<double>& mean_curvature); 


int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: curvature-calc [input.bin] [output_name] [frame_to_read]" << endl;
        return 1;
    }

    int Dim, Nx[3], ntypes, frame_to_read = 0;
    float L[3];
    string input_filename = argv[1];

    if (argc == 3) {
        frame_to_read = atoi(argv[3]);
    }
    // Make output file names
    char curvature_file[200];
    sprintf(curvature_file, "%s%03d_curvature.csv", argv[2], frame_to_read);
    char contour_file[200];
    sprintf(contour_file, "%s%03d_contour.vtk", argv[2], frame_to_read);

    vector<float> all_rho = read_density_grid(input_filename, Dim, Nx, L, ntypes, frame_to_read);
    if (all_rho.empty()) {
        cout << "Failed to read density grid data." << endl;
        return 1;
    }

    int M = 1;
    for (int i = 0; i < Dim; i++) {
        M *= Nx[i];
    }

    vector<float> volume_fraction(M);
    for (int i = 0; i < M; i++) {
        float sum_rho = 0.0;
        for (int j = 0; j < ntypes; j++) {
            sum_rho += all_rho[j * M + i];
    }
        volume_fraction[i] = all_rho[i] / sum_rho; // Assuming the first molecule type for volume fraction
    }

    // Apply Gaussian smoothing with periodic boundary conditions
    gaussian_smoothing(volume_fraction, Nx, 1.0);

    // VTK marching cubes
    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(Nx[0], Nx[1], Nx[2]);
    imageData->AllocateScalars(VTK_FLOAT, 1);

    for (int z = 0; z < Nx[2]; z++) {
        for (int y = 0; y < Nx[1]; y++) {
            for (int x = 0; x < Nx[0]; x++) {
                float* pixel = static_cast<float*>(imageData->GetScalarPointer(x, y, z));
                *pixel = volume_fraction[x + Nx[0] * (y + Nx[1] * z)];
            }
        }
    }

    vtkSmartPointer<vtkMarchingCubes> marchingCubes = vtkSmartPointer<vtkMarchingCubes>::New();
    marchingCubes->SetInputData(imageData);
    marchingCubes->ComputeNormalsOn();
    marchingCubes->SetValue(0, 0.5); // Isovalue for the isocontour

    marchingCubes->Update();
    vtkSmartPointer<vtkPolyData> contour = marchingCubes->GetOutput();
    // Calculate curvature
    vector<double> gaussian_curvature, mean_curvature;
    calculate_curvature(contour, gaussian_curvature, mean_curvature, volume_fraction, Nx);

    // Write curvature data to a file
    write_curvature_data(curvature_file, contour, gaussian_curvature, mean_curvature);

    // Optional: Write the contour to a file
    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetFileName(contour_file);
    writer->SetInputData(contour);
    writer->Write();

    return 0;
}

/**
 * Reads the density grid from a binary file.
 * 
 * @param filename The path to the binary file.
 * @param Dim The dimension of the grid (output parameter).
 * @param Nx The number of grid points in each dimension (output parameter).
 * @param L The grid size in each dimension (output parameter).
 * @param ntypes The number of particle types (output parameter).
 * @param frame_to_read The index of the frame to read from the file.
 * @return A vector containing the density values for all particle types in the specified frame.
 *         If the file cannot be opened or the specified frame is not found, an empty vector is returned.
 */
vector<float> read_density_grid(const string& filename, int& Dim, int* Nx, float* L, int& ntypes, int frame_to_read) {
    FILE* inp = fopen(filename.c_str(), "rb");
    if (!inp) {
        cout << "Failed to open " << filename << endl;
        return {};
    }

    fread(&Dim, sizeof(int), 1, inp);
    fread(Nx, sizeof(int), Dim, inp);
    fread(L, sizeof(float), Dim, inp);
    fread(&ntypes, sizeof(int), 1, inp);

    int M = 1;
    for (int i = 0; i < Dim; i++) {
        M *= Nx[i];
    }

    vector<float> all_rho(ntypes * M);
    int nframes = 0;
    bool frame_found = false;

    while (!feof(inp)) {
        int rt = fread(all_rho.data(), sizeof(float), M * ntypes, inp);
        if (rt != M * ntypes) {
            break;
        }
        if (nframes == frame_to_read) {
            frame_found = true;
            break;
        }
        nframes++;
    }

    fclose(inp);

    if (!frame_found) {
        cout << "Specified frame not found." << endl;
        return {};
    }

    return all_rho;
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
    }
}

/**
 * Applies Gaussian smoothing to a volume fraction vector.
 *
 * @param volume_fraction The input volume fraction vector to be smoothed.
 * @param Nx An array containing the dimensions of the volume.
 * @param sigma The standard deviation of the Gaussian kernel.
 */
void gaussian_smoothing(vector<float>& volume_fraction, int* Nx, float sigma) {
    int M = Nx[0] * Nx[1] * Nx[2];
    vector<float> smoothed(M, 0.0);

    int radius = static_cast<int>(1.0 * sigma);
    int size = 2 * radius + 1;
    vector<float> kernel(size);

    float sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        kernel[i + radius] = exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + radius];
    }
    for (auto& k : kernel) {
        k /= sum;
    }

    // Apply the Gaussian kernel in each dimension with periodic boundary conditions
    for (int z = 0; z < Nx[2]; z++) {
        for (int y = 0; y < Nx[1]; y++) {
            for (int x = 0; x < Nx[0]; x++) {
                float value = 0.0;
                for (int dz = -radius; dz <= radius; dz++) {
                    int zz = (z + dz + Nx[2]) % Nx[2];
                    for (int dy = -radius; dy <= radius; dy++) {
                        int yy = (y + dy + Nx[1]) % Nx[1];
                        for (int dx = -radius; dx <= radius; dx++) {
                            int xx = (x + dx + Nx[0]) % Nx[0];
                            value += volume_fraction[xx + Nx[0] * (yy + Nx[1] * zz)] * kernel[dx + radius] * kernel[dy + radius] * kernel[dz + radius];
                        }
                    }
                }
                smoothed[x + Nx[0] * (y + Nx[1] * z)] = value;
            }
        }
    }

    volume_fraction = smoothed;
}

/**
 * Calculates the Gaussian and mean curvature at each vertex of a given vtkPolyData object.
 *
 * @param polyData The vtkPolyData object representing the surface mesh.
 * @param gaussian_curvature Reference to a vector to store the calculated Gaussian curvature values.
 * @param mean_curvature Reference to a vector to store the calculated mean curvature values.
 * @param volume_fraction The volume fraction values used for interpolation.
 * @param Nx Array containing the dimensions of the volume fraction grid.
 */
void calculate_curvature(vtkSmartPointer<vtkPolyData> polyData, vector<double>& gaussian_curvature, vector<double>& mean_curvature, const vector<float>& volume_fraction, int* Nx) {
    vtkSmartPointer<vtkPoints> points = polyData->GetPoints();
    vtkSmartPointer<vtkCellArray> cells = polyData->GetPolys();

    int numPoints = points->GetNumberOfPoints();
    gaussian_curvature.resize(numPoints, 0.0);
    mean_curvature.resize(numPoints, 0.0);

    // Parameters for the Diffuse Approximation method
    const int Nunknw = 10;
    const int Nmin = -1;
    const int Nmax = 1;
    const int Nneigh = (Nmax - Nmin + 1) * (Nmax - Nmin + 1) * (Nmax - Nmin + 1);

    // Containers for storing curvature values
    MatrixXd P(Nneigh, Nunknw);
    MatrixXd A = MatrixXd::Zero(Nunknw, Nunknw);
    MatrixXd B(Nunknw, Nneigh);
    VectorXd C = VectorXd::Zero(Nunknw);
    VectorXd W = VectorXd::Zero(Nneigh);
    VectorXd F = VectorXd::Zero(Nneigh);

    // Calculate curvature at every vertex
    for (vtkIdType idx = 0; idx < numPoints; ++idx) {
        double vert[3];
        points->GetPoint(idx, vert);
        double Ox = vert[0], Oy = vert[1], Oz = vert[2];

        // Neighbor counter
        int neigh_count = 0;

        // Loop through neighbors of isocontour point
        for (int k = Nmin; k <= Nmax; k++) {
            double b3 = k;
            for (int j = Nmin; j <= Nmax; j++) {
                double b2 = j;
                for (int i = Nmin; i <= Nmax; i++) {
                    double b1 = i;

                    // Extract neighbor coordinates while accounting for PBC
                    double x = Ox + b1;
                    double y = Oy + b2;
                    double z = Oz + b3;

                    // Fill the P matrix
                    P(neigh_count, 0) = 1;
                    P(neigh_count, 1) = b1;
                    P(neigh_count, 2) = b2;
                    P(neigh_count, 3) = b3;
                    P(neigh_count, 4) = b1 * b1;
                    P(neigh_count, 5) = b2 * b2;
                    P(neigh_count, 6) = b3 * b3;
                    P(neigh_count, 7) = b1 * b2;
                    P(neigh_count, 8) = b1 * b3;
                    P(neigh_count, 9) = b2 * b3;

                    // Get interpolated value from volume_fraction
                    F(neigh_count) = trilinear_interpolate(volume_fraction, Nx, x, y, z);
                    W(neigh_count) = exp(-b1 * b1 - b2 * b2 - b3 * b3);

                    neigh_count++;
                }
            }
        }

        // Compute A, B, and C matrices
        for (int k = 0; k < Nneigh; k++) {
            for (int i = 0; i < Nunknw; i++) {
                B(i, k) += W(k) * P(k, i);
                C(i) += B(i, k) * F(k);
                for (int j = 0; j < Nunknw; j++) {
                    A(i, j) += P(k, i) * W(k) * P(k, j);
                }
            }
        }

        // Solve for partial derivatives
        VectorXd R = A.colPivHouseholderQr().solve(C);
        double Fx = R(1), Fy = R(2), Fz = R(3);
        double Fxx = R(4) * 2, Fyy = R(5) * 2, Fzz = R(6) * 2;
        double Fxy = R(7), Fxz = R(8), Fyz = R(9);

        // Mean Curvature calculation
        double Num_H = (Fx * Fx) * (Fyy + Fzz) - 2 * Fy * Fz * Fyz + (Fy * Fy) * (Fxx + Fzz) - 2 * Fx * Fz * Fxz + (Fz * Fz) * (Fxx + Fyy) - 2 * Fx * Fy * Fxy;
        double Den_H = pow(sqrt(Fx * Fx + Fy * Fy + Fz * Fz), 3);
        double H = Num_H / (2 * Den_H);

        // Gauss Curvature calculation
        double Num_K = (Fx * Fx) * (Fyy * Fzz - Fyz * Fyz) + 2 * Fz * Fy * (Fxy * Fxz - Fyz * Fxx) + (Fy * Fy) * (Fxx * Fzz - Fxz * Fxz) + 2 * Fx * Fz * (Fxy * Fyz - Fxz * Fyy) + (Fz * Fz) * (Fxx * Fyy - Fxy * Fxy) + 2 * Fx * Fy * (Fxz * Fyz - Fxy * Fzz);
        double Den_K = (Fx * Fx + Fy * Fy + Fz * Fz) * (Fx * Fx + Fy * Fy + Fz * Fz);
        double K = Num_K / Den_K;

        mean_curvature[idx] = H;
        gaussian_curvature[idx] = K;
    }
}

/**
 * Performs trilinear interpolation on a given volume fraction.
 *
 * @param volume_fraction The volume fraction data.
 * @param Nx The size of the volume in each dimension.
 * @param x The x-coordinate of the point to interpolate.
 * @param y The y-coordinate of the point to interpolate.
 * @param z The z-coordinate of the point to interpolate.
 * @return The interpolated value at the given point.
 */
double trilinear_interpolate(const vector<float>& volume_fraction, int* Nx, double x, double y, double z) {
    int Nx0 = Nx[0], Nx1 = Nx[1], Nx2 = Nx[2];
    // Calculate indices, ensuring they wrap around and are non-negative
    int i = ((static_cast<int>(x) % Nx0) + Nx0) % Nx0;
    int j = ((static_cast<int>(y) % Nx1) + Nx1) % Nx1;
    int k = ((static_cast<int>(z) % Nx2) + Nx2) % Nx2;
    int i_next = (i + 1) % Nx0;
    int j_next = (j + 1) % Nx1;
    int k_next = (k + 1) % Nx2;

    double Xinf = static_cast<double>(i), Xsup = Xinf + 1.0;
    double Yinf = static_cast<double>(j), Ysup = Yinf + 1.0;
    double Zinf = static_cast<double>(k), Zsup = Zinf + 1.0;

    double c000 = volume_fraction[i + Nx0 * (j + Nx1 * k)];
    double c100 = volume_fraction[i_next + Nx0 * (j + Nx1 * k)];
    double c010 = volume_fraction[i + Nx0 * (j_next + Nx1 * k)];
    double c110 = volume_fraction[i_next + Nx0 * (j_next + Nx1 * k)];
    double c001 = volume_fraction[i + Nx0 * (j + Nx1 * k_next)];
    double c101 = volume_fraction[i_next + Nx0 * (j + Nx1 * k_next)];
    double c011 = volume_fraction[i + Nx0 * (j_next + Nx1 * k_next)];
    double c111 = volume_fraction[i_next + Nx0 * (j_next + Nx1 * k_next)];

    double c00 = c000 * (Xsup - x) + c100 * (x - Xinf);
    double c10 = c010 * (Xsup - x) + c110 * (x - Xinf);
    double c01 = c001 * (Xsup - x) + c101 * (x - Xinf);
    double c11 = c011 * (Xsup - x) + c111 * (x - Xinf);

    double c0 = c00 * (Ysup - y) + c10 * (y - Yinf);
    double c1 = c01 * (Ysup - y) + c11 * (y - Yinf);

    return c0 * (Zsup - z) + c1 * (z - Zinf);
}

/**
 * @brief Writes curvature data to a file.
 * 
 * This function writes the curvature data of a vtkPolyData object to a file in a specific format.
 * The file will contain information about the triangle ID, vertex IDs, coordinates, and curvature values.
 * 
 * @param filename The name of the file to write the data to.
 * @param polyData The vtkPolyData object containing the geometry information.
 * @param gaussian_curvature A vector containing the Gaussian curvature values for each vertex.
 * @param mean_curvature A vector containing the mean curvature values for each vertex.
 */
void write_curvature_data(const std::string& filename, vtkSmartPointer<vtkPolyData> polyData, const vector<double>& gaussian_curvature, const vector<double>& mean_curvature) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    file << "TriangleID, VertexID1, VertexID2, VertexID3, X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, GaussianCurvature1, MeanCurvature1, GaussianCurvature2, MeanCurvature2, GaussianCurvature3, MeanCurvature3\n";

    vtkSmartPointer<vtkPoints> points = polyData->GetPoints();
    vtkSmartPointer<vtkCellArray> cells = polyData->GetPolys();

    vtkIdType npts;
    const vtkIdType* ptIds;
    int triangleID = 0;

    cells->InitTraversal();
    while (cells->GetNextCell(npts, ptIds)) {
        if (npts != 3) continue; // Ensure it's a triangle

        file << triangleID << ", " << ptIds[0] << ", " << ptIds[1] << ", " << ptIds[2];

        for (vtkIdType i = 0; i < npts; ++i) {
            double p[3];
            points->GetPoint(ptIds[i], p);
            file << ", " << p[0] << ", " << p[1] << ", " << p[2] << ", " << gaussian_curvature[ptIds[i]] << ", " << mean_curvature[ptIds[i]];
        }

        file << "\n";
        triangleID++;
    }

    file.close();
}