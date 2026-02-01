#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkFlyingEdges3D.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkPolyDataWriter.h>
#include <Eigen/Dense> // Include Eigen header for matrices and vectors
#include <omp.h>

using namespace Eigen; // Use Eigen namespace
using namespace std;

#define PI 3.141592653589793238462643383

void unstack(int id, int* nn, int* Nx, int Dim);
vector<float> read_density_grid(const string& filename, int& Dim, int* Nx, float* L, int& ntypes, int frame_to_read);
void gaussian_smoothing(vector<double>& volume_fraction, int* Nx, float sigma);
double trilinear_interpolate(const vector<double>& volume_fraction, int* Nx, double x, double y, double z);
void calculate_curvature(vtkSmartPointer<vtkPolyData> polyData, vector<double>& gaussian_curvature, vector<double>& mean_curvature, vector<double>& condition_numbers,const vector<double>& volume_fraction, int* Nx, int order);
void write_curvature_data(const std::string& filename, vtkSmartPointer<vtkPolyData> polyData, const vector<double>& gaussian_curvature, const vector<double>& mean_curvature,const vector<double>& condition_numbers); 
double cubicInterpolate(const double p[4], double t);
double tricubic_interpolate(const vector<double>& volume_fraction, int* Nx, double x, double y, double z); 

int main(int argc, char** argv) {

    auto start = std::chrono::high_resolution_clock::now();
    if (argc < 2) {
        cout << "Usage: prl-curvature-calc [input.bin] [output_name] [frame_to_read] [order of calc] [sigma for smoothing]" << endl;
        return 1;
    }

    int Dim, Nx[3], ntypes, order;
    float L[3], sigma ;
    string input_filename = argv[1];
    string output_filename = argv[2];
    cout << "Order of calculation: " << argv[4] << endl;
    int frame_to_read = atoi(argv[3]);
    if (argc >= 5) {
        order = atoi(argv[4]);
    }
    if (argc == 6) {
        sigma = atof(argv[5]);
    }
    // Make output file names
    char curvature_file[1028];
    sprintf(curvature_file, "%s%02d_order_%d_sigma_%.1f_curvature.csv", argv[2], frame_to_read, order, sigma);
    char contour_file[1028];
    sprintf(contour_file, "%s%02d_order_%d_sigma_%.1f_contour.vtk", argv[2], frame_to_read, order, sigma);

    vector<float> all_rho = read_density_grid(input_filename, Dim, Nx, L, ntypes, frame_to_read);
    if (all_rho.empty()) {
        cout << "Failed to read density grid data." << endl;
        return 1;
    }

    int M = 1;
    for (int i = 0; i < Dim; i++) {
        M *= Nx[i];
    }

    vector<double> volume_fraction(M);
    for (int i = 0; i < M; i++) {
        double sum_rho = 0.0;
        for (int j = 0; j < ntypes; j++) {
            sum_rho += all_rho[j * M + i];
    }
        volume_fraction[i] = all_rho[i] / sum_rho; // Assuming the first molecule type for volume fraction
    }

    // Apply Gaussian smoothing with periodic boundary conditions
    if (sigma > 0.0){
        gaussian_smoothing(volume_fraction, Nx, sigma);
    }

    // VTK marching cubes
    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(Nx[0], Nx[1], Nx[2]);
    imageData->AllocateScalars(VTK_DOUBLE, 1);
    for (int i = 0; i < M; i++) {
        int nn[3];
        unstack(i, nn, Nx, Dim);
        double* pixel = static_cast<double*>(imageData->GetScalarPointer(nn[0], nn[1], nn[2]));
        *pixel = volume_fraction[i];
    }

    vtkSmartPointer<vtkMarchingCubes> marchingCubes = vtkSmartPointer<vtkMarchingCubes>::New();
    marchingCubes->SetInputData(imageData);
    marchingCubes->ComputeNormalsOn();
    marchingCubes->SetValue(0, 0.500); // Isovalue for the isocontour

    marchingCubes->Update();
    vtkSmartPointer<vtkPolyData> contour = marchingCubes->GetOutput();
    // vtkSmartPointer<vtkFlyingEdges3D> flyingEdges = vtkSmartPointer<vtkFlyingEdges3D>::New();
    // flyingEdges->SetInputData(imageData);
    // flyingEdges->SetValue(0, 0.500);  // Isovalue for the isocontour
    // flyingEdges->Update();
    // vtkSmartPointer<vtkPolyData> contour = flyingEdges->GetOutput();

    // Smoothing the contour
    // vtkSmartPointer<vtkWindowedSincPolyDataFilter> smoother = vtkSmartPointer<vtkWindowedSincPolyDataFilter>::New();
    // smoother->SetInputData(contour); // 'contour' is the vtkPolyData from FlyingEdges
    // smoother->SetNumberOfIterations(20);
    // smoother->SetPassBand(0.1);  // Controls the smoothness
    // //smoother->BoundarySmoothingOff();
    // smoother->FeatureEdgeSmoothingOn();
    // smoother->SetFeatureAngle(90.0);
    // smoother->Update();

    // vtkSmartPointer<vtkPolyData> smoothedContour = smoother->GetOutput();

    // Calculate curvature
    vector<double> gaussian_curvature, mean_curvature, condition_numbers;
    calculate_curvature(contour, gaussian_curvature, mean_curvature, condition_numbers, volume_fraction, Nx, order);

    // Write curvature data to a file
    write_curvature_data(curvature_file, contour, gaussian_curvature, mean_curvature, condition_numbers);

    // Optional: Write the contour to a file
    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetFileName(contour_file);
    writer->SetInputData(contour);
    writer->Write();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Elapsed time: " << elapsed.count() << " s" << endl;
    
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
    cout << "Reading density grid data for frame: " << frame_to_read << endl;
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
        nn[1] = (id / Nx[0])- nn[2]*Nx[1];
        nn[0] = id - (nn[1] + nn[2]*Nx[1])*Nx[0];
    }
}

/**
 * Applies Gaussian smoothing to a volume fraction vector.
 *
 * @param volume_fraction The input volume fraction vector to be smoothed.
 * @param Nx An array containing the dimensions of the volume.
 * @param sigma The standard deviation of the Gaussian kernel.
 */
void gaussian_smoothing(vector<double>& volume_fraction, int* Nx, float sigma) {
    int M = Nx[0] * Nx[1] * Nx[2];
    vector<double> smoothed(M, 0.0);

    int radius = static_cast<int>(3.0 * sigma);
    int size = 2 * radius + 1;
    vector<double> kernel(size);

    double sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        kernel[i + radius] = exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + radius];
    }
    for (auto& k : kernel) {
        k /= sum;
    }

    // Apply the Gaussian kernel in each dimension with periodic boundary conditions
    #pragma omp parallel for 
    for (int i = 0; i < M ; i++){
        int nn[3];
        unstack(i, nn, Nx, 3);
        int x = nn[0], y = nn[1], z = nn[2];
        double value = 0.0;
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
        smoothed[i] = value;
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
void calculate_curvature(vtkSmartPointer<vtkPolyData> polyData, vector<double>& gaussian_curvature, vector<double>& mean_curvature, vector<double>& condition_numbers, const vector<double>& volume_fraction, int* Nx, int order) {
    vtkSmartPointer<vtkPoints> points = polyData->GetPoints();
    vtkSmartPointer<vtkCellArray> cells = polyData->GetPolys();

    int numPoints = points->GetNumberOfPoints();
    gaussian_curvature.resize(numPoints, 0.0);
    mean_curvature.resize(numPoints, 0.0);
    condition_numbers.resize(numPoints, 0.0);

    // Parameters for the Diffuse Approximation method
    int Nunknw = 10;
    if (order == 3) {
        Nunknw = 20;
    }
    else if (order == 4) {
        Nunknw = 35;
    }
    else if (order > 4 || order < 2) {
        cout << "Invalid order of calculation: "  << order << endl;
        return;
    }
    const int Nmax = 3;
    const int Nmin = -3;
    const int Nneigh = (Nmax - Nmin + 1) * (Nmax - Nmin + 1) * (Nmax - Nmin + 1);

    // Containers for storing curvature values
    MatrixXd P = MatrixXd::Zero(Nneigh, Nunknw);
    MatrixXd A = MatrixXd::Zero(Nunknw, Nunknw);
    MatrixXd B = MatrixXd::Zero(Nunknw, Nneigh);

    VectorXd C = VectorXd::Zero(Nunknw);
    VectorXd W = VectorXd::Zero(Nneigh);
    VectorXd F = VectorXd::Zero(Nneigh);

    // Calculate curvature at every vertex
    #pragma omp parallel for
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
                    double x = fmod((Ox + b1 + Nx[0]), Nx[0]);
                    double y = fmod((Oy + b2 + Nx[1]), Nx[1]);
                    double z = fmod((Oz + b3 + Nx[2]), Nx[2]);

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
                    if (order == 3){
                        P(neigh_count, 10) = b1 * b1 * b1;
                        P(neigh_count, 11) = b2 * b2 * b2;
                        P(neigh_count, 12) = b3 * b3 * b3;
                        P(neigh_count, 13) = b1 * b1 * b2;
                        P(neigh_count, 14) = b1 * b1 * b3;
                        P(neigh_count, 15) = b1 * b2 * b2;
                        P(neigh_count, 16) = b2 * b2 * b3;
                        P(neigh_count, 17) = b1 * b3 * b3;
                        P(neigh_count, 18) = b2 * b3 * b3;
                        P(neigh_count, 19) = b1 * b2 * b3;
                    }
                    else if (order == 4)
                    {
                        P(neigh_count, 10) = b1 * b1 * b1;
                        P(neigh_count, 11) = b2 * b2 * b2;
                        P(neigh_count, 12) = b3 * b3 * b3;
                        P(neigh_count, 13) = b1 * b1 * b2;
                        P(neigh_count, 14) = b1 * b1 * b3;
                        P(neigh_count, 15) = b1 * b2 * b2;
                        P(neigh_count, 16) = b2 * b2 * b3;
                        P(neigh_count, 17) = b1 * b3 * b3;
                        P(neigh_count, 18) = b2 * b3 * b3;
                        P(neigh_count, 19) = b1 * b2 * b3;
                        P(neigh_count, 20) = b1 * b1 * b1 * b1;
                        P(neigh_count, 21) = b2 * b2 * b2 * b2;
                        P(neigh_count, 22) = b3 * b3 * b3 * b3;
                        P(neigh_count, 23) = b1 * b1 * b1 * b2;
                        P(neigh_count, 24) = b1 * b1 * b1 * b3;
                        P(neigh_count, 25) = b1 * b2 * b2 * b2;
                        P(neigh_count, 26) = b2 * b2 * b2 * b3;
                        P(neigh_count, 27) = b1 * b3 * b3 * b3;
                        P(neigh_count, 28) = b2 * b3 * b3 * b3;
                        P(neigh_count, 29) = b1 * b1 * b2 * b2;
                        P(neigh_count, 30) = b1 * b1 * b3 * b3;
                        P(neigh_count, 31) = b2 * b2 * b3 * b3;
                        P(neigh_count, 32) = b1 * b1 * b2 * b3;
                        P(neigh_count, 33) = b1 * b2 * b2 * b3;
                        P(neigh_count, 34) = b1 * b2 * b3 * b3;

                    }
                    
                    // Get interpolated value from volume_fraction
                    F(neigh_count) = trilinear_interpolate(volume_fraction, Nx, x, y, z);
                    //F(neigh_count) = tricubic_interpolate(volume_fraction, Nx, x, y, z);
                    W(neigh_count) = exp(-(b1 * b1) - (b2 * b2) - (b3 * b3));

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
        // Compute condition number of A matrix
        JacobiSVD<MatrixXd> svd(A);
        double cond_number = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
        condition_numbers[idx] = cond_number;
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
double trilinear_interpolate(const vector<double>& volume_fraction, int* Nx, double x, double y, double z) {
    int Nx0 = Nx[0], Nx1 = Nx[1], Nx2 = Nx[2];
    // Calculate indices, ensuring they wrap around and are non-negative
    int i = static_cast<int>(x);
    int j = static_cast<int>(y);
    int k = static_cast<int>(z);
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
 * Performs cubic interpolation for one dimension.
 *
 * @param p Array of 4 values (control points) along one dimension.
 * @param t Fractional distance along the dimension (0 <= t <= 1).
 * @return Interpolated value.
 */
double cubicInterpolate(const double p[4], double t) {
    return p[1] + 0.5 * t * (p[2] - p[0] +
             t * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] +
             t * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}

/**
 * Performs tricubic interpolation on a given volume fraction.
 *
 * @param volume_fraction The volume fraction data.
 * @param Nx The size of the volume in each dimension.
 * @param x The x-coordinate of the point to interpolate.
 * @param y The y-coordinate of the point to interpolate.
 * @param z The z-coordinate of the point to interpolate.
 * @return The interpolated value at the given point.
 */
double tricubic_interpolate(const vector<double>& volume_fraction, int* Nx, double x, double y, double z) {
    int Nx0 = Nx[0], Nx1 = Nx[1], Nx2 = Nx[2];

    // Calculate integer indices surrounding the point (x, y, z)
    int i = static_cast<int>(x);
    int j = static_cast<int>(y);
    int k = static_cast<int>(z);

    // Compute fractional distances within the grid cell
    double tx = x - i;
    double ty = y - j;
    double tz = z - k;

    // Helper lambda to wrap around boundaries (Periodic Boundary Conditions)
    auto wrap = [](int idx, int max) {
        return (idx + max) % max;
    };

    // Extract a 4x4x4 neighborhood of values
    double values[4][4][4];
    for (int dz = 0; dz < 4; ++dz) {
        for (int dy = 0; dy < 4; ++dy) {
            for (int dx = 0; dx < 4; ++dx) {
                int x_idx = wrap(i + dx - 1, Nx0);
                int y_idx = wrap(j + dy - 1, Nx1);
                int z_idx = wrap(k + dz - 1, Nx2);
                values[dz][dy][dx] = volume_fraction[x_idx + Nx0 * (y_idx + Nx1 * z_idx)];
            }
        }
    }

    // Perform cubic interpolation along the x-axis for each yz-plane
    double temp[4][4];
    #pragma omp parallel for collapse(2)
    for (int dz = 0; dz < 4; ++dz) {
        for (int dy = 0; dy < 4; ++dy) {
            temp[dz][dy] = cubicInterpolate(values[dz][dy], tx);
        }
    }

    // Perform cubic interpolation along the y-axis for each z-plane
    double temp2[4];
    #pragma omp parallel for
    for (int dz = 0; dz < 4; ++dz) {
        temp2[dz] = cubicInterpolate(temp[dz], ty);
    }

    // Perform cubic interpolation along the z-axis
    return cubicInterpolate(temp2, tz);
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
void write_curvature_data(const std::string& filename, vtkSmartPointer<vtkPolyData> polyData, const vector<double>& gaussian_curvature, const vector<double>& mean_curvature, const vector<double>& condition_numbers) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    file << "TriangleID, VertexID1, VertexID2, VertexID3, X1, Y1, Z1, GaussianCurvature1, MeanCurvature1, ConditionNumber1,  X2, Y2, Z2, GaussianCurvature2, MeanCurvature2, ConditionNumber2, X3, Y3, Z3, GaussianCurvature3, MeanCurvature3, ConditionNumber3\n";

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
            file << ", " << p[0] << ", " << p[1] << ", " << p[2] << ", " << gaussian_curvature[ptIds[i]] << ", " << mean_curvature[ptIds[i]] << ", " << condition_numbers[ptIds[i]];
        }

        file << "\n";
        triangleID++;
    }

    file.close();
}
