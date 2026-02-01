#include <iostream>
#include <fstream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main() {
    int Nunknw = 20;
    MatrixXd A(Nunknw, Nunknw);
    VectorXd C(Nunknw);

    // Load A matrix
    ifstream matrixAFile("matrix_A.txt");
    for (int i = 0; i < Nunknw; ++i) {
        for (int j = 0; j < Nunknw; ++j) {
            matrixAFile >> A(i, j);
        }
    }
    matrixAFile.close();

    // Load C vector
    ifstream vectorCFile("vector_C.txt");
    for (int i = 0; i < Nunknw; ++i) {
        vectorCFile >> C(i);
    }
    vectorCFile.close();

    // Solve using Eigen's colPivHouseholderQr
    VectorXd solution_cpp = A.colPivHouseholderQr().solve(C);

    // Print solution for comparison
    cout << "Solution from Eigen (C++):\n" << solution_cpp << endl;

    // Load the expected solution from NumPy
    VectorXd solution_numpy(Nunknw);
    ifstream solutionFile("solution_numpy.txt");
    for (int i = 0; i < Nunknw; ++i) {
        solutionFile >> solution_numpy(i);
    }
    solutionFile.close();

    // Compare the solutions
    double error = (solution_cpp - solution_numpy).norm();
    cout << "Difference (norm) between Eigen and NumPy solutions: " << error << endl;

    return 0;
}
