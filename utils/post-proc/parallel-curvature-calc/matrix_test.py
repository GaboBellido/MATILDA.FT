import numpy as np
import scipy.linalg as la

np.random.seed(0)  # Set seed for reproducibility

# Generate a symmetric positive-definite matrix A
Nunknw = 20  # Set the size based on your configuration
A = np.random.rand(Nunknw, Nunknw)
A = 0.5 * (A + A.T) + Nunknw * np.eye(Nunknw)  # Make it symmetric and positive-definite

# Generate a random vector C
C = np.random.rand(Nunknw)

# Solve using NumPy
solution_numpy = la.solve(A, C)
# Save matrices to files
np.savetxt("matrix_A.txt", A)
np.savetxt("vector_C.txt", C)
np.savetxt("solution_numpy.txt", solution_numpy)  # Save the expected solution for verification
