#!/usr/bin/env python3
"""
Calculate Radial Distribution Function (RDF) from GSD trajectory files
Usage: python calc_rdf.py <trajectory.gsd> [output.dat]
"""

import numpy as np
import gsd.hoomd
import sys
from collections import defaultdict

def calculate_rdf(positions, box, type1_indices, type2_indices, rmax, nbins, volume):
    """
    Calculate RDF between two sets of particles

    Parameters:
    -----------
    positions : array of shape (N, 3)
        Particle positions
    box : array of shape (3,)
        Box dimensions
    type1_indices : array
        Indices of particles of type 1
    type2_indices : array
        Indices of particles of type 2
    rmax : float
        Maximum distance for RDF
    nbins : int
        Number of bins
    volume : float
        Box volume
    """
    dr = rmax / nbins
    hist = np.zeros(nbins)

    # Apply periodic boundary conditions
    def pbc_dist(r1, r2, box):
        dr = r1 - r2
        # Minimum image convention
        dr = dr - box * np.round(dr / box)
        return np.sqrt(np.sum(dr**2))

    # Calculate all pair distances
    n_pairs = 0
    for i in type1_indices:
        for j in type2_indices:
            if type1_indices is type2_indices and j <= i:
                # Avoid double counting for same-type RDF
                continue

            dist = pbc_dist(positions[i], positions[j], box)
            if dist < rmax:
                bin_idx = int(dist / dr)
                if bin_idx < nbins:
                    hist[bin_idx] += 1
                    n_pairs += 1

    # Normalize RDF
    r = np.linspace(dr/2, rmax - dr/2, nbins)

    # Number density
    if type1_indices is type2_indices:
        # Same type: N*(N-1)/2 pairs
        n_pairs_total = len(type1_indices) * (len(type1_indices) - 1) / 2
        rho = len(type2_indices) / volume
    else:
        # Different types: N1*N2 pairs
        n_pairs_total = len(type1_indices) * len(type2_indices)
        rho = len(type2_indices) / volume

    # Shell volume: 4πr²dr
    shell_volume = 4 * np.pi * r**2 * dr

    # g(r) = (pairs in shell) / (expected pairs in shell)
    # Expected pairs = N * rho * shell_volume
    expected = rho * shell_volume * len(type1_indices)

    rdf = np.zeros_like(r)
    mask = expected > 0
    rdf[mask] = hist[mask] / expected[mask]

    return r, rdf


def main():
    if len(sys.argv) < 2:
        print("Usage: python calc_rdf.py <trajectory.gsd> [output.dat]")
        sys.exit(1)

    traj_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "rdf.dat"

    print(f"Reading trajectory: {traj_file}")

    try:
        traj = gsd.hoomd.open(traj_file, 'r')
    except Exception as e:
        print(f"Error opening file: {e}")
        sys.exit(1)

    # Parameters
    rmax = 15.0  # Maximum distance for RDF
    nbins = 300  # Number of bins
    skip_frames = 10  # Skip initial frames for equilibration

    # Get particle types
    frame0 = traj[0]
    typeid = frame0.particles.typeid
    n_types = len(np.unique(typeid))

    print(f"Number of particles: {len(typeid)}")
    print(f"Number of types: {n_types}")
    print(f"Analyzing {len(traj) - skip_frames} frames (skipping first {skip_frames})")

    # Initialize RDF arrays for each type pair
    rdf_accum = {}
    pair_names = {}

    # Create index lists for each type
    type_indices = {}
    for t in range(n_types):
        type_indices[t] = np.where(typeid == t)[0]
        print(f"  Type {t+1}: {len(type_indices[t])} particles")

    # Calculate RDF for each type pair
    for type1 in range(n_types):
        for type2 in range(type1, n_types):
            key = (type1, type2)
            rdf_accum[key] = np.zeros(nbins)
            pair_names[key] = f"{type1+1}-{type2+1}"

    # Accumulate RDF over frames
    n_frames = 0
    for frame_idx in range(skip_frames, len(traj)):
        frame = traj[frame_idx]
        positions = frame.particles.position
        box = frame.configuration.box[:3]  # Get x, y, z dimensions
        volume = np.prod(box)

        for (type1, type2), name in pair_names.items():
            r, rdf = calculate_rdf(
                positions, box,
                type_indices[type1],
                type_indices[type2],
                rmax, nbins, volume
            )
            rdf_accum[(type1, type2)] += rdf

        n_frames += 1
        if (frame_idx - skip_frames) % 10 == 0:
            print(f"  Processed frame {frame_idx}/{len(traj)}")

    # Average and save
    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        f.write("# Radial Distribution Function\n")
        f.write(f"# Averaged over {n_frames} frames\n")
        f.write("# r")
        for name in pair_names.values():
            f.write(f"  g_{name}(r)")
        f.write("\n")

        for i in range(nbins):
            f.write(f"{r[i]:.6f}")
            for key in sorted(rdf_accum.keys()):
                avg_rdf = rdf_accum[key][i] / n_frames
                f.write(f"  {avg_rdf:.6f}")
            f.write("\n")

    print("Done!")
    print("\nPlot with: gnuplot -e \"plot 'rdf.dat' u 1:2 w l title '1-1', '' u 1:3 w l title '1-2', '' u 1:4 w l title '2-2'\"")

if __name__ == "__main__":
    main()
