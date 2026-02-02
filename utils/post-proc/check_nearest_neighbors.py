#!/usr/bin/env python3
"""
Quick check of nearest-neighbor distances to verify particle sizes
Usage: python check_nearest_neighbors.py <trajectory.gsd>
"""

import numpy as np
import gsd.hoomd
import sys

def pbc_distance(r1, r2, box):
    """Calculate distance with periodic boundary conditions"""
    dr = r1 - r2
    dr = dr - box * np.round(dr / box)
    return np.sqrt(np.sum(dr**2))

def check_nearest_neighbors(traj_file, n_frames=10):
    """Check nearest-neighbor distances for each particle type pair"""

    print(f"Reading: {traj_file}")
    traj = gsd.hoomd.open(traj_file, 'r')

    # Get particle information from first frame
    frame0 = traj[0]
    typeid = frame0.particles.typeid
    n_types = len(np.unique(typeid))

    print(f"Number of particles: {len(typeid)}")
    print(f"Number of types: {n_types}")

    # Create index lists for each type
    type_indices = {}
    for t in range(n_types):
        type_indices[t] = np.where(typeid == t)[0]
        print(f"  Type {t+1}: {len(type_indices[t])} particles")

    # Sample frames evenly throughout trajectory
    frame_indices = np.linspace(len(traj)//2, len(traj)-1, min(n_frames, len(traj)//2), dtype=int)

    # Store minimum distances for each type pair
    min_distances = {(t1, t2): [] for t1 in range(n_types) for t2 in range(t1, n_types)}

    print(f"\nAnalyzing {len(frame_indices)} frames...")

    for idx, frame_idx in enumerate(frame_indices):
        frame = traj[frame_idx]
        positions = frame.particles.position
        box = frame.configuration.box[:3]

        for type1 in range(n_types):
            for type2 in range(type1, n_types):
                indices1 = type_indices[type1]
                indices2 = type_indices[type2]

                min_dist = float('inf')

                for i in indices1:
                    for j in indices2:
                        if type1 == type2 and j <= i:
                            continue

                        dist = pbc_distance(positions[i], positions[j], box)
                        min_dist = min(min_dist, dist)

                min_distances[(type1, type2)].append(min_dist)

        if idx % max(1, len(frame_indices)//5) == 0:
            print(f"  Processed {idx+1}/{len(frame_indices)} frames")

    # Print results
    print("\n" + "="*60)
    print("NEAREST-NEIGHBOR DISTANCE ANALYSIS")
    print("="*60)

    expected_distances = {
        (0, 0): 2.0,   # Polymer-Polymer: 2 × 1.0
        (0, 1): 4.5,   # Polymer-NP: 1.0 + 3.5
        (1, 1): 7.0    # NP-NP: 2 × 3.5
    }

    pair_names = {
        (0, 0): "Polymer-Polymer",
        (0, 1): "Polymer-NP",
        (1, 1): "NP-NP"
    }

    for (type1, type2), distances in sorted(min_distances.items()):
        if not distances:
            continue

        avg_min = np.mean(distances)
        std_min = np.std(distances)
        abs_min = np.min(distances)

        name = pair_names.get((type1, type2), f"Type {type1+1}-{type2+1}")
        expected = expected_distances.get((type1, type2), "N/A")

        print(f"\n{name}:")
        print(f"  Expected contact distance: {expected}")
        print(f"  Average minimum distance:  {avg_min:.3f} ± {std_min:.3f}")
        print(f"  Absolute minimum:          {abs_min:.3f}")

        if isinstance(expected, float):
            deviation = avg_min - expected
            percent_dev = 100 * deviation / expected
            print(f"  Deviation from expected:   {deviation:+.3f} ({percent_dev:+.1f}%)")

            if abs(percent_dev) > 10:
                print(f"  ⚠️  WARNING: Large deviation from expected!")
            elif abs_min < expected * 0.9:
                print(f"  ⚠️  WARNING: Particles penetrating expected hard core!")
            else:
                print(f"  ✓ Looks reasonable")

    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_nearest_neighbors.py <trajectory.gsd> [n_frames]")
        sys.exit(1)

    traj_file = sys.argv[1]
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    check_nearest_neighbors(traj_file, n_frames)
