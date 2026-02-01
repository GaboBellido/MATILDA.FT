#!/usr/bin/env python3
"""
Plot the ERF and Gaussian potential profiles to visualize effective particle sizes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def gaussian_potential(r, A, sigma):
    """Gaussian potential: A * exp(-(r/sigma)^2)"""
    return A * np.exp(-(r/sigma)**2)

def erf_potential_1d(r, A, Rp, sigma):
    """
    ERF potential in 1D (simplified version for visualization)
    This is an approximation of the smeared step function interaction
    """
    # Approximate the interaction as a complementary error function
    # erfc(x) = 1 - erf(x)
    xi = np.sqrt(2) * sigma  # Width parameter
    return A * (1 - erf((r - 2*Rp) / xi))

def plot_potentials():
    """Plot potential profiles for polymer-polymer, polymer-NP, and NP-NP interactions"""

    # Distance range
    r = np.linspace(0.1, 15, 1000)

    # Your parameters
    # Polymer-Polymer: Gaussian
    A_pp = 10.0
    sigma_pp = 1.0

    # Polymer-NP: ERF with arithmetic mean radius
    A_pnp = 15.0
    Rp_pnp = 2.25  # Arithmetic mean of (1.0 + 3.5)/2
    sigma_pnp = 0.25

    # NP-NP: ERF
    A_npnp = 50.0
    Rp_npnp = 3.5
    sigma_npnp = 0.5

    # Calculate potentials
    U_pp = gaussian_potential(r, A_pp, sigma_pp)
    U_pnp = erf_potential_1d(r, A_pnp, Rp_pnp, sigma_pnp)
    U_npnp = erf_potential_1d(r, A_npnp, Rp_npnp, sigma_npnp)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: All potentials
    ax1.plot(r, U_pp, 'b-', linewidth=2, label='Polymer-Polymer (Gaussian)')
    ax1.plot(r, U_pnp, 'g-', linewidth=2, label='Polymer-NP (ERF)')
    ax1.plot(r, U_npnp, 'r-', linewidth=2, label='NP-NP (ERF)')

    # Mark expected contact distances
    ax1.axvline(2.0, color='b', linestyle='--', alpha=0.5, label='Expected: r=2.0 (P-P)')
    ax1.axvline(4.5, color='g', linestyle='--', alpha=0.5, label='Expected: r=4.5 (P-NP)')
    ax1.axvline(7.0, color='r', linestyle='--', alpha=0.5, label='Expected: r=7.0 (NP-NP)')

    # Mark thermal energy kT=1
    ax1.axhline(1.0, color='k', linestyle=':', alpha=0.5, label='kT = 1.0')

    ax1.set_xlabel('Distance r', fontsize=12)
    ax1.set_ylabel('Potential Energy U(r)', fontsize=12)
    ax1.set_title('Potential Energy Profiles', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 15)
    ax1.set_ylim(-1, max(A_npnp, A_pnp, A_pp) * 1.1)

    # Plot 2: Zoomed view near contact distances
    ax2.plot(r, U_pp, 'b-', linewidth=2, label='Polymer-Polymer')
    ax2.plot(r, U_pnp, 'g-', linewidth=2, label='Polymer-NP')
    ax2.plot(r, U_npnp, 'r-', linewidth=2, label='NP-NP')

    ax2.axvline(2.0, color='b', linestyle='--', alpha=0.5)
    ax2.axvline(4.5, color='g', linestyle='--', alpha=0.5)
    ax2.axvline(7.0, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(1.0, color='k', linestyle=':', alpha=0.5, label='kT = 1.0')

    ax2.set_xlabel('Distance r', fontsize=12)
    ax2.set_ylabel('Potential Energy U(r)', fontsize=12)
    ax2.set_title('Zoomed View (0-10)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 20)

    plt.tight_layout()
    plt.savefig('potential_profiles.png', dpi=300, bbox_inches='tight')
    print("Saved potential profiles to: potential_profiles.png")

    # Print effective hard-core distances (where U ~ 10 kT)
    print("\nEffective hard-core distances (U ≈ 10 kT):")
    threshold = 10.0

    r_pp = r[np.argmin(np.abs(U_pp - threshold))]
    r_pnp = r[np.argmin(np.abs(U_pnp - threshold))]
    r_npnp = r[np.argmin(np.abs(U_npnp - threshold))]

    print(f"  Polymer-Polymer: r ≈ {r_pp:.3f} (expected: 2.0)")
    print(f"  Polymer-NP: r ≈ {r_pnp:.3f} (expected: 4.5)")
    print(f"  NP-NP: r ≈ {r_npnp:.3f} (expected: 7.0)")

    plt.show()

if __name__ == "__main__":
    print("Plotting potential energy profiles...")
    print("\nParameters:")
    print("  Polymer radius: 1.0")
    print("  Nanoparticle radius: 3.5")
    print("  Polymer-NP ERF radius (arithmetic mean): 2.25")
    print()
    plot_potentials()
