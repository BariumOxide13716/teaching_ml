
import numpy as np

def lennardjones(
        rs: np.ndarray,
        epsilon: float = 1.0,
        sigma: float = 1.0
) -> np.ndarray:
    
    """
    A function that generates the Lennard-Jones potential
    given a list of distances
    Input:
    rs: an array of distances (np.ndarray of doubles)
    epsilon: depth of the potential well
    sigma: finite distance at which the inter-particle potential is zero
    Output:
    enes: an array of LJ energies (np.ndarray of doubles)
    """
    return np.array([4.0 * epsilon * (sigma / r)**12 - (sigma / r)**6 for r in rs])

def morse(
        rs: np.ndarray,
        D_e: float = 1.0,
        alpha: float = 1.0,
        r_e: float = 1.0
) -> np.ndarray:
    """
    A function that generates the Morse potential
    given a list of distances
    Input:
    rs: an array of distances (np.ndarray of doubles)
    D_e: depth of the potential well
    alpha: width of the potential well
    r_e: equilibrium bond distance
    Output:
    enes: an array of Morse energies (np.ndarray of doubles)
    """
    return np.array([D_e * (1 - np.exp(-alpha * (r - r_e)))**2 for r in rs])
