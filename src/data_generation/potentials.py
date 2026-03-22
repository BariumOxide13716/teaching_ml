from data_generation.gases import VDWGas
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


def gaussian(
        rs: np.ndarray,
        A: float = 1.0,
        r_0: float = 1.0,
        sigma: float = 1.0
) -> np.ndarray:
    """
    A function that generates a Gaussian potential
    given a list of distances
    Input:
    rs: an array of distances (np.ndarray of doubles)
    A: amplitude of the Gaussian
    r_0: center of the Gaussian
    sigma: width of the Gaussian
    Output:
    enes: an array of Gaussian energies (np.ndarray of doubles)
    """
    return np.array([A * np.exp(-((r - r_0)**2) / (2 * sigma**2)) for r in rs])


def linear(
        rs: np.ndarray,
        slope: float = 1.0,
        intercept: float = 0.0
) -> np.ndarray:
    """
    A function that generates a linear potential
    given a list of distances
    Input:
    rs: an array of distances (np.ndarray of doubles)
    slope: slope of the linear function
    intercept: intercept of the linear function
    Output:
    enes: an array of linear energies (np.ndarray of doubles)
    """
    return np.array([slope * r + intercept for r in rs])


def vdwgass(
        rs: np.ndarray,
        a: float = 1.0,
        b: float = 0.0,
        t: float = 1.0
) -> np.ndarray:
    """
    A function that generates the van der Waals gas potential
    given a list of distances
    Input:
    rs: an array of distances (np.ndarray of doubles)
    a: van der Waals constant a
    b: van der Waals constant b
    t: temperature
    Output:
    enes: an array of van der Waals energies (np.ndarray of doubles)
    """
    vdw_gas = VDWGas(a=a, b=b)
    return vdw_gas.vt_p(v_list=rs, t=t)