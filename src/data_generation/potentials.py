
def lenardjones(
        rs: list[float],
        epsilon: float = 1.0,
        sigma: float = 1.0
):
    
    """
    A function that generates the Lenard-Jones potential
    given a list of distances
    Input:
    rs: list of distances
    epsilon: depth of the potential well
    sigma: finite distance at which the inter-particle potential is zero
    Output:
    enes: list of LJ energies
    """
    return 4.0 * epsilon * [(sigma / r)**12 - (sigma / r)**6 for r in rs]

def morse(
        rs: list[float],
        D_e: float = 1.0,
        alpha: float = 1.0,
        r_e: float = 1.0
):
    """
    A function that generates the Morse potential
    given a list of distances
    Input:
    rs: list of distances
    D_e: depth of the potential well
    alpha: width of the potential well
    r_e: equilibrium bond distance
    Output:
    enes: list of Morse energies
    """
    import numpy as np
    return [D_e * (1 - np.exp(-alpha * (r - r_e)))**2 for r in rs]
