
def abscissa_gen(start: float, 
                 end: float, 
                 step: float) -> list[float]:
    """
    Generate a list of abscissa values from start to end with a given step.
    
    Args:
        start: the starting value of the abscissa
        end: the ending value of the abscissa
        step: the step size for generating the abscissa values
    
    Returns:
        A list of abscissa values.
    """
    import numpy as np
    
    return np.arange(start, end + step, step).tolist()