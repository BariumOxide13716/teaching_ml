
import numpy as np
def abscissa_gen(start: float, 
                 end: float, 
                 step: float) -> np.ndarray:
    """
    Generate a list of abscissa values from start to end with a given step.
    
    Args:
        start: the starting value of the abscissa
        end: the ending value of the abscissa
        step: the step size for generating the abscissa values
    
    Returns:
        An array of abscissa values.
    """
    import numpy as np
    
    return np.arange(start, end + step, step)

def use_partial(my_list: np.ndarray,
                ratio_to_discard: float) -> np.ndarray:
        """
        Discard the first and the last ratio_to_discard portion of the array.
        Args:
            my_list: the array to be processed
            ratio_to_discard: the ratio of the array to be discarded from both ends
        Returns:
            An array with the first and the last ratio_to_discard portion of the array discarded.
        """
        n = len(my_list)
        start_index = int(n * ratio_to_discard)
        end_index = n - start_index
        return my_list[start_index:end_index]