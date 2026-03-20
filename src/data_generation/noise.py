

def noise_sine(
    xs: list[float],
    amplitude: float = 1.0,
    frequency: float = 1.0,
    initial_state: float = 0.0,
    bias: float = 0.0
) -> list[float]:
    """
    Sine noise function that adds noise to the data as
    noise = A * sin(2 * pi * f * (x - x_0)) + B
    Input:
        xs: a list of x values (list of doubles)
        amplitude: A
        frequency: f
        initial_state: x_0
        bias: B

    Output:
        a list of noise values with the same length as xs (list of doubles)
    """
    import numpy as np
    return [ amplitude * np.sin(2 * np.pi * frequency * (x - initial_state)) + bias for x in xs ]

def noise_gaussian(
        xs: list[float],
        seed: int = 0,
        amplitude: float = 1.0,
        mean: float = 0.0,
        standard_deviation: float = 1.0
) -> list[float]:
    """
    Gaussian noise function that adds noise to the data as
    noise = A * N(mean, standard_deviation)
    Input:
        xs: a list of x values (list of doubles)
        seed: the random seed for reproducibility
        amplitude: A
        mean: the mean of the Gaussian distribution
        standard_deviation: the standard deviation of the Gaussian distribution

    Output:
        a list of noise values with the same length as xs (list of doubles)
    """
    import numpy as np
    np.random.seed(seed)
    return [ amplitude * np.random.normal(mean, standard_deviation) for _ in xs ]

def noise_decay(
        xs: list[float],
        initial_state: float = 1.0,
        decay_rate: float = 0.1
) -> list[float]:
    """
    Decay noise function that adds noise to the data as
    noise = x_0 * exp(-decay_rate * x)
    Input:
        xs: a list of x values (list of doubles)
        initial_state: x_0
        decay_rate: the decay rate

    Output:
        a list of noise values with the same length as xs (list of doubles)
    """
    import numpy as np
    return [ initial_state * np.exp(-decay_rate * x) for x in xs ]