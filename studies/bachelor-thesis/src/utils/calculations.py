def standard_deviation_bounds(number, standard_deviation, n):
    """
    Calculate the lower and upper bounds using standard deviation.

    Parameters:
    number (float): The central value
    standard_deviation (float): The standard deviation
    n (int): The number of standard deviations

    Returns:
    tuple: (lower_bound, upper_bound)
    """
    if standard_deviation < 0 or n < 0:
        raise ValueError("Standard deviation and n must be non-negative")

    lower_bound = number - n * standard_deviation
    upper_bound = number + n * standard_deviation
    return lower_bound, upper_bound