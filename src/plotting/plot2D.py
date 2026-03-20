import numpy as np
"""
    Taking in several combination of x and y values.
    Plot them on a 2D graph with different colors and labels.
    Input:
        xs: a list of a list of x values
        ys: a list of a list of y values
    Optional Input:
        labels: a list of labels for each line
        title: the title of the plot
        xlabel: the label for the x-axis
        ylabel: the label for the y-axis
        markers: the marker of each x-y combination
        colors: the color for each line
        save_path: the path to save the plot. Default is None,
                which means the plot will be saved 
                in the current directory with the name "plot2d.png"
    Output:
        plotting_status: a boolean indicating whether 
        the plotting was successful

"""

def data_structure_checker(data: list) -> bool:
    """
    Check if the input data is a list of numpy ndarrays.
    """
    assert isinstance(data, list), "Input data must be a list."
    for item in data:
        if not isinstance(item, np.ndarray):
            return False
    return True

def plot2d(
    xs: list,
    ys: list,
    labels: list = None,
    title: str = "2D Plot",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    markers: list = None,
    colors: list = None,
    save_path: str = None
) -> bool:
    """
    Plot the given x and y values on a 2D graph.
    """
    import matplotlib.pyplot as plt

    # Check if xs and ys are lists of lists
    if not data_structure_checker(xs):
        print("Error: xs must be a list of numpy ndarrays.")
        return False
    if not data_structure_checker(ys):
        print("Error: ys must be a list of numpy ndarrays.")
        return False
    # Check if the lengths of xs and ys match
    if len(xs) != len(ys):
        print("Error: The number of x and y sets must be the same.")
        return False
    # Set default labels, markers, and colors if not provided
    if labels is None:
        labels = [f"Line {i+1}" for i in range(len(xs))]
    if markers is None:
        markers = ['o'] * len(xs)
    if colors is None:
        colors = plt.cm.viridis(range(len(xs)))
    
    # Plot each line
    for x, y, label, marker, color in zip(xs, ys, labels, markers, colors):
        plt.plot(x, y, marker=marker, color=color, label=label)
    # Set title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    # Save the plot    if save_path is None:
    save_path = "plot2d.png"
    plt.savefig(save_path)
    plt.close()
    return True
