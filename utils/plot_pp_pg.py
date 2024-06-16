import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
from imbalanced_classification.ppo.ppo_agent import *

def plot_tpr_pp(weight_matrix, plot_version="Weights"):
    classes = int(len(weight_matrix))
    index = np.arange(classes)

    # Separate positive and negative values and their indices
    positive_values = [np.abs(value) if value >= 0 else 0 for value in weight_matrix]
    negative_values = [np.abs(value) if value < 0 else 0 for value in weight_matrix]

    fig, ax = plt.subplots()

    # Plot positive values in blue
    ax.bar(index, positive_values, color='blue')

    # Plot negative values in red
    ax.bar(index, negative_values, color='red')

    # Add title and labels
    plt.xlabel('Class')
    plt.ylabel(plot_version)
    plt.title(plot_version + ' Weight by Class')
    plt.xticks(index, [f'Class {i}' for i in range(classes)])

    # plt.show()
    return fig


def plot_weights_gender(a, plot_version="Weights"):
    classes = int(len(a)/2)
    bar_width = 0.35
    index = np.arange(classes)  # Changed to np.arange for numerical operations


    weight_matrix = a.reshape((2, classes),order='F')  # Reshaped to (2, classes) instead of (classes, 2)
    gender_words = ["Men", "Women"]
    gender_colors = ['blue', 'red']

    fig, ax = plt.subplots()

    # Create bars for each gender
    for i, gender in enumerate(gender_words):
        ax.bar(index + bar_width * i, weight_matrix[i, :], width=bar_width, label=f'Gender: {gender}', color=gender_colors[i])  # Adjusted indexing

    # Add title and labels
    plt.xlabel('Class')

    plt.ylabel(plot_version)
    plt.title(plot_version + 'Weight by Class and Gender')
    plt.xticks(index + bar_width/2, [i for i in range(classes)])  # Adjusted labels for clarity
    ax.legend()

    return fig



def solve_nnls_qp(A, B):
    """
    Solve the non-negative least squares problem using quadratic programming.

    Parameters:
    A (numpy.ndarray): The matrix representing the system (m x n).
    B (numpy.ndarray): The reward vector (m x 1).

    Returns:
    numpy.ndarray: The optimized weights vector W (n x 1) with non-negative values.
    """

    # Suppress cvxopt output
    solvers.options['show_progress'] = False
    A = np.array(A).astype(np.double)
    B = np.array(B).astype(np.double)

    # Convert numpy arrays to cvxopt matrices
    Q = matrix(np.dot(A.T, A))
    c = matrix(-np.dot(A.T, B))
    G = matrix(np.diag(np.ones(A.shape[1]) * -1))  # For non-negativity constraints
    h = matrix(np.zeros(A.shape[1]))

    # Solve the QP problem
    sol = solvers.qp(Q, c, G, h)

    # Extract and return the solution
    W = np.array(sol['x']).flatten()
    return W
