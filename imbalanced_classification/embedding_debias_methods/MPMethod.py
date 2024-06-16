import gc
import numpy as np
import scipy
import sys
from typing import List

# np.random.seed(10)


class MPMethod:
    def __init__(self, task, directions, input_dim=768):
        self.task = task
        self.W = directions
        self.input_dim = input_dim

    def get_rowspace_projection(self, W: np.ndarray) -> np.ndarray:
        """
        :param W: the matrix over its nullspace to project
        :return: the projection matrix over the rowspace
        """

        if np.allclose(W, 0):
            w_basis = np.zeros_like(W.T)
        else:
            w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

        P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

        del w_basis
        del W
        gc.collect()
        return P_W

    def debias_by_specific_directions(self, directions: List[np.ndarray], input_dim: int):
        """
        the goal of this function is to perform INLP on a set of user-provided directiosn (instead of learning those directions).
        :param directions: list of vectors, as numpy arrays.
        :param input_dim: dimensionality of the vectors.
        """

        rowspace_projections = []

        W = np.empty((0, input_dim), dtype="float64")

        for v in directions:
            W = np.vstack((W, v))

        Q = self.get_rowspace_projection(W)
        rowspace_projections.append(Q)

        P = np.eye(input_dim, dtype="float64") - Q

        del Q
        del W
        del directions
        gc.collect()
        return P, rowspace_projections

    def mean_projection_method(self):
        P, rowspace_projs = self.debias_by_specific_directions(self.W, input_dim=self.input_dim)
        return P


def assign_dep(x_train, y_train, value):
    dep_mask = y_train == value
    dep_x_train = x_train[dep_mask]
    return dep_x_train


def get_labels(x_train, y_train):
    labels_list = []
    unique_labels = list(set(y_train))
    for i in unique_labels:
        labels_list.append(assign_dep(x_train, y_train, i))
    return labels_list


def get_directions(input_x_train, input_y_train):
    input_labels = get_labels(input_x_train, input_y_train)

    weights = []

    for i, value in enumerate(input_labels):
        tmp_list = input_labels.copy()
        tmp_list.pop(i)

        target_sum = np.mean(value, axis=0)
        rest_of_sums = [np.mean(x, axis=0) for x in tmp_list]
        v_means = target_sum - np.mean(rest_of_sums, axis=0)
        v_means = v_means / np.linalg.norm(v_means)
        v_means = v_means.reshape(1, -1)

        weights.append(v_means)

    return weights


def assign_dep_weighted(x_train, y_train, value, input_weights):
    dep_mask = y_train == value
    dep_x_train = x_train[dep_mask]
    dep_weights = input_weights[dep_mask]
    return dep_x_train, dep_weights


def get_labels_weighted(x_train, y_train, input_weights):
    labels_list = []
    unique_labels = list(set(y_train))
    for i in unique_labels:
        labels_list.append(assign_dep_weighted(x_train, y_train, i, input_weights))
    return labels_list

def get_directions_weighted(input_x_train, input_y_train, input_weights):
    input_labels = get_labels_weighted(input_x_train, input_y_train, input_weights)

    weights = []

    for i, value_tuple in enumerate(input_labels):
        tmp_list = input_labels.copy()
        tmp_list.pop(i)

        value, input_weights = value_tuple

        target_sum = np.average(value, axis=0, weights=input_weights)
        rest_of_sums = [np.average(x, axis=0, weights=w_i) for x, w_i in tmp_list]
        v_means = target_sum - np.mean(rest_of_sums, axis=0)
        v_means = v_means / np.linalg.norm(v_means)
        v_means = v_means.reshape(1, -1)

        weights.append(v_means)

    return weights