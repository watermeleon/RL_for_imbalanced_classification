# Code from https://github.com/yanaiela/amnesic_probing,
# https://github.com/shauli-ravfogel/nullspace_projection  with some modifications

import numpy as np
from typing import List
import gc
import scipy
import tqdm

# np.random.seed(10)


class INLPMethod:
    def __init__(self, m, parameters):
        """
        :param classifier_class: the sklearn classifier class (SVM/Perceptron etc.)
        :param cls_params: a dictionary, containing the params for the sklearn classifier
        """
        self.model = m
        self.parameters = parameters
        self.classifier_class = self.model(**self.parameters)

    def train_network(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray):
        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set / Person's R in the case of regression
        """

        # TEST MINI-BATCH
        batch_size = min(len(X_train), 100000)
        # print("batch size: ", batch_size)
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = Y_train[i:i + batch_size]

            # Train the classifier on the current batch
            self.classifier_class.partial_fit(X_batch, y_batch, classes=np.unique(Y_train))

            del X_batch
            del y_batch
            gc.collect()

        # TEST MINI-BATCH

        score = self.classifier_class.score(X_dev, Y_dev)
        weights = self.get_weights()

        del X_train
        del Y_train
        del X_dev
        del Y_dev
        gc.collect()

        return score, weights

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """
        w = self.classifier_class.coef_
        if len(w.shape) == 1:
            w = np.expand_dims(w, 0)
        return w

    def get_rowspace_projection(self, W: np.ndarray) -> np.ndarray:
        """
        :param W: the matrix over its nullspace to project
        :return: the projection matrix over the rowspace
        """
        # W -> (41, 768)
        if np.allclose(W, 0):
            w_basis = np.zeros_like(W.T)
        else:
            # Construct an orthonormal basis for the range of W using SVD
            w_basis = scipy.linalg.orth(W.T)  # orthogonal basis
        w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
        P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace
        # Shape -> (768, 768)
        del w_basis
        del W
        gc.collect()

        return P_W

    def get_projection_to_intersection_of_nullspaces(self, rowspace_projection_matrices: List[np.ndarray],
                                                     input_dim: int):
        """
        Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
        this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
        uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
        N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
        :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
        :param input_dim: input dim
        """

        I = np.eye(input_dim)
        Q = np.sum(rowspace_projection_matrices, axis=0)
        P = I - self.get_rowspace_projection(Q)

        del I
        del Q
        gc.collect()
        return P
    
    def random_class_generator(self, main_task_labels):
        """ Yield class labels in a random order."""
        while True:
            np.random.shuffle(main_task_labels)
            for cls in main_task_labels:
                yield cls


    def get_debiasing_projection(self, num_classifiers: int, input_dim: int,
                                 is_autoregressive: bool,
                                 min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                                 Y_dev: np.ndarray, best_iter_diff=0.01, summary_writer=None, verbose=False, by_class=False,
                                 Y_train_main=None, Y_dev_main=None ) \
            -> (np.ndarray, list, list, list, tuple):
        """
        :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
        :param input_dim: size of input vectors
        :param is_autoregressive: whether to train the ith classiifer on the data projected to the nullsapces of w1,...,wi-1
        :param min_accuracy: above this threshold, ignore the learned classifier
        :param X_train: ndarray, training vectors
        :param Y_train: ndarray, training labels (protected attributes)
        :param X_dev: ndarray, eval vectors
        :param Y_dev: ndarray, eval labels (protected attributes)
        :param best_iter_diff: float, diff from majority, used to decide on best iteration
        :return: P, the debiasing projection; rowspace_projections, the list of all rowspace projection;
                Ws, the list of all calssifiers.
        """

        x_train_cp = X_train.copy()
        x_dev_cp = X_dev.copy()
        rowspace_projections = []
        accuracy_per_iteration = []
        Ws = []
        all_projections = []
        best_projection = None
        iters_under_threshold = 0
        prev_acc = -99.
        iters_no_change = 0

        # pbar = tqdm(range(num_classifiers))
        if by_class:
            print("By class training")
            if ((Y_train_main is None) or (Y_dev_main is None)):
                raise Exception("Need main-task labels for by-class training.")
            main_task_labels = list(set(Y_train_main.tolist()))

            random_class_gen = self.random_class_generator(main_task_labels)
        else:
            relevant_idx_train = np.ones(x_train_cp.shape[0], dtype=bool)
            relevant_idx_dev = np.ones(x_dev_cp.shape[0], dtype=bool)


        # number of classifiers/iterations: default 20
        pbar = tqdm.tqdm(range(num_classifiers)) 
        for i in pbar:
            if verbose: print("iteration: ", i)

            if by_class:
                cls = next(random_class_gen)
                relevant_idx_train = Y_train_main == cls
                relevant_idx_dev = Y_dev_main == cls
            # else:

            # train network with projected x_train and x_dev
            acc, W = self.train_network(x_train_cp[relevant_idx_train], Y_train[relevant_idx_train], x_dev_cp[relevant_idx_dev], Y_dev[relevant_idx_dev])
            if verbose: print("shape of w: ", W.shape)
            accuracy_per_iteration.append(acc)

            # pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
            if verbose: print("iteration: {}, accuracy: {}".format(i, acc))


            if iters_under_threshold >= 3:
                # if verbose: print('3 iterations under the minimum accuracy.. stopping the process')
                print('3 iterations under the minimum accuracy.. stopping the process')
                break
            # print("acc is ", acc)
            if acc <= min_accuracy and best_projection is not None:
                iters_under_threshold += 1
                continue

            if prev_acc == acc:
                iters_no_change += 1
            else:
                iters_no_change = 0

            if iters_no_change >= 3:
                # if verbose: print('3 iterations with no accuracy change.. stopping the process')
                print('3 iterations with no accuracy change.. stopping the process')
                break

            prev_acc = acc

            Ws.append(W)

            # (768, 768)
            P_rowspace_wi = self.get_rowspace_projection(W)

            rowspace_projections.append(P_rowspace_wi)

            del P_rowspace_wi
            gc.collect()

            if is_autoregressive:
                """
                to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far
                 (instead of doing X = P_iX, which is problematic when w_i is not exactly orthogonal to w_i-1,...,w1,
                  due to e.g inexact argmin calculation).
                """
                # use the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
                # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))

                if verbose: print("Get projection to intersection of nullspaces")
                P = self.get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

                all_projections.append(P)

                # project
                if verbose: print("Project train and dev set")

                del x_train_cp
                del x_dev_cp
                gc.collect()

                x_train_cp = np.dot(X_train, P)
                x_dev_cp = np.dot(X_dev, P)

                # the first iteration that gets closest performance (or less) to majority
                if (acc - min_accuracy) <= best_iter_diff and best_projection is None:
                    if verbose: print('projection saved timestamp: {}'.format(i))
                    best_projection = (P, i + 1)

                del P
                gc.collect()

        """
        calculate final projection matrix P=PnPn-1....P2P1
        since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
        by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability,
        i.e., we explicitly project to intersection of all nullspaces (this is not critical at this point; I-(P1+...+PN)
        is roughly as accurate as this)
        """

        final_p = self.get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

        if best_projection is None:
            if verbose: print('projection saved timestamp: {}'.format(num_classifiers))
            if verbose: print('using all of the iterations as the final projection')
            best_projection = (final_p, num_classifiers)

        del X_train
        del X_dev
        del x_train_cp
        del x_dev_cp
        del W

        gc.collect()

        return final_p, rowspace_projections, Ws, all_projections, best_projection, accuracy_per_iteration

    def get_projection_matrix(self, num_clfs, x_train, y_train, x_dev, y_dev,
                              majority_acc, summary_writer=None, verbose=False, by_class=False, Y_train_main=None, Y_dev_main=None):

        # 'early_stopping': True} -> must be False for partial fit
        dim = x_train.shape[1]
        if verbose: print("X dimensionality: ", dim)
        P, _, _, all_projections, best_projection, accuracy_per_iteration = self.get_debiasing_projection(
            num_clfs, dim,
            is_autoregressive=True,
            min_accuracy=majority_acc,
            X_train=x_train, Y_train=y_train,
            X_dev=x_dev, Y_dev=y_dev,
            summary_writer=summary_writer,
            verbose=verbose,
            by_class=by_class,
            Y_train_main=Y_train_main,
            Y_dev_main=Y_dev_main
            )

        return P, all_projections, best_projection, accuracy_per_iteration
