from .base_tracker import BaseTracker

import numpy as np
from hmmlearn import hmm
from sklearn.utils import check_random_state
from utils.evalution import classify_path, error_rate
import matplotlib.pyplot as plt


class HmmTracker(BaseTracker):
    def __init__(self, n_hidden, cov_type='diag'):
        super().__init__()
        self.n_hidden = n_hidden
        self.cov_type = cov_type
        self.transmat_cdf = None

        self.model = hmm.GaussianHMM(n_components=self.n_hidden, covariance_type=self.cov_type)

    def predict_and_update(self):
        return ...

    def predict_only(self, observation):
        """
        Implement only the predict stage
        """

        if self.transmat_cdf is None:
            raise Exception("transmat_cdf is None -- call train first before predict")
        states = self.model.predict(observation)  # this will predict hidden state of the observed data

        # from this point on, we used the predicted state to generate prediction for next state
        rand_state = check_random_state(self.model.random_state)
        next_state = (self.transmat_cdf[states[-1]] > rand_state.rand()).argmax()
        next_obs = self.model._generate_sample_from_state(next_state, rand_state)
        return next_state, next_obs

    def train(self, X):

        lengths = [len(t) for t in X]  # in case trajectories has varying length
        X_train = [i for sublist in X for i in sublist]  # flatten out the list

        self.model.fit(X_train, lengths)

        self.transmat_cdf = np.cumsum(self.model.transmat_, axis=1)

    def evaluate(self, X):

        X_test = [x[:-1] for x in X]
        y_test = [x[len(x) - 1] for x in X]

        pred = []
        for t in X_test:
            _, next_point = self.predict_only(t)
            pred.append(next_point)

        path_test, path_predict, accuracy = classify_path(X, pred)
        abs_error, pct_error, total_dist, path_length = error_rate(X, pred, y_test)

        print("Average error (in pixel): ", abs_error)
        print("Average percentage error: ", pct_error)
        print("Classification accuracy (isCrosswalk): ", accuracy)


    def plot_pattern(self, n=200):

        X, Z = self.model.sample(n)

        # Plot the sampled data
        plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=5, mfc="orange", alpha=0.7)

        plt.legend(loc='best')
        plt.show()
