from os.path import join
from numpy.random import triangular
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pow, sqrt
import math
import matplotlib
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm

def make_df(path, *classes, sep=","):
    dfs = []
    for i, c in enumerate(classes):
        df = pd.read_csv(join(path, c), names=["x", "y"], sep=sep, dtype={"x": "float", "y": "float"}, engine="python")
        df["class"] = i
        dfs.append(df)
    return pd.concat(dfs, axis=0).reset_index(drop=True)

class GMM:

    def gaussian(self, x, mean, cov):
        
        return multivariate_normal.pdf(x, mean=mean, cov=cov)

    def loglikeli(self, data, ncomponents, mu, cov, pi):
        ll = 0
        for n in range(data.shape[0]):
            s = 0
            for q in range(ncomponents):
                s += pi[q] * self.gaussian(data[n], mu[q], cov[q])
            ll += np.log(s)
        return ll

    def weight_likelihood(self, x, q, mu, cov, pi):
        
        return pi[q] * self.gaussian(x, mu[q], cov[q])

    def initial_params(self, X, ncomponents, mu, cov):
        splitted = np.split(X, ncomponents)
        for q in range(ncomponents):
            mu[q] = np.mean(splitted[q], axis=0)
            cov[q] = np.cov(splitted[q], rowvar=False, bias=True)
        return (mu, cov)

    def e_step(self, train, ncomponents, mu, cov, pi):
        assigned_clusters = np.zeros((train.shape[0], 1))
        gamma = np.zeros((train.shape[0], ncomponents))

        for n in range(train.shape[0]):
            posterior = []
            for q in range(ncomponents):
                posterior.append(self.weight_likelihood(train[n], q, mu, cov, pi))

            posterior = np.array(posterior) / np.sum(posterior)
            gamma[n, :] = posterior.ravel()
            assigned_clusters[n] = np.argmax(posterior)
            assert np.sum(posterior) > 0.99, sum(posterior)

        return assigned_clusters, gamma


    def m_step(self, train, ncomponents, mu, cov, pi, assigned_clusters, gamma, nfeatures):
        Nq = np.sum(gamma, axis=0)
        N = train.shape[0]
        pi = Nq / N

        for q in range(ncomponents):
            new_mu = gamma[:, q].reshape(-1, 1).T @ train
            new_mu = new_mu.reshape(-1, nfeatures)
            new_mu /= Nq[q]

            new_cov = np.zeros((nfeatures, nfeatures))

            for n in range(train.shape[0]):
                diff = train[n] - mu[q, :]
                diff = diff.reshape(-1, 1)
                new_cov += gamma[n, q] * (diff @ (diff).T)

            new_cov /= Nq[q]
            # new_cov.flat[:: (nfeatures + 1)] += 1e-6

            mu[q] = new_mu
            cov[q, ...] = new_cov

        return mu, cov, pi


    def gmm_fit(self, X, ncomponents=2):
        self.ncomponents = ncomponents
        train = np.asarray(X)
        nfeatures = X.shape[1]
        threshold = 1

        mu = np.zeros((ncomponents, nfeatures))
        cov = np.zeros((ncomponents, nfeatures, nfeatures))
        pi = np.ones((ncomponents, 1)) / ncomponents
        mu, cov = self.initial_params(train, ncomponents, mu, cov)

        old_ll = 0
        iterations = 0

        while 1:
            ll = self.loglikeli(train, ncomponents, mu, cov, pi)

            if abs(ll - old_ll) < threshold:
                break
            old_ll = ll

            assigned_clusters, gamma = self.e_step(train, ncomponents, mu, cov, pi)
            mu, cov, pi = self.m_step(train, ncomponents, mu, cov, pi, assigned_clusters, gamma, nfeatures)
            iterations += 1

        self.mu = mu 
        self.cov = cov 
        self.pi = pi

        return self

    def score_samples(self, X):
        test = np.asarray(X)
        nfeatures = test.shape[1]
        scores = np.zeros((X.shape[0], 1))

        for n in range(X.shape[0]):
            p = 0
            for q in range(self.ncomponents):
                p += self.weight_likelihood(test[n], q, self.mu, self.cov, self.pi)
            scores[n] = p
        return scores


def gmm_predict(data, ncomponents=2):

    data = np.asarray(data)
    X = data[:, :2].astype(float)
    y = data[:, 2]
    nclasses = len(np.unique(y))
    gmms = []
    for n in tqdm(range(nclasses)):
        gm = GMM().gmm_fit(X[y == n], ncomponents)
        gmms.append(gm)

    scores = []
    for n in range(nclasses):
        scores.append(gmms[n].score_samples(X).reshape(-1, 1))

    return np.argmax(np.concatenate(scores, axis=1), axis=1)


def plot_predict(data):
    # uncomment for plots
    plt.scatter(data["x"], data["y"], c=data["class"])
    plt.title("plot of data")
    plt.show()

    sns.jointplot(data=data, x="x", y="y", hue="class")
    plt.show()

    assigned = gmm_predict(data)

    data["assigned"] = assigned
    accuracy = np.sum(assigned == np.asarray(data["class"])) / assigned.shape[0]

    plt.scatter(data["x"], data["y"], c=data["assigned"])
    plt.title(f"plot of predicted data | Accuracy = {accuracy*100:.03f}%")
    plt.show()

# matplotlib.use("Agg")
if __name__ == "__main__":
    LS = "./data_1/"
    NLS = "./nls_data/"
    ls_data = make_df(LS, "Class1.txt", "Class2.txt")
    nls_data = make_df(NLS, "class1.txt", "class2.txt")

    plot_predict(nls_data)
