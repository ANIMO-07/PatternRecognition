from os.path import join
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
from math import exp, pow, sqrt, log
from sklearn.cluster import KMeans
import math

def make_df(path, *classes, sep=","):
    dfs = []
    for i, c in enumerate(classes):
        df = pd.read_csv(join(path, c), names=["x", "y"], sep=sep, dtype={"x" : "float", "y" : "float"}, engine="python")
        df["class"] = i 
        dfs.append(df)
    return pd.concat(dfs, axis=0).reset_index(drop=True) 

def gaussian(x, mean, cov):
    d = cov.shape[0]
    exponent = np.exp(-0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean))

    t = pow(pow(2 * math.pi, d / 2) * sqrt(abs(np.linalg.det(cov))), -1)
    return t * exponent

def loglikeli(data, class_, ncomponents, nfeatures, mu, cov, pi):
    ll = 0
    tmp_mu = mu[class_, ...]
    tmp_cov = cov[class_, ...]
    tmp_pi = pi[class_, ...]

    for n in range(data.shape[0]):
        
        s = 0
        for q in range(ncomponents):
            s += tmp_pi[q] * gaussian(data[n,...], tmp_mu[q,...], tmp_cov[q,...])
        try:
            ll += log(s)
        except ValueError:
            pass
    return ll

def assign_classes(data, nclasses, ncomponents, mu, cov, pi):
    assigned = np.ones((data.shape[0], 1))
    for n in range(data.shape[0]):
        probs = []
        for class_ in range(nclasses):
            p = 0
            for q in range(ncomponents):
                p += pi[class_, q] * gaussian(data[n, ...], mu[class_, q, ...], cov[class_, q, ...])
            probs.append(p)
        assigned[n] = np.argmax(probs)
    return assigned

def gmm_fit(data, ncomponents=2):
    data = np.asarray(data)
    nclasses = len(np.unique(data[:,2]))
    nfeatures = np.shape(data)[1] - 1

    threshold = 1

    mu = np.ones((nclasses, ncomponents, nfeatures))
    cov = np.ones((nclasses, ncomponents, nfeatures, nfeatures))
    pi = np.ones((nclasses, ncomponents)) / ncomponents

    assigned_clusters = np.ones((data.shape[0]))

    weight_likelihood = lambda x, class_, q : pi[class_, q] * gaussian(x, mu[class_, q, :], cov[class_, q, ...])

    for class_ in range(nclasses):
        train = data[data[:,2] == class_][:, :-1]
        assigned_clusters_ = assigned_clusters[data[:,2] == class_]

        kmeans = KMeans(n_clusters=ncomponents, init="k-means++", max_iter=500, algorithm = 'auto')
        pred = kmeans.fit_predict(train)

        mu[class_, ...] = kmeans.cluster_centers_

        for q in range(ncomponents):
            cov[class_, q, ...] = np.cov(train[pred == q].T)

        old_ll = 0
        iterations = 0
        try:
            while 1:
                # print(f"{iterations=}")
                ll = loglikeli(train, class_, ncomponents, nfeatures, mu, cov, pi)
                # print(class_, old_ll, ll)
                if ll == 0:
                    break

                if (abs(ll - old_ll) < threshold):
                    break
                else:
                    old_ll = ll
                
                assigned_clusters_ = np.ones((train.shape[0], 1))

                for n in range(train.shape[0]):
                    denominator = np.sum([weight_likelihood(train[n, :], class_, q) for q in range(ncomponents)])
                    posterior = []
                    for q in range(ncomponents):
                        posterior.append(weight_likelihood(train[n, :], class_, q) / (denominator + 1e-8))
                    assigned_clusters_[n] = np.argmax(posterior)

                N = train.shape[0]
                Nq = np.array([np.sum(assigned_clusters_ == q) for q in range(ncomponents)])
                pi[class_, :] = Nq / N

                for q in range(ncomponents):
                    new_mu = np.zeros((1, nfeatures))
                    for n in range(train.shape[0]):
                        new_mu += weight_likelihood(train[n, :], class_, q) * train[n, :]
                    
                    new_mu /= Nq[q, ...]

                    new_cov = np.zeros((1, nfeatures, nfeatures))
                    for n in range(train.shape[0]):
                        new_cov += weight_likelihood(train[n, :], class_, q) * (train[n, :] - mu[class_, q, :]) @ (train[n, :] - mu[class_, q, :]).T
                    new_cov /= Nq[q, ...]
                    new_cov.flat[::(nfeatures + 1) ] += 1e-5
                    
                    mu[class_, q, :] = new_mu
                    cov[class_, q, ...] = new_cov 

                assigned_clusters[data[:,2] == class_] = assigned_clusters_.ravel()
                iterations += 1
        except Exception as e:
            pass
            
    return mu, cov, pi

def gmm_predict(data, mu, cov, pi, ncomponents=2):
    data = np.asarray(data)
    nclasses = len(np.unique(data[:,2]))
    nfeatures = np.shape(data)[1] - 1
    assigned = []
    for n in range(data.shape[0]):
        for c in range(nclasses):
            prob = []
            s = 0
            for q in range(ncomponents):
                s += pi[c, q] * gaussian(data[n, :-1], mu[c, q, ...], cov[c, q, ...])
            prob.append(s)
        assigned.append(np.argmax(prob))

    return assigned


if __name__ == '__main__':
    LS = "./data_1/"
    NLS = "./nls_data/"
    ls_data = make_df(LS, "Class1.txt", "Class2.txt")
    nls_data = make_df(NLS, "class1.txt", "class2.txt")

    # uncomment for plots
    plt.scatter(nls_data["x"], nls_data["y"], c=nls_data["class"])
    plt.title("plot of data")
    plt.show()

    # sns.jointplot(data=nls_data, x="x", y="y", hue="class")
    # plt.show()

    mu, cov, pi = gmm_fit(nls_data)
    assigned = gmm_predict(nls_data, mu, cov, pi)

    nls_data["assigned"] = assigned

    plt.scatter(nls_data["x"], nls_data["y"], c=nls_data["assigned"])
    plt.title("plot of data")
    plt.show()

