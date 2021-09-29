from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from math import sqrt, pi, exp
import numpy as np
import pandas as pd


def gaussian(x, mean, cov):
    d = cov.shape[0]
    exponent = exp(-0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean))
    t = pow(pow(2 * pi, d / 2) * sqrt(abs(np.linalg.det(cov))), -1)
    return t * exponent


def bayesian_train(df):

    means = []
    covs = []
    classes = len(df["class"].value_counts()) # no of classes
    class_freq = np.zeros((classes,), dtype="float") # frequency of classes for finding priors

    # splitting the data
    X_train, X_test = train_test_split(df.values, test_size=0.2, random_state=42)
    train_data = pd.DataFrame({"x": X_train[:, 0], "y": X_train[:, 1], "class": X_train[:, 2].astype(int)})
    test_data = pd.DataFrame({"x": X_test[:, 0], "y": X_test[:, 1], "class": X_test[:, 2].astype(int)})

    for i, group in train_data.groupby("class"):
        group = group.drop(columns=["class"])
        class_freq[i] = group.shape[0]
        # MLE of Gaussian
        # just take mean and cov of data itself.
        means.append(np.mean(group, axis=0))
        covs.append(np.cov(group.T))

    priors = class_freq / class_freq.sum()

    return classes, priors, means, covs, train_data, test_data


def bayesian_test(classes, priors, means, covs, test_data):

    truth = test_data["class"].values
    prediction = np.zeros((test_data.shape[0],))

    for index, row in test_data.iterrows():
        # iterating over test data
        x = np.array([row["x"], row["y"]])
        g = []
        # calculating prior * likelihood
        for i in range(classes):
            g.append(priors[i] * gaussian(x, means[i], covs[i]))
        prediction[index] = np.argmax(g)

    return truth, prediction, test_data

def normalized_histograms_train(df):

    classes = len(df["class"].value_counts())
    class_freq = np.zeros((classes,), dtype="float")

    X_train, X_test = train_test_split(df.values, test_size=0.2, random_state=42)
    train_data = pd.DataFrame({"x": X_train[:, 0], "y": X_train[:, 1], "class": X_train[:, 2].astype(int)})
    test_data = pd.DataFrame({"x": X_test[:, 0], "y": X_test[:, 1], "class": X_test[:, 2].astype(int)})

    class_probs = [] 
    bins = [] 

    for i, group in train_data.groupby("class"):
        group = group.drop(columns=["class"])
        class_freq[i] = group.shape[0]
        # calculating histogram for each feature
        feature_probs = [] 
        feature_bins = [] 
        for c in group.columns:
            hist, bin_edges = np.histogram(group.loc[:, (c)].values, density=True)
            probabilities = hist * np.diff(bin_edges)
            feature_probs.append(probabilities)
            feature_bins.append(bin_edges)
        class_probs.append(feature_probs)
        bins.append(feature_bins)

    priors = class_freq / class_freq.sum()
    return classes, priors, class_probs, bins, train_data, test_data

def histogram_prob(feature_probs, feature_bins, x):

    probability = 1 
    for feature, probs, bins in zip(x, feature_probs, feature_bins):
        for i in range(len(bins) - 1):
            if bins[i] < feature <= bins[i+1]:
                probability *= probs[i]
                break 
        else:
            return 0
    return probability

def normalized_histograms_test(classes, priors, class_probs, bins, test_data):
    truth = test_data["class"].values
    prediction = np.zeros((test_data.shape[0],))

    for index, row in test_data.iterrows():
        x = np.array([row["x"], row["y"]])
        g = []
        for i in range(classes):
            g.append(priors[i] * histogram_prob(class_probs[i], bins[i], x))
        prediction[index] = np.argmax(g)

    return truth, prediction, test_data

if __name__ == "__main__":
    from data import ls_data, nls_data, real_data

    # classes, priors, means, covs, train_data, test_data = bayesian_train(ls_data)
    # truth, prediction, data = bayesian_test(classes, priors, means, covs, test_data)

    # classes, priors, means, covs, train_data, test_data = bayesian_train(nls_data)
    # truth, prediction, data = bayesian_test(classes, priors, means, covs, test_data)

    # classes, priors, means, covs, train_data, test_data = bayesian_train(real_data)
    # truth, prediction, data = bayesian_test(classes, priors, means, covs, test_data)

    # classes, priors, class_probs, bins, train_data, test_data = normalized_histograms_train(ls_data)
    # truth, prediction, data = normalized_histograms_test(classes, priors, class_probs, bins, test_data)    

    classes, priors, class_probs, bins, train_data, test_data = normalized_histograms_train(nls_data)
    truth, prediction, data = normalized_histograms_test(classes, priors, class_probs, bins, test_data)

    fig, ax = plt.subplots(ncols=2, figsize=(5 * 2, 5 * 1))
    color = np.array(["red", "green", "blue"])

    for i, group in data.groupby("class"):
        ax[1].scatter(label=i, c=color[i], x=group["x"], y=group["y"])

    ax[0].scatter(
        data["x"], data["y"], c=color[prediction.astype(int)]
    )

    accuracy = (truth == prediction).sum() / truth.shape[0]
    cm = confusion_matrix(truth, prediction)

    print('Confusion matrix\n\n', cm)

    print('\nTrue Positives(TP) = ', cm[0,0])

    print('\nTrue Negatives(TN) = ', cm[1,1])

    print('\nFalse Positives(FP) = ', cm[0,1])

    print('\nFalse Negatives(FN) = ', cm[1,0])

    ax[0].set_title("prediction")
    ax[1].set_title("Truth")
    fig.suptitle(f"accuracy : {accuracy*100:.5f}", fontsize=20)

    plt.legend()
    plt.show()
    plt.close()

    # fig, ax = plt.subplots(nrows=2, ncols=2)

    # for i in range(classes):
    #     for j, k in enumerate(class_probs[i]):
    #         ax[i,j].bar(bins[i][j][:-1], k)
    #         ax[i,j].set_title(f" class {i}, feature {j} ")

    # plt.show()
