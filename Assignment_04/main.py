from util import Perceptron
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import seaborn as sns

sns.set()

# get test and training data
def get_data(path1, path2):
    class1 = pd.read_csv(path1, header=None)
    class1.columns = ["x", "y"]
    class1["class"] = 0

    class2 = pd.read_csv(path2, header=None)
    class2.columns = ["x", "y"]
    class2["class"] = 1

    data = pd.concat([class1, class2], ignore_index=True)
    y = data["class"]
    data = pd.DataFrame(data)
    data.drop("class", axis=1, inplace=True)

    return train_test_split(data, y, test_size=0.3, random_state=42)


# function to plot decision boundary
def make_meshgrid(x, y, h=0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_data(X_train, X_test, y_train, y_test):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    sns.scatterplot(
        x="x",
        y="y",
        hue=y_train,
        data=X_train,
        ax=ax[0],
    )
    ax[0].set_title("Training Data", fontsize=20)
    sns.scatterplot(
        x="x",
        y="y",
        hue=y_test,
        data=X_test,
        ax=ax[1],
    )
    ax[1].set_title("Test Data", fontsize=20)
    plt.show()


def plot_decision_boundary(model, X, y):
    predicted_test = model.predict(X)
    cf = confusion_matrix(predicted_test, y)
    accuracy = accuracy_score(predicted_test, y)

    fig, ax = plt.subplots()

    X0, X1 = X["x"], X["y"]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=predicted_test, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_title("Decision Boundary for predicted data", fontsize=20)
    plt.show()

    print("\nAccuracy : %1.4f" % (accuracy * 100))
    print("Confusion Matrix \n", cf)
    return


if __name__ == "__main__":

    path = "ls_data"
    path1 = f"./{path}/class1.txt"
    path2 = f"./{path}/class2.txt"

    [X_train, X_test, Y_train, Y_test] = get_data(path1, path2)

    ## Part 1
    print("\n 1. Using Perceptron :- \n")
    plot_data(X_train, X_test, Y_train, Y_test)
    model = Perceptron(eta=0.9, n_iter=100).fit(np.array(X_train), Y_train)
    plot_decision_boundary(model, X_test, Y_test)

    ## Part 2

    print("\n 2. Using MLP :- \n")
    plot_data(X_train, X_test, Y_train, Y_test)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, Y_train)
    plot_decision_boundary(clf, X_test, Y_test)

    ## Part 3
    print("\n 3. Using SVM :- \n")
    plot_data(X_train, X_test, Y_train, Y_test)
    svc = SVC(kernel="linear").fit(X_train, Y_train)
    plot_decision_boundary(svc, X_test, Y_test)
