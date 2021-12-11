import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


class KMeans:
    def __init__(self, eps=1, max_iters=30):
        self.eps = eps
        self.max_iters = max_iters

    def _get_initial_centroids(self, X, k):
        """
        k random data points from dataset X
        """
        return X[np.random.choice(X.shape[0], size=k, replace=False), ...]

    def _get_clusters(self, X, centroids):
        """
        Assigns clusters.
        """

        k = centroids.shape[0]

        clusters = {i: [] for i in range(k)}

        distance_matrix = self._get_euclidean_distance(X, centroids)

        closest_cluster_ids = np.argmin(distance_matrix, axis=1)

        for i in range(k):
            clusters[i] = np.nonzero(closest_cluster_ids == i)[0]

        return clusters

    def _get_euclidean_distance(self, X, centroids):
        """
        Computes euclidean distance.

        X => n x d
        centroids => k x d
        """
        k = centroids.shape[0]

        distances = np.sum(np.power(X[:, None, :] - centroids[None, ...], 2), axis=2)

        return np.sqrt(distances)

    def _has_centroids_covered(self, previous_centroids, new_centroids):
        """
        checks if any of centroids moved more then eps
        """
        distances_between_old_and_new_centroids = np.sqrt(
            np.sum(np.power(previous_centroids - new_centroids, 2), axis=1)
        )
        centroids_moved = np.max(distances_between_old_and_new_centroids) > self.eps

        return centroids_moved

    def fit(self, X, k):
        X = np.asarray(X)
        print("Fitting....")
        new_centroids = self._get_initial_centroids(X=X, k=k)

        centroids_moved = False
        iterations = 0

        while 1:
            if iterations > self.max_iters:
                break

            previous_centroids = new_centroids.copy()
            clusters = self._get_clusters(X, previous_centroids)

            new_centroids = np.array([np.mean(X[clusters[i]], axis=0) for i in range(k)])

            centroids_moved = self._has_centroids_covered(previous_centroids, new_centroids)

            if iterations != 0 and centroids_moved == False:
                break

            iterations += 1

        self.centers = new_centroids.copy()

        return self

    def fit_predict(self, X, k):
        self.fit(X, k)

        return self.predict(X)

    def predict(self, X):
        X = np.asarray(X)
        print("Predicting....")
        distance = self._get_euclidean_distance(X, self.centers)
        assigned_centroid = np.argmin(distance, axis=1)
        return assigned_centroid, self.centers


# %% q1


def build_data(class1_path, class2_path):
    class1 = pd.read_csv(class1_path, header=None)
    class1.columns = ["A", "B"]
    class1["class"] = 0

    class2 = pd.read_csv(class2_path, header=None)
    class2.columns = ["A", "B"]
    class2["class"] = 1

    data = pd.concat([class1, class2], ignore_index=True).reset_index(drop=True)

    return data[data.columns[:2]], data["class"]


def question1(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.30)

    colors = np.array(["orange", "blue"])
    model = KMeans()
    preds, _ = model.fit_predict(X_train.values, 2)

    X_test["pred"], centres = model.predict(X_test.values)

    for i, j in X_test.groupby("pred"):
        plt.scatter(
            j["A"],
            j["B"],
            c=colors[j["pred"]],
            label=f"class {i} Test",
            alpha=0.8,
            s=10,
        )

    score = accuracy_score(X_test["pred"], Y_test)
    if score < 0.5:
        score = 1 - score
    print("Accuracy Score for K-means on NLS data:", score)
    plt.scatter(centres[:, 0], centres[:, 1], c=colors[model.predict(centres)[0]], s=50)
    plt.title("Scatter Plot", fontsize=20)
    plt.xlabel("Attr 1", fontsize=14)
    plt.ylabel("Attr 2", fontsize=14)
    plt.legend()
    plt.show()
    plt.close()


X, Y = build_data(os.path.join("./nls_data", "class1.txt"), os.path.join("./nls_data", "class2.txt"))
question1(X, Y)

# %% q2 funcs


def KMeansCluster(vectorized, K=5):
    label, center = KMeans(max_iters=30).fit_predict(vectorized, K)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res


def showImages(orig_image, new_image, K=5):
    result_image = new_image.reshape((orig_image.shape))
    fig, ax = plt.subplots(ncols=2, figsize=(15, 15))
    ax[0].imshow(orig_image)
    ax[0].set_title("Original Image")
    ax[1].imshow(result_image)
    ax[1].set_title(f"Segmented Image when K = {K}")
    plt.show()
    plt.close()


# %% q2
print("Part 1")

image = cv2.imread("Image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

vectorized = image.reshape((-1, 3))
vectorized = np.float32(vectorized)

K = 5
res = KMeansCluster(vectorized, K=K)
showImages(image, res, K)

print(" Part 2")

weights = 1
x_len, y_len = image.shape[:-1]

xx, yy = np.meshgrid(np.arange(y_len) * 256 * weights / y_len, np.arange(x_len) * 256 * weights / x_len)

locs = np.dstack((xx, yy)).reshape(-1, 2)

image_with_loc = np.concatenate((image.reshape(-1, 3), locs), axis=1)
vectorized = np.float32(image_with_loc)

K = 5
res = KMeansCluster(vectorized, K=K)
res_img = res[..., :-2]
showImages(image, res_img, K)

# %%
