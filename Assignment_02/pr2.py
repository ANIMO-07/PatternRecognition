# %% Imports

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# %% not gajju bhaai

def KMeansCluster(vectorized, K=5):
    # print(vectorized)
    # K means clustering using the vectorized linear array

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    attempts = 10
    ret, label, center = cv2.kmeans(
        vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]
    return res


def showImages(orig_image, new_image, K=5):
    # print(orig_image)
    # print(new_image)

    # plotting images

    result_image = new_image.reshape((orig_image.shape))

    figure_size = 15
    plt.figure(figsize=(figure_size, figure_size))
    plt.subplot(1, 2, 1), plt.imshow(orig_image)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(result_image)
    plt.title('Segmented Image when K = %i' %
              K), plt.xticks([]), plt.yticks([])
    plt.show()


# %% Preparing the data for the model
print("Part 1")

image = cv2.imread("Image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

vectorized = image.reshape((-1, 3))
vectorized = np.float32(vectorized)

K = 5
res = KMeansCluster(vectorized, K=K)
showImages(image, res, K)

print(" Part 2")

image_with_loc = []
x_len = len(image)
y_len = len(image[0])

weights = 1

for i in range(len(image)):
    for j in range(len(image[i])):
        tup = image[i][j]
        tup = np.append(tup, [i * 256 * weights / x_len,
                        j * 256 * weights / y_len])
        image_with_loc.append(tup)

vectorized = np.float32(image_with_loc)

K = 5
res = KMeansCluster(vectorized, K=K)
res_img = []

for x in res:
    res_img.append(x[:-2])

res_img = np.array(res_img)
showImages(image, res_img, K)


# %% gajju

tol = pow(10, -3)
img = cv2.imread("Image.jpg", 1)

row, col, _ = img.shape
K = 10

def plot_clusters(iteration, cluster_colors, segmented_image, shape):
    cv2.imwrite(f"test_{iteration}.png", cluster_colors[segmented_image].reshape(shape))

colors = img.reshape(-1, 3).astype("float")
cluster_colors = colors[np.random.choice(colors.shape[0], size=K), ...]
old_cluster_colors = np.copy(cluster_colors)
old_distortion = 0
check_tol = None

iteration = 0
while 1:
    print(f"iteration {iteration}")
    l2 = np.sqrt(np.sum(np.power(colors[:, None, :] - cluster_colors[None, ...], 2), axis=2))
    
    # indexes => n x 1 x 2
    # centers => 1 x k x 2
    # expected => n x k x 2
    
    cluster_number = np.argmin(l2, axis=1)
    new_cluster_colors = np.zeros_like(cluster_colors)

    for i in range(K):
        new_cluster_colors[i] = np.mean(colors[np.nonzero(cluster_number == i)], axis=0)
    
    check_tol = np.sqrt(np.sum(np.power(colors - new_cluster_colors[cluster_number], 2), axis=1))

    plot_clusters(iteration, new_cluster_colors, cluster_number, img.shape)
    iteration += 1
    print(np.sum(check_tol), old_distortion)
    if np.sum(check_tol) - old_distortion < tol:
        cluster_colors = np.copy(new_cluster_colors)
        break
    else:
        old_distortion = np.sum(check_tol)
        old_cluster_colors = np.copy(cluster_colors)
        cluster_colors = np.copy(new_cluster_colors)

cluster_colors = cluster_colors[:, :2].astype("int")
# %% q1



def build_data(class1_path, class2_path):
    class1 = pd.read_csv(class1_path, header=None)
    class1.columns = ['A', 'B']
    class1['class'] = 0

    class2 = pd.read_csv(class2_path, header=None)
    class2.columns = ['A', 'B']
    class2['class'] = 1

    data = pd.concat([class1, class2], ignore_index=True)
    return train_test_split(data, data['class'], test_size=0.2, random_state=42, shuffle=True)


def plot_scatter_plot(train_data, test_data, prediction):
    colors = ['b', 'g', 'r']

    for i, j in train_data.groupby('class'):
        plt.scatter(j['A'], j['B'], c=colors[i],
                    label="class "+str(i)+" Train", alpha=0.8, s=10)

    colors = ['c', 'm', 'y']
    test_predict = test_data
    test_predict['class'] = prediction

    for i, j in test_predict.groupby('class'):
        plt.scatter(j['A'], j['B'], c=colors[i],
                    label="class "+str(i)+" Test", alpha=0.8, s=10)

    # Finally showing the scatter plot
    plt.title("Scatter Plot", fontsize=20)
    plt.xlabel("Attr 1", fontsize=14)
    plt.ylabel("Attr 2", fontsize=14)
    plt.legend()
    plt.show()


def question1(X_train, X_test, Y_train, Y_test):
    # K neighbours classifiers

    neigh = KMeans(2)
    neigh.fit(X_train.values, Y_train.values)
    prediction = neigh.predict(X_test.values)

    conf_matrix = confusion_matrix(Y_test, prediction)
    model_accuracy = accuracy_score(Y_test, prediction)

    print(" Confusion Matrix: \n", conf_matrix)
    print(" Accuracy:", model_accuracy)

    plot_scatter_plot(X_train, X_test, prediction)


[X_train2, X_test2, Y_train2, Y_test2] = build_data(
    'nls_data\class1.txt', 'nls_data\class2.txt')
question1(X_train2, X_test2, Y_train2, Y_test2)
