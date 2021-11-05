# %% Imports

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# %% q2 funcs

def KMeansCluster(vectorized, K=5):
    # print(vectorized)
    # K means clustering using the vectorized linear array

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

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



# %% q1


def build_data(class1_path, class2_path):
    class1 = pd.read_csv(class1_path, header=None)
    class1.columns = ['A', 'B']
    class1['class'] = 0

    class2 = pd.read_csv(class2_path, header=None)
    class2.columns = ['A', 'B']
    class2['class'] = 1

    data = pd.concat([class1, class2], ignore_index=True).sample(frac=1).reset_index(drop=True)

    return data[data.columns[:2]], data['class']


def question1(X_train, Y_train):

    model = KMeans(2)
    model.fit(X_train.values)
    preds = model.labels_
	
    print("Confusion Matrix: \n", confusion_matrix(preds, Y_train))
    print("Accuracy:", accuracy_score(preds, Y_train))

    X_train["Class"] = preds

    for i, j in X_train.groupby('Class'):
        plt.scatter(j['A'], j['B'], label="class "+str(i)+" Test", alpha=0.8, s=10)

    centres = model.cluster_centers_
    plt.scatter(centres[:,0], centres[:,1], s = 50)
    # Finally showing the scatter plot
    plt.title("Scatter Plot", fontsize=20)
    plt.xlabel("Attr 1", fontsize=14)
    plt.ylabel("Attr 2", fontsize=14)
    plt.legend()
    plt.show()


X_train, X_test = build_data('nls_data\class1.txt', 'nls_data\class2.txt')
question1(X_train, X_test)



# %%
