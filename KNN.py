import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

penguins = pd.read_csv(
    "https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")

penguins = penguins.dropna()
# counts = penguins.species_short.value_counts()
Adelies = penguins.loc[penguins["species_short"] == "Adelie"]
Gentoos = penguins.loc[penguins["species_short"] == "Gentoo"]
Chinstraps = penguins.loc[penguins["species_short"] == "Chinstrap"]

X_train_dataframe = pd.concat([Adelies.iloc[:100, :], Gentoos.iloc[:80, :], Chinstraps.iloc[:50, :]])
X_test_dataframe = pd.concat([Adelies.iloc[100:, :], Gentoos.iloc[80:, :], Chinstraps.iloc[50:, :]])


def most_frequent(List):
    return max(set(List), key=List.count)


def euclidean(p1, p2):
    dist = np.sqrt(np.sum((p1 - p2) ** 2))
    return dist


def dist_neigh(X_train, y_train, X_test, k):
    dists_all = []
    labels_all = []
    for item in X_test:
        point_dist = []
        for j in range(len(X_train)):
            distances = euclidean(X_train[j, :], item)
            point_dist.append(distances)
        point_dist = np.array(point_dist)
        dist = np.argsort(point_dist)[:k]
        labels = y_train[dist]
        dists_all.append(dist)
        labels_all.append(labels)
    return np.array(dists_all), labels_all


def predict(X_train, y_train, X_test, y_test, k):
    dists, labels_all = dist_neigh(X_train, y_train, X_test, k)
    y_pred = []
    for item in labels_all:
        y_pred.append(most_frequent(item.tolist()))
    return accuracy_score(y_test, y_pred)


X_train = X_train_dataframe[
    [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
].values

y_train_labels_vector = X_train_dataframe.species_short.map({"Adelie": 0, "Gentoo": 1, "Chinstrap": 2})
y_train = y_train_labels_vector.to_numpy()

X_test = X_test_dataframe[
    [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
].values

y_test_labels_vector = X_test_dataframe.species_short.map({"Adelie": 0, "Gentoo": 1, "Chinstrap": 2})
y_test = y_test_labels_vector.to_numpy()

print("KNN for all features:")
k = 1
score = predict(X_train, y_train, X_test, y_test, k)
print("\nCorrect Prediction Rate for k={} is: ".format(k), score)

k = 3
score = predict(X_train, y_train, X_test, y_test, k)
print("Correct Prediction Rate for k={} is: ".format(k), score)

k = 5
score = predict(X_train, y_train, X_test, y_test, k)
print("Correct Prediction Rate for k={} is: ".format(k), score)

X_train = X_train_dataframe[
    [
        "culmen_length_mm",
        "flipper_length_mm",
    ]
].values

y_train_labels_vector = X_train_dataframe.species_short.map({"Adelie": 0, "Gentoo": 1, "Chinstrap": 2})
y_train = y_train_labels_vector.to_numpy()

X_test = X_test_dataframe[
    [
        "culmen_length_mm",
        "flipper_length_mm",

    ]
].values

y_test_labels_vector = X_test_dataframe.species_short.map({"Adelie": 0, "Gentoo": 1, "Chinstrap": 2})
y_test = y_test_labels_vector.to_numpy()

print("\n\nKNN for features: ['culmen_length_mm', 'flipper_length_mm']:")
k = 1
score = predict(X_train, y_train, X_test, y_test, k)
print("\nCorrect Prediction Rate for k={} is: ".format(k), score)

k = 3
score = predict(X_train, y_train, X_test, y_test, k)
print("Correct Prediction Rate for k={} is: ".format(k), score)

k = 5
score = predict(X_train, y_train, X_test, y_test, k)
print("Correct Prediction Rate for k={} is: ".format(k), score)
