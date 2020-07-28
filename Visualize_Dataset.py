import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
from sklearn.datasets import load_iris
dataset = load_iris()
features = dataset.data.T

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

sepal_length_label = dataset.feature_names[0]
sepal_width_label = dataset.feature_names[1]
petal_length_label = dataset.feature_names[2]
petal_width_label = dataset.feature_names[3]

plt.scatter(sepal_length, sepal_width, c=dataset.target)
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)
plt.show()
