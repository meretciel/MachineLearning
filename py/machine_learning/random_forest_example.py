import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from machine_learning.visualization.utils import plot_decision_regions
from machine_learning.visualization.colors import get_color_palette
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


data_dir = Path('/home/ryan/workspace/projects/MachineLearning/data/iris')
output_dir = Path('/home/ryan/workspace/projects/MachineLearning/data/output')


data = pd.read_csv(data_dir / "iris.data", header=None)
feature_names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
data.columns = ["x1", "x2", "x3", "x4", "y_str"]

label_encoder = LabelEncoder()
label_encoder.fit(data["y_str"])
n = len(label_encoder.classes_)
print(f"Number of classes: {n}")

# Geneal Setting
random_state = 1        # random seed
test_size = 0.3         # used in the training and testing split

# Prepare training and testing data
y = label_encoder.transform(data["y_str"])
# X = data[["x1", "x2", "x3", "x4"]].values
X = data[["x1", "x2"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train_std, y_train)


fig, axs = plt.subplots(1, 2, layout='constrained')
x1_min = X_train_std[:, 0].min() - 1
x1_max = X_train_std[:, 0].max() + 1
x2_min = X_train_std[:, 1].min() - 1
x2_max = X_train_std[:, 1].max() + 1

plot_decision_regions(axs[0], X_train_std[:, 0], X_train_std[:, 1], y_train, forest, marker_alpha=1.0)
axs[0].set_title("Training")
plot_decision_regions(axs[1], X_test_std[:, 0], X_test_std[:, 1], y_test, forest, x1_range=(x1_min, x1_max), x2_range=(x2_min, x2_max), marker_alpha=1.0)
axs[1].set_title("Test")

fig.suptitle("Random Forest")
fig.align_labels()
plt.show()
