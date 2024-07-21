import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from machine_learning.visualization.colors import get_color_palette
from machine_learning.visualization.markers import get_marker_list


def plot_decision_regions(ax, x1, x2, y, classifier, x1_range=None, x2_range=None, resolution=0.02, alpha=0.26,
                          marker_alpha=None):
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    colors = get_color_palette(n_classes)
    markers = get_marker_list(n_classes)
    cmap = ListedColormap(colors)

    if x1_range is None:
        x1_min = x1.min() - 1
        x1_max = x1.max() + 1
    else:
        x1_min, x1_max = x1_range

    if x2_range is None:
        x2_min = x2.min() - 1
        x2_max = x2.max() + 1
    else:
        x2_min, x2_max = x2_range

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    data = np.array([xx1.ravel(), xx2.ravel()]).T
    label = classifier.predict(data).reshape(xx1.shape)
    print(label)
    ax.contourf(xx1, xx2, label, alpha=alpha, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    if marker_alpha is None:
        marker_alpha = min(1., alpha * 2 + 0.1)

    for cl in range(n_classes):
        ax.scatter(x=x1[y == cl], y=x2[y == cl], alpha=marker_alpha,
                   c=colors[cl], marker=markers[cl], label=f'Class {cl}', edgecolor="black")
