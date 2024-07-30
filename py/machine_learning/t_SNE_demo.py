import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

from sklearn.datasets import load_digits
digits = load_digits()
data = digits['data']
print(f"data shape: {data.shape}")

# Generate sample images
# fig, ax = plt.subplots(1, 4)
#
# for i in range(4):
#     ax[i].imshow(digits.images[i], cmap='Greys')
# plt.show()

y_digits = digits.target
X_digits = digits.data

tsne = TSNE(n_components=2, init='pca', random_state=100)
X_digits_tsne = tsne.fit_transform(X_digits)
assert X_digits_tsne.shape == (X_digits.shape[0], 2)

f = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')

for i in range(10):
    plt.scatter(X_digits_tsne[y_digits == i, 0],
                X_digits_tsne[y_digits == i, 1])

for i in range(10):
    xtext, ytext = np.median(X_digits_tsne[y_digits == i, :], axis=0)
    txt = ax.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])

plt.show()
