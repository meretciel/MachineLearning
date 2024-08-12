import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for tt in range(100):
    print(f"processing {tt}")
    fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(8, 6))

    X = np.arange(-2, 2.005, 0.005)
    Y = np.arange(-2, 2.005, 0.005)

    X, Y = np.meshgrid(X, Y)
    Z = X * X - Y * Y + np.sin(X) - np.sin(Y * Y) - 0.2 * X + Y ** 4
    levels = np.linspace(Z.min(), Z.max(), 50)

    CS = axs[0].contourf(X, Y, Z, levels=levels)
    fig.colorbar(CS, ax=axs[0])

    df = pd.read_csv(f"/home/ryan/workspace/tmp/gradient_descent_output_{tt}.csv")
    prev_x = None
    prev_y = None
    for k, row in df.iterrows():
        x = row['x']
        y = row['y']
        if prev_x is not None and prev_y is not None:
            axs[0].plot([prev_x, x], [prev_y, y], linewidth=1, color='grey')

        axs[0].scatter(x, y, s=5, color='orange')

        prev_x = x
        prev_y = y

    axs[0].scatter(df.iloc[-1, 0], df.iloc[-1, 1], marker='*', color='red', s=200, zorder=100)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    axs[1].plot(np.arange(df.shape[0]), df['value'].values)

    # plt.show()
    plt.savefig(f"/home/ryan/workspace/tmp/gradient_descent_output_v2_{tt}.png")
