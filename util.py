import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_mnist_csv(filepath):
    df = pd.read_csv(filepath)
    
    y = df.iloc[:, 0].values
    Y = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1.0
    X = df.iloc[:, 1:].values
    
    X = X / 255.0
    
    return X, Y

def mnist_visualize(
        data,
        index: int = 0,
        num_images: int = 10,
        cols: int = 5,
        figsize: tuple | None = None,
        titles: list[str] | None = None,
        cmap: str = "gray",
        title_fmt: str = "Label: {}",
        title_fontsize: int = 10,
        h_pad: float = 1.0,
        w_pad: float = 0.3,
):
    X, y = data
    rows = int(np.ceil(num_images / cols))

    if figsize is None:
        figsize = (2.2 * cols, 2.4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(num_images):
        img_idx = index + i
        ax = axes[i]

        if img_idx < len(X):
            img = X[img_idx].reshape(28, 28)
            ax.imshow(img, cmap=cmap)
            
            # Choose title
            if titles is not None:
                title = titles[i]
            else:
                title = title_fmt.format(y[img_idx])

            ax.set_title(title, fontsize=title_fontsize, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(num_images, rows * cols):
        axes[j].axis("off")

    plt.tight_layout(h_pad=h_pad, w_pad=w_pad)
    plt.show()