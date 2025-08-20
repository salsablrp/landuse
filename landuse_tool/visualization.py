import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_prediction(map_array, cmap_list, title="Prediction Map"):
    plt.figure(figsize=(10, 8))
    plt.imshow(map_array, cmap=ListedColormap(cmap_list))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
