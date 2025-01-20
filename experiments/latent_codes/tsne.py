import argparse
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the features .npy file")
    args = parser.parse_args()

    input_file = np.load(args.input_file, allow_pickle=True)
    dic = input_file.item()

    features = dic['latent_vecs'].squeeze()

    labels = dic['class_ids']
    labels = [(label[0][0], label[1][0]) for label in labels]

    # assert features.shape[0] == labels.shape[0]

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(features)

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 3),
                    data=df).set(title="T-SNE projection")

    plt.show()
