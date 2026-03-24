from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


# ---------------------------
# SCALING FUNCTION
# ---------------------------
def scale_data(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


# ---------------------------
# PCA FUNCTION
# ---------------------------
def run_pca(data):
    pca = PCA(n_components=2)
    return pca.fit_transform(data)


# ---------------------------
# CLUSTER VISUALIZATION
# ---------------------------
def plot_clusters(components, labels):
    os.makedirs("results/pca_outputs", exist_ok=True)

    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.title("Customer Segments")

    # Save PCA plot
    plt.savefig("results/pca_outputs/pca_plot.png")

    plt.show()


# ---------------------------
# ELBOW METHOD (IMPORTANT)
# ---------------------------
def elbow_method(data):
    os.makedirs("results/cluster_plots", exist_ok=True)

    inertia = []
    K = range(1, 10)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    plt.figure()
    plt.plot(K, inertia, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    # Save elbow plot
    plt.savefig("results/cluster_plots/elbow_plot.png")

    plt.show()