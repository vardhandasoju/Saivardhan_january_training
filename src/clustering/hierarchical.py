from sklearn.cluster import AgglomerativeClustering

def run_hierarchical(data, k=4):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(data)
    return labels