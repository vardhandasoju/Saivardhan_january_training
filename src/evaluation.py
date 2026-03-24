from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate(data, labels):
    sil = silhouette_score(data, labels)
    db = davies_bouldin_score(data, labels)
    return sil, db