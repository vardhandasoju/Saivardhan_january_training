from sklearn.cluster import DBSCAN

def run_dbscan(data):
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(data)
    return labels