from sklearn.mixture import GaussianMixture

def run_gmm(data, k=4):
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(data)
    return labels