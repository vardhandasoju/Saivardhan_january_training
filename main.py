from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import create_features
from src.clustering.kmeans import run_kmeans
from src.evaluation import evaluate
from src.utils import scale_data, run_pca, plot_clusters, elbow_method

import os

DATA_PATH = "data/raw/marketing_campaign.csv"

def main():
    print(" Starting Project")

    # LOAD DATA
    print("Loading data...")
    df = load_data(DATA_PATH)

    # PREPROCESSING
    
    print("Cleaning data...")
    df = clean_data(df)

    
    
    # FEATURE ENGINEERING
   
   
    print("Feature engineering...")
    df = create_features(df)

    features = df[["Age","Income","Recency_days","Frequency","Monetary"]]

    
    
    # SCALING
    
    
    print("Scaling data...")
    data = scale_data(features)

    
    
    # ELBOW METHOD (IMPORTANT)
   
   
    print("Running Elbow Method...")
    elbow_method(data)

    
    
    # KMEANS
    
    
    print("Running KMeans...")
    labels, model = run_kmeans(data)

  
  
    # EVALUATION
    
    
    print("Evaluating model...")
    sil, db = evaluate(data, labels)

    print("Silhouette Score:", sil)
    print("Davies Bouldin Score:", db)

    
    
    # SAVE METRICS
    
    
    os.makedirs("results/metrics", exist_ok=True)

    with open("results/metrics/scores.txt", "w") as f:
        f.write("Model Evaluation Results\n")
        f.write("------------------------\n")
        f.write(f"Silhouette Score: {sil}\n")
        f.write(f"Davies Bouldin Score: {db}\n")


    # SAVE CLUSTERED DATA
    
    
    os.makedirs("results/cluster_plots", exist_ok=True)

    df["Cluster"] = labels
    df.to_csv("results/cluster_plots/clustered_data.csv", index=False)

  
  
    # PCA + VISUALIZATION
  
  
    print("Running PCA...")
    components = run_pca(data)

    print("Plotting clusters...")
    plot_clusters(components, labels)

    print("✅ Project Completed Successfully!")

if __name__ == "__main__":
    main()