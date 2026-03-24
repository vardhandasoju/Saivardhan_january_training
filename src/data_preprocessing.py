import pandas as pd
from sklearn.preprocessing import LabelEncoder
def load_data(path):
    return pd.read_csv(path, sep="\t")
def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df["Age"] = 2026 - df["Year_Birth"]
    df = df.drop(["ID","Year_Birth","Dt_Customer"], axis=1)
    le = LabelEncoder()
    df["Education"] = le.fit_transform(df["Education"])
    df["Marital_Status"] = le.fit_transform(df["Marital_Status"])
    return df