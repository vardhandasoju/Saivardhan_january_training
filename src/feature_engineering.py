def create_features(df):
    df["Frequency"] = (
        df["NumWebPurchases"] +
        df["NumStorePurchases"] +
        df["NumCatalogPurchases"]
    )
    df["Monetary"] = (
        df["MntWines"] +
        df["MntFruits"] +
        df["MntMeatProducts"] +
        df["MntFishProducts"] +
        df["MntSweetProducts"] +
        df["MntGoldProds"]
    )
    df.rename(columns={"Recency":"Recency_days"}, inplace=True)
    return df