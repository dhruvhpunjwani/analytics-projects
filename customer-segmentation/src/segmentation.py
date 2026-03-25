import pandas as pd
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("../data/sample_data.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Reference date
reference_date = df["InvoiceDate"].max()

# RFM
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "CustomerID": "count",
    "Amount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm)

# Save output
rfm.to_csv("../outputs/segmentation_results.csv")

print(rfm)
