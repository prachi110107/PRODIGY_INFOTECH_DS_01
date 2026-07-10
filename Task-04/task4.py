# ==========================================================
# PRODIGY INFOTECH
# Task-04
# Analyze and visualize sentiment patterns in social media
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

columns = ["Tweet_ID", "Entity", "Sentiment", "Tweet"]

df = pd.read_csv(
    "twitter_training.csv",   # Change if your filename is different
    names=columns,
    header=None
)

# ----------------------------------------------------------
# Display Dataset Information
# ----------------------------------------------------------

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------------------------------------
# Remove Missing Values
# ----------------------------------------------------------

df.dropna(inplace=True)

# ----------------------------------------------------------
# Sentiment Distribution
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Sentiment",
    order=df["Sentiment"].value_counts().index
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("Sentiment_Distribution.png")
plt.show()
plt.close()

# ----------------------------------------------------------
# Top 10 Most Mentioned Brands/Entities
# ----------------------------------------------------------

top_entities = df["Entity"].value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_entities.values,
    y=top_entities.index
)

plt.title("Top 10 Most Mentioned Brands")
plt.xlabel("Number of Tweets")
plt.ylabel("Brand")

plt.tight_layout()

plt.savefig("Top_Entities.png")
plt.show()
plt.close()

# ----------------------------------------------------------
# Sentiment by Top 5 Brands
# ----------------------------------------------------------

top5 = df["Entity"].value_counts().head(5).index

filtered = df[df["Entity"].isin(top5)]

plt.figure(figsize=(12,6))

sns.countplot(
    data=filtered,
    x="Entity",
    hue="Sentiment"
)

plt.title("Sentiment by Top 5 Brands")

plt.tight_layout()

plt.savefig("Sentiment_By_Entity.png")
plt.show()
plt.close()

# ----------------------------------------------------------
# Pie Chart of Sentiments
# ----------------------------------------------------------

sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(7,7))

plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Overall Sentiment Distribution")

plt.savefig("Sentiment_PieChart.png")
plt.show()
plt.close()

# ----------------------------------------------------------
# Heatmap of Sentiment Counts
# ----------------------------------------------------------

pivot = pd.crosstab(df["Entity"], df["Sentiment"])

top10 = pivot.head(10)

plt.figure(figsize=(10,6))

sns.heatmap(
    top10,
    annot=True,
    cmap="YlGnBu",
    fmt="d"
)

plt.title("Sentiment Heatmap (Top 10 Entities)")

plt.tight_layout()

plt.savefig("Sentiment_Heatmap.png")
plt.show()
plt.close()

print("\nAnalysis Completed Successfully!")
print("Graphs have been saved in your Task-04 folder.")