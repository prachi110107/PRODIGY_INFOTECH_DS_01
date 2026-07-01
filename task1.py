import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    "API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv",
    skiprows=4
)

# Keep only required columns
df = df[["Country Name", "Country Code", "2022"]]

# Remove rows with missing population
df = df.dropna()

# Convert population column to numeric
df["2022"] = pd.to_numeric(df["2022"])

# Remove aggregate entries
aggregates = [
    "WLD","HIC","LIC","LMC","MIC","UMC","EAP","ECA","EUU",
    "LCN","MEA","NAC","SAS","SSF","ARB"
]

df = df[~df["Country Code"].isin(aggregates)]

# ---------------- Histogram ----------------

plt.figure(figsize=(10,6))

plt.hist(
    df["2022"],
    bins=25,
    edgecolor="black"
)

plt.xscale("log")

plt.title("Distribution of Country Populations (2022)",
          fontsize=16,
          fontweight="bold")

plt.xlabel("Population (Log Scale)")
plt.ylabel("Number of Countries")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("Histogram.png", dpi=300)

plt.show()

# ---------------- Bar Chart ----------------

top10 = df.sort_values(by="2022", ascending=False).head(10)

plt.figure(figsize=(11,6))

plt.bar(
    top10["Country Name"],
    top10["2022"]/1e9
)

plt.title("Top 10 Most Populous Countries (2022)",
          fontsize=16,
          fontweight="bold")

plt.xlabel("Country")
plt.ylabel("Population (Billions)")

plt.xticks(rotation=45)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig("BarChart.png", dpi=300)

plt.show()

print("Task 01 completed successfully!")