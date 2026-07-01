import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(
    "API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv",
    skiprows=4
)

# Keep required columns
df = df[["Country Name", "Country Code", "2022"]]

# Remove missing values
df = df.dropna()

# Convert population to numeric
df["2022"] = pd.to_numeric(df["2022"])

# Remove aggregate regions
aggregates = [
    "WLD","HIC","LIC","LMC","MIC","UMC",
    "EAP","ECA","EUU","LCN","MEA",
    "NAC","SAS","SSF","ARB"
]

df = df[~df["Country Code"].isin(aggregates)]

# ==========================
# Histogram
# ==========================

plt.figure(figsize=(10,6))

plt.hist(
    df["2022"],
    bins=25,
    edgecolor="black"
)

plt.xscale("log")

plt.title(
    "Distribution of Country Populations (2022)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Population (Log Scale)")
plt.ylabel("Number of Countries")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("Histogram.png", dpi=300)

plt.show()

# ==========================
# Horizontal Bar Chart
# ==========================

top10 = df.sort_values(by="2022", ascending=False).head(10)

plt.figure(figsize=(10,6))

bars = plt.barh(
    top10["Country Name"],
    top10["2022"] / 1e9
)

plt.title(
    "Top 10 Most Populous Countries (2022)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Population (Billions)")

plt.gca().invert_yaxis()

# Add values on bars
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.02,
        bar.get_y() + bar.get_height()/2,
        f"{width:.2f}",
        va="center"
    )

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig("BarChart.png", dpi=300)

plt.show()

# ==========================
# Pie Chart
# ==========================

top5 = df.sort_values(by="2022", ascending=False).head(5)

plt.figure(figsize=(8,8))

plt.pie(
    top5["2022"],
    labels=top5["Country Name"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Population Share of Top 5 Countries (2022)",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig("PieChart.png", dpi=300)

plt.show()

# ==========================
# Box Plot
# ==========================

plt.figure(figsize=(8,6))

plt.boxplot(
    df["2022"],
    patch_artist=True
)

plt.yscale("log")

plt.title(
    "Population Distribution (Box Plot)",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel("Population (Log Scale)")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("BoxPlot.png", dpi=300)

plt.show()

print("🎉 Task 01 completed successfully!")