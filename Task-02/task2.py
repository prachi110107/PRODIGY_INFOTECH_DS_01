# ==========================================================
# PRODIGY INFOTECH
# Data Science Internship
# Task-02 : Data Cleaning and Exploratory Data Analysis
# Dataset : Titanic Dataset
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Plot Settings
# -----------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("train.csv")

print("="*60)
print("TITANIC DATASET LOADED SUCCESSFULLY")
print("="*60)

print("\nFirst Five Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# -----------------------------
# Data Cleaning
# -----------------------------

# Fill Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column
df.drop(columns=["Cabin"], inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# -----------------------------
# Feature Engineering
# -----------------------------

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# ==========================================================
# GRAPH 1 : Survival Count
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    x="Survived",
    data=df,
    palette="Set2"
)

plt.title("Passenger Survival Count",fontsize=15,fontweight="bold")
plt.xlabel("Survived")
plt.ylabel("Passengers")

plt.tight_layout()
plt.savefig("Survival_Count.png",dpi=300)

plt.show()

# ==========================================================
# GRAPH 2 : Survival by Gender
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df,
    palette="viridis"
)

plt.title("Survival by Gender",fontsize=15,fontweight="bold")

plt.tight_layout()
plt.savefig("Survival_by_Gender.png",dpi=300)

plt.show()

# ==========================================================
# GRAPH 3 : Passenger Class
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=df,
    palette="Set1"
)

plt.title("Passenger Class vs Survival",fontsize=15,fontweight="bold")

plt.tight_layout()
plt.savefig("Passenger_Class.png",dpi=300)

plt.show()

# ==========================================================
# GRAPH 4 : Age Distribution
# ==========================================================

plt.figure(figsize=(10,6))

sns.histplot(
    df["Age"],
    bins=30,
    kde=True,
    color="royalblue"
)

plt.title("Age Distribution",fontsize=15,fontweight="bold")

plt.tight_layout()
plt.savefig("Age_Distribution.png",dpi=300)

plt.show()

# ==========================================================
# GRAPH 5 : Fare Distribution
# ==========================================================

plt.figure(figsize=(10,6))

sns.histplot(
    df["Fare"],
    bins=30,
    kde=True,
    color="orange"
)

plt.title("Fare Distribution",fontsize=15,fontweight="bold")

plt.tight_layout()
plt.savefig("Fare_Distribution.png",dpi=300)

plt.show()

# ==========================================================
# GRAPH 6 : Box Plot (Age)
# ==========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["Age"],
    color="lightgreen"
)

plt.title("Age Box Plot",fontsize=15,fontweight="bold")

plt.tight_layout()
plt.savefig("Boxplot_Age.png",dpi=300)

plt.show()


# ==========================================================
# GRAPH 7 : Embarked Port vs Survival
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x="Embarked",
    hue="Survived",
    data=df,
    palette="Dark2"
)

plt.title("Embarked Port vs Survival", fontsize=15, fontweight="bold")
plt.xlabel("Embarked Port")
plt.ylabel("Passenger Count")

plt.tight_layout()
plt.savefig("Embarked_Count.png", dpi=300)

plt.show()


# ==========================================================
# GRAPH 8 : Family Size Distribution
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x="FamilySize",
    data=df,
    palette="pastel"
)

plt.title("Family Size Distribution", fontsize=15, fontweight="bold")
plt.xlabel("Family Size")
plt.ylabel("Passengers")

plt.tight_layout()
plt.savefig("Family_Size.png", dpi=300)

plt.show()


# ==========================================================
# GRAPH 9 : Correlation Heatmap
# ==========================================================

plt.figure(figsize=(10,8))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Correlation Heatmap", fontsize=15, fontweight="bold")

plt.tight_layout()
plt.savefig("Correlation_Heatmap.png", dpi=300)

plt.show()


# ==========================================================
# GRAPH 10 : Survival Pie Chart
# ==========================================================

survival = df["Survived"].value_counts()

plt.figure(figsize=(7,7))

plt.pie(
    survival,
    labels=["Did Not Survive", "Survived"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["tomato", "lightgreen"],
    wedgeprops={"edgecolor":"black"}
)

plt.title("Passenger Survival Percentage", fontsize=15, fontweight="bold")

plt.tight_layout()
plt.savefig("Survival_PieChart.png", dpi=300)

plt.show()


# ==========================================================
# GRAPH 11 : Pair Plot
# ==========================================================

pair = sns.pairplot(
    df[["Age", "Fare", "Pclass", "Survived"]],
    hue="Survived",
    palette="husl"
)

pair.fig.suptitle("Pair Plot of Important Features", y=1.02, fontsize=16)

pair.savefig("Pairplot.png", dpi=300)

plt.show()


# ==========================================================
# GRAPH 12 : Survival Rate by Passenger Class
# ==========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x="Pclass",
    y="Survived",
    data=df,
    palette="Blues"
)

plt.title("Average Survival Rate by Passenger Class", fontsize=15, fontweight="bold")
plt.xlabel("Passenger Class")
plt.ylabel("Average Survival Rate")

plt.tight_layout()
plt.savefig("SurvivalRate_PClass.png", dpi=300)

plt.show()


# ==========================================================
# GRAPH 13 : Survival Rate by Embarked Port
# ==========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x="Embarked",
    y="Survived",
    data=df,
    palette="Set3"
)

plt.title("Average Survival Rate by Embarked Port", fontsize=15, fontweight="bold")
plt.xlabel("Embarked Port")
plt.ylabel("Average Survival Rate")

plt.tight_layout()
plt.savefig("SurvivalRate_Embarked.png", dpi=300)

plt.show()


# ==========================================================
# FINAL INSIGHTS
# ==========================================================

print("\n" + "="*70)
print("                 TITANIC DATASET - KEY INSIGHTS")
print("="*70)

print("""
1. Female passengers had a significantly higher survival rate than males.

2. First-class passengers were more likely to survive than those in
   second and third class.

3. Younger passengers showed a slightly better survival rate.

4. Passengers who paid higher fares generally had better chances of survival.

5. Southampton (S) was the most common embarkation port.

6. Most passengers travelled with small family sizes.

7. The Age feature contained missing values which were filled using the median.

8. The Cabin column had many missing values and was removed during cleaning.

9. Correlation analysis indicates that Passenger Class and Fare are important
   features influencing survival.
""")

print("="*70)
print("TASK-02 COMPLETED SUCCESSFULLY")
print("="*70)

print("""
Generated Output Files:

✔ Survival_Count.png
✔ Survival_by_Gender.png
✔ Passenger_Class.png
✔ Age_Distribution.png
✔ Fare_Distribution.png
✔ Boxplot_Age.png
✔ Embarked_Count.png
✔ Family_Size.png
✔ Correlation_Heatmap.png
✔ Survival_PieChart.png
✔ Pairplot.png
✔ SurvivalRate_PClass.png
✔ SurvivalRate_Embarked.png

All graphs have been saved successfully.
""") 
