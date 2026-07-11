import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("US_Accidents_March23.csv", nrows=100000)

print(df.head())
print(df.info())

# -----------------------------
# Select Useful Columns
# -----------------------------
columns = [
    'Severity',
    'Start_Time',
    'Weather_Condition',
    'State',
    'Start_Lat',
    'Start_Lng'
]

df = df[columns]

# Remove missing values
df.dropna(inplace=True)

# Convert Start_Time to datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

# Extract Hour
df['Hour'] = df['Start_Time'].dt.hour

sns.set_style("whitegrid")

# ---------------------------------------
# Graph 1 : Severity Distribution
# ---------------------------------------
plt.figure(figsize=(7,5))
sns.countplot(x='Severity', data=df)
plt.title("Severity Distribution")
plt.savefig("Severity_Distribution.png")
plt.show()

# ---------------------------------------
# Graph 2 : Weather Condition
# ---------------------------------------
top_weather = df['Weather_Condition'].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(
    x=top_weather.values,
    y=top_weather.index
)
plt.title("Top 10 Weather Conditions")
plt.xlabel("Accident Count")
plt.ylabel("Weather")
plt.savefig("Accidents_By_Weather.png")
plt.show()

# ---------------------------------------
# Graph 3 : Hour of Day
# ---------------------------------------
plt.figure(figsize=(10,5))
sns.countplot(x='Hour', data=df)
plt.title("Accidents by Hour")
plt.savefig("Accidents_By_Hour.png")
plt.show()

# ---------------------------------------
# Graph 4 : Top 10 States
# ---------------------------------------
top_states = df['State'].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(
    x=top_states.values,
    y=top_states.index
)

plt.title("Top 10 States with Most Accidents")
plt.xlabel("Accident Count")
plt.ylabel("State")
plt.savefig("Top_States.png")
plt.show()

# ---------------------------------------
# Graph 5 : Accident Hotspots
# ---------------------------------------
sample = df.sample(5000, random_state=42)

plt.figure(figsize=(10,8))
plt.scatter(
    sample['Start_Lng'],
    sample['Start_Lat'],
    s=1,
    alpha=0.5
)

plt.title("Accident Hotspots")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("Accident_Hotspots.png")
plt.show()

# ---------------------------------------
# Graph 6 : Correlation Heatmap
# ---------------------------------------
corr = df[['Severity','Start_Lat','Start_Lng','Hour']].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("Correlation_Heatmap.png")
plt.show()

print("Task 05 Completed Successfully!") 
