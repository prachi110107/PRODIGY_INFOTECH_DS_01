# =====================================================
# PRODIGY INFOTECH - TASK 03
# Decision Tree Classifier on Bank Marketing Dataset
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("bank.csv", sep=";")

print(df.head())
print(df.info())

# ----------------------------
# Encode Categorical Columns
# ----------------------------
encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = encoder.fit_transform(df[column])

# ----------------------------
# Features and Target
# ----------------------------
X = df.drop("y", axis=1)
y = df["y"]

# ----------------------------
# Split Dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Train Decision Tree
# ----------------------------
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# Accuracy
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("ConfusionMatrix.png")

plt.show()

# ----------------------------
# Feature Importance
# ----------------------------
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values()

plt.figure(figsize=(10,6))

importance.plot(kind="barh")

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("FeatureImportance.png")

plt.show()

# ----------------------------
# Decision Tree
# ----------------------------
plt.figure(figsize=(20,10))
plot_tree(
    model,
    filled=True,
    rounded=True,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    fontsize=8
)


plt.savefig("DecisionTree.png")

plt.show()

print("\nTask-03 Completed Successfully!")