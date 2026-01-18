import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------
# Load Dataset
# ----------------------------------------
df = pd.read_csv("data/zomato.csv")

# ----------------------------------------
# Define Features and Target
# ----------------------------------------
input_features = [
    "Road_traffic_density",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle"
]

target = "Time_taken (min)"

df = df[input_features + [target]]
df.dropna(inplace=True)

# ----------------------------------------
# Encode categorical features
# ----------------------------------------
encoder = LabelEncoder()
categorical_cols = [
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle"
]

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# ----------------------------------------
# FEATURE-WISE EDA
# ----------------------------------------
print("\n========= FEATURE-WISE EDA =========\n")

for feature in input_features:

    print(f"\n--- Feature: {feature} ---")

    # -----------------------------
    # 1. Distribution Plot
    # -----------------------------
    plt.figure(figsize=(6,4))
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.savefig(f"eda_distribution_{feature}.png")
    plt.show()

    # -----------------------------
    # 2. Skewness
    # -----------------------------
    skew_value = df[feature].skew()
    print(f"Skewness of {feature}: {skew_value}")

    # -----------------------------
    # 3. Outlier Detection (Boxplot)
    # -----------------------------
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[feature])
    plt.title(f"Outliers in {feature}")
    plt.savefig(f"eda_outliers_{feature}.png")
    plt.show()

    # -----------------------------
    # 4. Correlation with Target
    # -----------------------------
    corr_value = df[[feature, target]].corr().iloc[0,1]
    print(f"Correlation of {feature} with {target}: {corr_value}")

# ----------------------------------------
# Overall Correlation Heatmap
# ----------------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (All Features)")
plt.savefig("eda_correlation_heatmap_all_features.png")
plt.show()
