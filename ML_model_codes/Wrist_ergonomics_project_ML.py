# Wrist_ergonomics_project_ML.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv("imu_features.csv")

# Clean column names and string columns
df.columns = df.columns.str.strip()
for col in ["Movement", "Condition"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()


# 2. Map movement names to numbers

movement_map = {
    "Pegs diagonally placed": 1,
    "Pegs in a straight line": 2
}

df["Movement_Code"] = df["Movement"].map(movement_map)
df = df.dropna(subset=["Movement_Code"])
df["Movement_Code"] = df["Movement_Code"].astype(int)

print("\nMovement mapping:")
print(df[["Movement", "Movement_Code"]].drop_duplicates())

# 3. Create new features from the data

# Calculate overall acceleration and gyro magnitude
df["ACC_MAG"] = np.sqrt(
    df["ACC X (G)_mean"]**2 +
    df["ACC Y (G)_mean"]**2 +
    df["ACC Z (G)_mean"]**2
)

df["GYRO_MAG"] = np.sqrt(
    df["GYRO X (deg/s)_mean"]**2 +
    df["GYRO Y (deg/s)_mean"]**2 +
    df["GYRO Z (deg/s)_mean"]**2
)

# Jerk = change in acceleration between frames
df["JERK"] = df["ACC_MAG"].diff().abs().fillna(0)

# Keep all mean, std, max, min columns + new features
feature_cols = [col for col in df.columns if "_mean" in col or "_std" in col or "_max" in col or "_min" in col]
feature_cols.extend(["ACC_MAG", "GYRO_MAG", "JERK"])


# 4. Machine learning - predict movement type

X = df[feature_cols].values
y = df["Movement_Code"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Make labels 0 and 1 for binary classification
y=y-1

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost model
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n============================")
print("MACHINE LEARNING RESULTS")
print("============================")
print(f"ML Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. ANOVA and compare movements

def analyze_subset(subset_df, subset_name):
    move1 = subset_df[subset_df["Movement_Code"]==1]
    move2 = subset_df[subset_df["Movement_Code"]==2]

    print(f"\n============================")
    print(f"{subset_name} - ANOVA & MOVEMENT COMPARISON")
    print("============================")
    feature_cols =["ACC_MAG", "GYRO_MAG", "JERK"]
    # ANOVA test
    for metric in feature_cols:
        try:
            F, p = f_oneway(move1[metric], move2[metric])
            print(f"{metric}: F={F:.4f}, p={p:.5f}")
        except:
            print(f"{metric}: ANOVA could not be computed.")

     # Compare movements
    move1_wins = 0
    move2_wins = 0
    for metric, label in zip(feature_cols, ["Overall Acceleration", "Rotational Load", "Smoothness (Jerk)"]):
        m1 = move1[metric].mean()
        m2 = move2[metric].mean()
        better = "Pegs diagonally placed" if m1 < m2 else "Pegs in a straight line"
        print(f"\n{label}:")
        print(f"Pegs diagonally placed mean = {m1:.4f}")
        print(f"Pegs in a straight line mean = {m2:.4f}")
        print(f"Better movement based on {label} = {better}")

        # Count wins
        if m1 < m2:
            move1_wins += 1
        else:
            move2_wins += 1

    overall = "Pegs diagonally placed" if move1_wins > move2_wins else "Pegs in a straight line"
    print(f"\n Overall best movement for {subset_name}: {overall} (Move1 wins={move1_wins}, Move2 wins={move2_wins})")


# 6. Analyze Exo and NoExo separately

analyze_subset(df[df["Condition"] == "Exo"], "Exo")
analyze_subset(df[df["Condition"] == "NoExo"], "No-Exo")


# 7. Compare Exo vs NoExo for each movement

print("\n============================")
print("EXO METRIC SUMMARY PER MOVEMENT")
print("============================")

movements = ["Pegs diagonally placed", "Pegs in a straight line"]
metrics = ["ACC_MAG", "GYRO_MAG", "JERK"]

for movement_name in movements:
    print(f"\nMovement: {movement_name}")
    
    mv = df[df["Movement"] == movement_name]
    exo = mv[mv["Condition"] == "Exo"]
    noexo = mv[mv["Condition"] == "NoExo"]
    
    for metric in metrics:
        # Exo stats
        exo_mean = exo[metric].mean()
        exo_std  = exo[metric].std()
        exo_min  = exo[metric].min()
        exo_max  = exo[metric].max()
        
        # No-Exo stats
        no_mean = noexo[metric].mean()
        no_std  = noexo[metric].std()
        no_min  = noexo[metric].min()
        no_max  = noexo[metric].max()
        
        print(f"  {metric}:")
        print(f"    Exo   : mean = {exo_mean:.4f}, std = {exo_std:.4f}, "
              f"min = {exo_min:.4f}, max = {exo_max:.4f}")
        print(f"    No-Exo: mean = {no_mean:.4f}, std = {no_std:.4f}, "
              f"min = {no_min:.4f}, max = {no_max:.4f}")
       

# ======================================================
# 8. Plotting results
# ======================================================
def simple_boxplot(metric, ylabel):
    plt.figure(figsize=(6,4))
    sns.boxplot(
        x="Movement", y=metric, hue="Condition",
        data=df
    )
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.title(f"{metric} by Movement and Condition")
    plt.legend(title="Condition", loc="upper right")
    plt.tight_layout()
    
    # Save the plot as a PNG file
    plt.savefig(f"{metric}_boxplot.png")
    plt.close()  # Close figure to avoid overlapping plots

# Save all three plots
simple_boxplot("ACC_MAG", "Acceleration magnitude (G)")
simple_boxplot("GYRO_MAG", "Gyro magnitude (deg/s)")
simple_boxplot("JERK", "Jerk")

print("\nAll plots saved as PNG files!")


print("\n===================================================================")
print("All analysis, ML, ANOVA, comparisons, and plots completed successfully!")
print("===================================================================\n")