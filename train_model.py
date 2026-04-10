# 
# Disease Prediction System using Machine Learning
# GTU Internship Project | Healthcare Domain
# 
# Dataset: Pima Indians Diabetes Dataset (UCI / Kaggle)
# Download: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# OR use: sklearn's built-in dataset loader (used here for zero-setup)
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes   # fallback; we simulate Pima below
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import pickle
import warnings
warnings.filterwarnings("ignore")

# 
# STEP 1 – LOAD DATASET
# 
# We recreate the Pima Indians Diabetes dataset column structure.
# To use the real CSV: df = pd.read_csv("data/diabetes.csv")
np.random.seed(42)
n = 768
df = pd.DataFrame({
    "Pregnancies":        np.random.randint(0, 18, n),
    "Glucose":            np.random.randint(0, 200, n),
    "BloodPressure":      np.random.randint(0, 122, n),
    "SkinThickness":      np.random.randint(0, 100, n),
    "Insulin":            np.random.randint(0, 846, n),
    "BMI":                np.round(np.random.uniform(0, 67, n), 1),
    "DiabetesPedigreeFunction": np.round(np.random.uniform(0.07, 2.5, n), 3),
    "Age":                np.random.randint(21, 82, n),
})
# Realistic outcome: higher glucose / BMI → more likely diabetic
prob = (df["Glucose"] / 200 * 0.5 + df["BMI"] / 67 * 0.3 +
        df["Age"] / 82 * 0.2)
df["Outcome"] = (prob > np.random.uniform(0.35, 0.65, n)).astype(int)

print("=" * 60)
print("  DISEASE PREDICTION SYSTEM — DIABETES DATASET")
print("=" * 60)
print(f"\n Dataset shape : {df.shape}")
print(f"   Columns       : {list(df.columns)}")
print(f"   Class balance :\n{df['Outcome'].value_counts()}\n")

# 
# STEP 2 – EDA (Exploratory Data Analysis)
# 
print("─" * 60)
print("STEP 2 : Exploratory Data Analysis")
print("─" * 60)
print("\nBasic statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# Columns where 0 actually means missing in the medical context
zero_as_null = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print(f"\nZero counts (medically impossible zeroes):")
for col in zero_as_null:
    print(f"  {col:30s}: {(df[col] == 0).sum()} zeros")

# ── Plot 1 : Feature distributions ──────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold")
for ax, col in zip(axes.flatten(), df.columns):
    color = "#E74C3C" if col == "Outcome" else "#3498DB"
    df[col].hist(ax=ax, bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig("models/eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: models/eda_distributions.png")

# ── Plot 2 : Correlation heatmap ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, ax=ax,
            linewidths=0.5, linecolor="white")
ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("models/eda_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: models/eda_correlation.png")

# ── Plot 3 : Outcome distribution ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df["Outcome"].value_counts().plot.pie(
    ax=axes[0], autopct="%1.1f%%",
    labels=["No Diabetes", "Diabetes"],
    colors=["#2ECC71", "#E74C3C"], startangle=90)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("")
sns.boxplot(data=df, x="Outcome", y="Glucose",
            palette={"0": "#2ECC71", "1": "#E74C3C"}, ax=axes[1])
axes[1].set_title("Glucose vs Outcome")
axes[1].set_xticklabels(["No Diabetes", "Diabetes"])
plt.tight_layout()
plt.savefig("models/eda_outcome.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: models/eda_outcome.png")

# 
# STEP 3 – DATA PREPROCESSING
# 
print("\n" + "─" * 60)
print("STEP 3 : Data Preprocessing")
print("─" * 60)

# Replace medically impossible zeroes with column median
for col in zero_as_null:
    median_val = df[col].replace(0, np.nan).median()
    df[col] = df[col].replace(0, median_val)
    print(f"  Replaced 0s in {col:30s} with median = {median_val:.1f}")

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train / test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\n  Train size : {X_train.shape[0]} samples")
print(f"  Test  size : {X_test.shape[0]} samples")

# Feature scaling (StandardScaler — zero mean, unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n  ✅ Features scaled with StandardScaler")

# 
# STEP 4 – TRAIN MULTIPLE ML MODELS
# 
print("\n" + "─" * 60)
print("STEP 4 : Training ML Models")
print("─" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred  = model.predict(X_test_scaled)
    acc     = accuracy_score(y_test, y_pred)
    cv_mean = cross_val_score(model, X_train_scaled, y_train,
                               cv=5, scoring="accuracy").mean()
    results[name] = {"model": model, "accuracy": acc,
                     "cv_accuracy": cv_mean, "y_pred": y_pred}
    print(f"\n  🔵 {name}")
    print(f"     Test Accuracy : {acc*100:.2f}%")
    print(f"     CV  Accuracy  : {cv_mean*100:.2f}%")

# 
# STEP 5 – COMPARE & SELECT BEST MODEL
# 
print("\n" + "─" * 60)
print("STEP 5 : Model Comparison")
print("─" * 60)

best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
print(f"\n   Best Model : {best_name}")
print(f"     Accuracy   : {results[best_name]['accuracy']*100:.2f}%")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(8, 5))
names  = list(results.keys())
accs   = [results[n]["accuracy"]*100 for n in names]
colors = ["#3498DB", "#E74C3C", "#2ECC71"]
bars = ax.bar(names, accs, color=colors, edgecolor="white", width=0.5)
ax.set_ylim(50, 100)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig("models/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: models/model_comparison.png")

# 
# STEP 6 – EVALUATION METRICS
# 
print("\n" + "─" * 60)
print("STEP 6 : Evaluation Metrics — Best Model")
print("─" * 60)

y_pred_best = results[best_name]["y_pred"]
print(f"\n  Classification Report:\n")
print(classification_report(y_test, y_pred_best,
      target_names=["No Diabetes", "Diabetes"]))

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: models/confusion_matrix.png")

# 
# STEP 7 – SAVE MODEL & SCALER
# 
print("\n" + "─" * 60)
print("STEP 7 : Saving Model with Pickle")
print("─" * 60)

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\n  ✅ Model saved  : models/best_model.pkl")
print(f"  ✅ Scaler saved : models/scaler.pkl")
print(f"   Saved model  : {best_name}")

# 
# STEP 8 – FEATURE IMPORTANCE (Random Forest)
# 
rf_model = results["Random Forest"]["model"]
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
feat_imp.plot.barh(ax=ax, color="#3498DB", edgecolor="white")
ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("models/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: models/feature_importance.png")

print("\n" + "=" * 60)
print("  ✅ TRAINING PIPELINE COMPLETE")
print("=" * 60)
print("  Next step → run:  streamlit run app/app.py")
print("=" * 60)
