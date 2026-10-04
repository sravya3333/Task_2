import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("data/customers.csv")

print("Dataset shape:", df.shape)
print(df.head())

# -------------------------
# 2. Data preprocessing
# -------------------------
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Encode categorical variable
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# Define target and features
X = df.drop(columns=["status"])
y = df["status"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 3. Model definitions
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []

# -------------------------
# 4. Train and evaluate
# -------------------------
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ROC AUC for binary classification
    try:
        prob = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, prob)
    except:
        roc_auc = None

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("ROC AUC:", roc_auc)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"outputs/{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.show()

    # Plot ROC curve if available
    if roc_auc is not None:
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"{name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"outputs/{name.lower().replace(' ', '_')}_roc_curve.png")
        plt.show()

# -------------------------
# 5. Compare performance results
# -------------------------
results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df.sort_values("Accuracy", ascending=False))

# Plot model comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/model_accuracy_comparison.png")
plt.show()

# -------------------------
# 6. Best model summary
# -------------------------
best_model = results_df.sort_values("Accuracy", ascending=False).iloc[0]
print("\nBest performing model:")
print(best_model)

# Optional: cross-validation
print("\nCross-validation scores (Accuracy):")
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")