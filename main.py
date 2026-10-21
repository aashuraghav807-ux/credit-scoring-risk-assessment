import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Column names from german.doc
columns = [
    "checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "credit_risk"
]

# Load the German Credit dataset
df = pd.read_csv(
    "data/german.data",
    sep=" ",
    header=None,
    names=columns
)

print("Dataset loaded successfully!")
print(df.head())
print("Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nCredit Risk Distribution:")
print(df["credit_risk"].value_counts())

print("\nBasic Statistics:")
print(df.describe())

# Credit Risk Distribution
plt.figure(figsize=(6, 4))

df["credit_risk"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Credit Risk Distribution")
plt.xlabel("Credit Risk")
plt.ylabel("Number of Applicants")
plt.xticks([0, 1], ["Good (1)", "Bad (2)"], rotation=0)

plt.tight_layout()
plt.savefig("graphs/credit_risk_distribution.png")
plt.show()

# Credit Amount Distribution
plt.figure(figsize=(7, 4))

plt.hist(df["credit_amount"], bins=20)

plt.title("Credit Amount Distribution")
plt.xlabel("Credit Amount")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.savefig("graphs/credit_amount_distribution.png")
plt.show()

# Loan Duration Distribution
plt.figure(figsize=(7, 4))

plt.hist(df["duration_months"], bins=15)

plt.title("Loan Duration Distribution")
plt.xlabel("Duration (Months)")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.savefig("graphs/loan_duration_distribution.png")
plt.show()

# Credit Amount vs Credit Risk
plt.figure(figsize=(7, 4))

df.boxplot(
    column="credit_amount",
    by="credit_risk"
)

plt.title("Credit Amount by Credit Risk")
plt.suptitle("")
plt.xlabel("Credit Risk")
plt.ylabel("Credit Amount")
plt.xticks([1, 2], ["Good (1)", "Bad (2)"])

plt.tight_layout()
plt.savefig("graphs/credit_amount_by_risk.png")
plt.show()

# Check unique values of categorical columns
categorical_columns = df.select_dtypes(include="str").columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())

# Separate features and target
X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include="str").columns
numerical_columns = X.select_dtypes(exclude="str").columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Fit the preprocessor on training data
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("\nEncoded training data shape:", X_train_encoded.shape)
print("Encoded testing data shape:", X_test_encoded.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Scale the encoded features
scaler = StandardScaler(with_mean=False)

X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Accuracy:", accuracy)

from sklearn.metrics import classification_report, confusion_matrix

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Good", "Bad"]
    )
)

from sklearn.tree import DecisionTreeClassifier

# Decision Tree Model
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

dt_model.fit(X_train_encoded, y_train)

dt_pred = dt_model.predict(X_test_encoded)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", dt_accuracy)

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

print("\nDecision Tree Classification Report:")
print(classification_report(
    y_test,
    dt_pred,
    target_names=["Good", "Bad"]
))

from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train_encoded, y_train)

rf_pred = rf_model.predict(X_test_encoded)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test,
    rf_pred,
    target_names=["Good", "Bad"]
))

# Cost-Sensitive Logistic Regression
cost_model = LogisticRegression(
    max_iter=1000,
    class_weight={1: 1, 2: 5},
    random_state=42
)

cost_model.fit(X_train_scaled, y_train)

cost_pred = cost_model.predict(X_test_scaled)

cost_accuracy = accuracy_score(y_test, cost_pred)

print("\nCost-Sensitive Logistic Regression Accuracy:", cost_accuracy)

print("\nCost-Sensitive Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, cost_pred))

print("\nCost-Sensitive Logistic Regression Classification Report:")
print(classification_report(
    y_test,
    cost_pred,
    target_names=["Good", "Bad"]
))

def calculate_credit_cost(cm):
    good_to_bad = cm[0, 1]
    bad_to_good = cm[1, 0]

    total_cost = (good_to_bad * 1) + (bad_to_good * 5)

    return total_cost

# Feature Interpretation - Cost-Sensitive Logistic Regression

feature_names = preprocessor.get_feature_names_out()
coefficients = cost_model.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute Coefficient": abs(coefficients)
}).sort_values(
    "Absolute Coefficient",
    ascending=False
)

print("\n===== TOP IMPORTANT FEATURES =====")
print(feature_importance.head(15))

# Feature Importance Graph

top_features = feature_importance.head(10).sort_values(
    "Coefficient"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

plt.title("Top 10 Features - Cost-Sensitive Logistic Regression")
plt.xlabel("Coefficient")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    "graphs/cost_sensitive_lr_feature_importance.png",
    dpi=300
)

plt.show()

# Confusion Matrices
lr_cm = confusion_matrix(y_test, y_pred)
dt_cm = confusion_matrix(y_test, dt_pred)
rf_cm = confusion_matrix(y_test, rf_pred)
cost_cm = confusion_matrix(y_test, cost_pred)


# Cost Calculation
print("\n===== COST ANALYSIS =====")

print("Logistic Regression Cost:",
      calculate_credit_cost(lr_cm))

print("Decision Tree Cost:",
      calculate_credit_cost(dt_cm))

print("Random Forest Cost:",
      calculate_credit_cost(rf_cm))

print("Cost-Sensitive Logistic Regression Cost:",
      calculate_credit_cost(cost_cm))

# Model Comparison

lr_report = classification_report(
    y_test, y_pred, output_dict=True
)

dt_report = classification_report(
    y_test, dt_pred, output_dict=True
)

rf_report = classification_report(
    y_test, rf_pred, output_dict=True
)

cost_report = classification_report(
    y_test, cost_pred, output_dict=True
)


comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Cost-Sensitive Logistic Regression"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, cost_pred)
    ],

    "Bad Recall": [
        lr_report["2"]["recall"],
        dt_report["2"]["recall"],
        rf_report["2"]["recall"],
        cost_report["2"]["recall"]
    ],

    "Total Cost": [
        calculate_credit_cost(lr_cm),
        calculate_credit_cost(dt_cm),
        calculate_credit_cost(rf_cm),
        calculate_credit_cost(cost_cm)
    ]
})


print("\n===== MODEL COMPARISON =====")
print(comparison)

import matplotlib.pyplot as plt

# Model Accuracy Comparison
models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Cost-Sensitive LR"
]

accuracies = [
    0.775,
    0.645,
    0.765,
    0.635
]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

# Display accuracy values on bars
for i, value in enumerate(accuracies):
    plt.text(i, value + 0.02, f"{value:.1%}", ha="center")

plt.tight_layout()

# Save graph
plt.savefig("graphs/model_accuracy_comparison.png", dpi=300)
plt.show()

# Bad Recall Comparison

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Cost-Sensitive LR"
]

bad_recalls = [
    lr_report["2"]["recall"],
    dt_report["2"]["recall"],
    rf_report["2"]["recall"],
    cost_report["2"]["recall"]
]

plt.figure(figsize=(10, 6))

plt.bar(models, bad_recalls)

plt.title("Bad Credit Recall Comparison")
plt.xlabel("Models")
plt.ylabel("Bad Credit Recall")

plt.ylim(0, 1)

for i, value in enumerate(bad_recalls):
    plt.text(i, value + 0.02, f"{value:.1%}", ha="center")

plt.tight_layout()

plt.savefig("graphs/bad_recall_comparison.png", dpi=300)

plt.show()

# Total Misclassification Cost Comparison

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Cost-Sensitive LR"
]

total_costs = [
    calculate_credit_cost(lr_cm),
    calculate_credit_cost(dt_cm),
    calculate_credit_cost(rf_cm),
    calculate_credit_cost(cost_cm)
]

plt.figure(figsize=(10, 6))

plt.bar(models, total_costs)

plt.title("Total Misclassification Cost Comparison")
plt.xlabel("Models")
plt.ylabel("Total Misclassification Cost")

for i, value in enumerate(total_costs):
    plt.text(i, value + 3, str(value), ha="center")

plt.tight_layout()

plt.savefig("graphs/total_cost_comparison.png", dpi=300)

plt.show()

# Confusion Matrix Visualizations

from sklearn.metrics import ConfusionMatrixDisplay

# Logistic Regression
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Good", "Bad"]
)
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix_logistic_regression.png", dpi=300)
plt.show()


# Decision Tree
ConfusionMatrixDisplay.from_predictions(
    y_test, dt_pred,
    display_labels=["Good", "Bad"]
)
plt.title("Confusion Matrix - Decision Tree")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix_decision_tree.png", dpi=300)
plt.show()


# Random Forest
ConfusionMatrixDisplay.from_predictions(
    y_test, rf_pred,
    display_labels=["Good", "Bad"]
)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix_random_forest.png", dpi=300)
plt.show()


# Cost-Sensitive Logistic Regression
ConfusionMatrixDisplay.from_predictions(
    y_test, cost_pred,
    display_labels=["Good", "Bad"]
)
plt.title("Confusion Matrix - Cost-Sensitive Logistic Regression")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix_cost_sensitive_lr.png", dpi=300)
plt.show()

# Age Distribution

plt.figure(figsize=(10, 6))

plt.hist(df["age"], bins=10, edgecolor="black")

plt.title("Age Distribution of Credit Applicants")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("graphs/age_distribution.png", dpi=300)

plt.show()

# Credit Amount vs Loan Duration

plt.figure(figsize=(10, 6))

plt.scatter(
    df["duration_months"],
    df["credit_amount"],
    alpha=0.6
)

plt.title("Credit Amount vs Loan Duration")
plt.xlabel("Loan Duration (Months)")
plt.ylabel("Credit Amount")

plt.tight_layout()

plt.savefig("graphs/credit_amount_vs_loan_duration.png", dpi=300)

plt.show()

# Save Final Model and Preprocessor

import joblib

joblib.dump(
    cost_model,
    "cost_sensitive_logistic_regression.pkl"
)

joblib.dump(
    preprocessor,
    "preprocessor.pkl"
)

print("\n===== MODEL SAVED SUCCESSFULLY =====")
print("Final model: cost_sensitive_logistic_regression.pkl")
print("Preprocessor: preprocessor.pkl")

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Scaler: scaler.pkl")
