import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import joblib


# Load dataset
df = pd.read_csv("telco-cust-churn.csv")

# Basic information
print("First 5 Rows:")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Info:")
df.info()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

print("\nShape after removing missing values:")
print(df.shape)

# Remove customerID
df = df.drop("customerID", axis=1)

# Convert target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\nChurn Value Counts:")
print(df["Churn"].value_counts())

# Convert categorical columns into numerical values
df = pd.get_dummies(df, drop_first=True)

print("\nShape after Encoding:")
print(df.shape)

print("\nFirst 5 Rows after Encoding:")
print(df.head())

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\n==============================")
print("Random Forest Model")
print("==============================")

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

log_acc = accuracy_score(y_test, y_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nModel Comparison")
print("----------------")
print("Logistic Regression Accuracy:", log_acc)
print("Random Forest Accuracy:", rf_acc)

models = ["Logistic Regression", "Random Forest"]
accuracy = [log_acc, rf_acc]

plt.figure(figsize=(6,5))
plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# Confusion Matrix Visualization
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)

plt.title("Random Forest Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(rf, X_test, y_test)

plt.title("ROC Curve - Random Forest")
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# Save Random Forest Model
joblib.dump(rf, "customer_churn_model.pkl")

print("Model saved successfully!")