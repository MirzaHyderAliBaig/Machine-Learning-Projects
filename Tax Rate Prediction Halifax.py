# ----------------------------
# 1. Import required libraries
# ----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------
# 2. Load dataset
# ----------------------------
df = pd.read_csv("Tax_Rates_HRM.csv")

# Preview first 5 rows and columns
print("First 5 rows:\n", df.head())
print("\nColumns in dataset:\n", df.columns)

# ----------------------------
# 3. Check for missing values
# ----------------------------
print("\nMissing values per column:\n", df.isnull().sum())
df = df.dropna()

# ----------------------------
# 4. Remove extreme outliers in 'Rate'
# ----------------------------
# Remove top 1% of rates to avoid skewing
upper_limit = df["Rate"].quantile(0.99)
df = df[df["Rate"] <= upper_limit]

# ----------------------------
# 5. Encode categorical columns
# ----------------------------
categorical_cols = ["Calculation_Type", "Rate_Type", "Rate_Code"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ----------------------------
# 6. Feature selection
# ----------------------------
# Numeric features
numeric_features = ["Bill_Year", "Minimum_Rate", "Maximum_Rate"]

# One-hot encoded categorical columns
encoded_features = [col for col in df.columns if
                    col.startswith("Calculation_Type_") or
                    col.startswith("Rate_Type_") or
                    col.startswith("Rate_Code_")]

# Combine all features
features = numeric_features + encoded_features
X = df[features]

# Target variable: log-transform to handle skew
y = np.log1p(df["Rate"])  # log(1 + Rate) avoids log(0)

print("\nFeatures used:\n", X.head())

# ----------------------------
# 7. Split dataset into training and testing sets
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 8. Feature scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 9. Train model
# ----------------------------
# You can switch to Ridge regression by uncommenting next line
# model = Ridge(alpha=1.0)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ----------------------------
# 10. Make predictions
# ----------------------------
y_pred = model.predict(X_test_scaled)

# Convert back from log-scale to original Rate
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

# ----------------------------
# 11. Evaluate model
# ----------------------------
mse = mean_squared_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("\nMean Squared Error:", mse)
print("R² Score:", r2)

# ----------------------------
# 12. Feature coefficients
# ----------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nFeature Coefficients (sorted by importance):\n", coefficients)

# ----------------------------
# 13. Plot Actual vs Predicted Tax Rate
# ----------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.6)
plt.xlabel("Actual Tax Rate")
plt.ylabel("Predicted Tax Rate")
plt.title("Actual vs Predicted Tax Rate (Log-Transformed Model)")
plt.grid(True)
plt.show()


