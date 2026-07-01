import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score


housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print("\nFirst 5 Rows:")
print(X.head())

print("\nDataset Information:")
print(X.info())

print("\nStatistical Summary:")
print(X.describe())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)


models = {
    "Linear Regression": linear_model,
    "Ridge Regression": ridge_model,
    "Decision Tree": tree_model
}

results = []

print("\n========== MODEL RESULTS ==========\n")

for name, model in models.items():

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append([name, rmse, r2])

    print(f"{name}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2 Score : {r2:.4f}")
    print("-" * 40)


results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "R2 Score"]
)

print("\nPerformance Comparison")
print(results_df)


results_df.to_csv("model_results.csv", index=False)


plt.figure(figsize=(8,5))

plt.bar(results_df["Model"], results_df["R2 Score"])

plt.title("Model Comparison (R2 Score)")
plt.xlabel("Models")
plt.ylabel("R2 Score")

plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)

plt.show()


best_model_name = results_df.loc[
    results_df["R2 Score"].idxmax(),
    "Model"
]

print("\nBest Model:", best_model_name)


if best_model_name == "Linear Regression":
    best_model = linear_model
elif best_model_name == "Ridge Regression":
    best_model = ridge_model
else:
    best_model = tree_model

predictions = best_model.predict(X_test)

plt.figure(figsize=(6,6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
)

plt.tight_layout()

plt.savefig("actual_vs_predicted.png", dpi=300)

plt.show()

print("\nProject Completed Successfully!")
print("Files Generated:")
print("1. model_results.csv")
print("2. model_comparison.png")
print("3. actual_vs_predicted.png")