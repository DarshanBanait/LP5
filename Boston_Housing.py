# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("boston_housing.csv")
print("Dataset shape:", df.shape)
df.head()

# %%
# 3. Split features and target
X = df.drop("MEDV", axis=1)  # MEDV is the target column
y = df["MEDV"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),  # Linear activation for regression
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# %%
history = model.fit(
    X_train_scaled, y_train, epochs=100, validation_split=0.1, verbose=0
)

# %%
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# %%
# 9. Predict on test data
y_pred = model.predict(X_test_scaled).flatten()

# 10. Show 10 actual vs predicted values
results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
results_df = results_df.reset_index(drop=True)
results_df.head(10)

# %%
# 11. Performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
