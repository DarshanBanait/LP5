# %%
# 📦 Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# %%
# 📂 Step 2: Load Dataset
# Make sure the file is in the current directory or specify the full path
column_names = [
    "letter",
    "x-box",
    "y-box",
    "width",
    "height",
    "onpix",
    "x-bar",
    "y-bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x-ege",
    "xegvy",
    "y-ege",
    "yegvx",
]

df = pd.read_csv("OCR/letter-recognition.data", header=None, names=column_names)
df.head()

# %%
# 🏷️ Step 3: Preprocess Data
X = df.drop("letter", axis=1)
y = df["letter"]

# Encode labels (A-Z => 0-25)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot encode targets
y_onehot = to_categorical(y_encoded)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("X shape:", X_scaled.shape)
print("y shape (one-hot):", y_onehot.shape)

# %%
# 🔀 Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.2, random_state=42
)

# %%
# 🧠 Step 5: Define Deep Neural Network
model = Sequential(
    [
        Dense(128, input_shape=(16,), activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(26, activation="softmax"),  # 26 output classes (A-Z)
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# %%
# 🏋️ Step 6: Train the Model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32)

# %%
# 📊 Step 7: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# %%
# 📈 Step 8: Plot Training Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.show()

# %%
# 🔍 Step 9: Make Predictions (optional)
predictions = model.predict(X_test)
pred_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Decode class labels
pred_labels = le.inverse_transform(pred_classes)
true_labels = le.inverse_transform(true_classes)

# Show some predictions
for i in range(5):
    print(f"True: {true_labels[i]} | Predicted: {pred_labels[i]}")
