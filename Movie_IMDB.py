# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 2. Load dataset
df = pd.read_csv("IMDB_Dataset.csv")
print("Shape:", df.shape)
df.head()

# %%
# 3. Encode target variable (positive -> 1, negative -> 0)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

# %%
# 5. Tokenize and pad text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 200
X_train_pad = pad_sequences(
    X_train_seq, maxlen=max_len, padding="post", truncating="post"
)
X_test_pad = pad_sequences(
    X_test_seq, maxlen=max_len, padding="post", truncating="post"
)

# %%
# 6. Build DNN model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 7. Train the model
history = model.fit(
    X_train_pad, y_train, epochs=10, validation_data=(X_test_pad, y_test), verbose=2
)

# %%
# 8. Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.show()

# %%
# 9. Predictions and thresholding
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# %%
# 10. Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# 11. Show 10 actual reviews and predicted sentiment
sample_reviews = X_test.reset_index(drop=True)[:10]
sample_preds = y_pred[:10]
sample_true = y_test.reset_index(drop=True)[:10]

for i in range(10):
    print(
        f"\nReview {i+1}:\n{sample_reviews[i][:300]}..."
    )  # print only first 300 chars
    print(f"Actual Sentiment   : {'Positive' if sample_true[i] == 1 else 'Negative'}")
    print(f"Predicted Sentiment: {'Positive' if sample_preds[i] == 1 else 'Negative'}")
