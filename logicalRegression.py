from nGrams import generate_ngrams
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample data (extended for demonstration)
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The lazy dog sleeps all day.",
    "The quick fox runs fast."
]
labels = [1, 0, 1]  # Positive, Negative, Positive

# Generate bigrams and features
all_bigrams = set()
for text in texts:
    bigrams = generate_ngrams(text, n=2)
    all_bigrams.update(bigrams)

# Create feature matrix (presence of bigrams)
X = []
for text in texts:
    bigrams = generate_ngrams(text, n=2)
    row = [1 if bigram in bigrams else 0 for bigram in all_bigrams]
    X.append(row)
X = np.array(X)
y = np.array(labels)

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict sentiment for a new sentence
new_text = "The quick fox jumps high."
new_bigrams = generate_ngrams(new_text, n=2)
new_X = np.array([[1 if bigram in new_bigrams else 0 for bigram in all_bigrams]])
prediction = model.predict(new_X)
prob = model.predict_proba(new_X)[0]
print(f"\nNew Text: {new_text}")
print(f"Predicted Sentiment: {'Positive' if prediction[0] == 1 else 'Negative/Neutral'}")
print(f"Probability (Positive): {prob[1]:.2f}")

print("\nThis prediction was based on this sample data:")
for i, text in enumerate(texts):  # Corrected loop using enumerate
    print(f"{text} -> {labels[i]}")