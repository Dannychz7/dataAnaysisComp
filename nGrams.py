# Program Purpose: Generate n-grams from a given text input
# Author: Grok (for demonstration)

import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.linear_model import LinearRegression
import numpy as np

# Download NLTK tokenizer (uncomment if not already downloaded)
# nltk.download('punkt')

def generate_ngrams(text, n=2):
    """
    Generate n-grams from a text string.
    
    Args:
        text (str): Input text to process
        n (int): Size of the n-grams (e.g., 2 for bigrams, 3 for trigrams)
    
    Returns:
        list: List of n-grams (tuples of words)
    """
    # Check for valid n
    if n < 1:
        raise ValueError("n must be a positive integer")
    
    # Convert text to lowercase and tokenize into words
    text = text.lower()
    words = word_tokenize(text)
    
    # If text is shorter than n, return empty list
    if len(words) < n:
        return []
    
    # Generate n-grams using a sliding window
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])  # Use tuple for hashable n-grams
        ngrams.append(ngram)
    
    return ngrams

def ngram_frequencies(ngrams):
    """
    Calculate the frequency of each unique n-gram.
    
    Args:
        ngrams (list): List of n-grams
    
    Returns:
        Counter: Frequency count of each n-gram
    """
    return Counter(ngrams)

def print_ngrams(ngrams, freq_counter, n):
    """
    Print n-grams and their frequencies.
    
    Args:
        ngrams (list): List of n-grams
        freq_counter (Counter): Frequency count of n-grams
        n (int): Size of the n-grams
    """
    print(f"\n{n}-grams generated:")
    print("-" * 40)
    for ngram in ngrams:
        print(f"{' '.join(ngram)}")
    
    print(f"\nTop {n}-grams by frequency:")
    print("-" * 40)
    for ngram, freq in freq_counter.most_common(5):  # Top 5 most frequent
        print(f"{' '.join(ngram)}: {freq} times")

def extract_features(ngrams, text):
    words = word_tokenize(text.lower())
    features = []
    for ngram in ngrams:
        avg_length = np.mean([len(word) for word in ngram])
        avg_position = np.mean([words.index(word) + 1 for word in ngram if word in words]) / len(words)
        features.append([avg_length, avg_position])
    return np.array(features)

# Example usage
if __name__ == "__main__":
    # Sample text
    sample_text = "The quick brown fox jumps over the lazy dog and the quick fox runs."
    
    # Generate bigrams (n=2)
    bigrams = generate_ngrams(sample_text, n=2)
    bigram_freq = ngram_frequencies(bigrams)
    
    # Generate trigrams (n=3)
    trigrams = generate_ngrams(sample_text, n=3)
    trigram_freq = ngram_frequencies(trigrams)
    
    # Print results
    print_ngrams(bigrams, bigram_freq, 2)
    print_ngrams(trigrams, trigram_freq, 3)
    
    print("-" * 80)
    print("Here is a linear Regression that predicts the frequency of bigrams in a corpus based on the average word length of the bigram and its position in the sentence")

    text = "The quick brown fox jumps over the lazy dog."
    bigrams = generate_ngrams(text, n=2)
    bigram_freq = ngram_frequencies(bigrams)

    # Prepare data
    X = extract_features(bigrams, text)  # Features: avg word length, normalized position
    y = np.array([bigram_freq[ngram] for ngram in bigrams])  # Target: frequencies

    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict frequencies
    predictions = model.predict(X)
    print("\nPredicted vs Actual Frequencies:")
    for ngram, pred, actual in zip(bigrams, predictions, y):
        print(f"{' '.join(ngram)}: Predicted = {pred:.2f}, Actual = {actual}")
        

