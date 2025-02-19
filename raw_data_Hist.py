# Program Purpose: This is a program that uses pands and matlab to vizulize the data in the research paper, "Comparting Comparative Sentences"
# Author: Daniel Chavez

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download the NLTK stopwords if not already downloaded
nltk.download('stopwords')

# List of stop words
stop_words = set(stopwords.words('english'))

def getTotalFreqWords(data, top_n=20):
        # Step 2: Get sentences from the column
    sentences = data['sentence'].astype(str)  # Assuming the column is named 'sentence'

    # Step 3: Tokenize words and remove stop words
    all_words = []
    for sentence in sentences:
        words = sentence.split()  # Split by whitespace
        for word in words:
            # Convert to lowercase and remove stop words
            word = word.lower()
            if word not in stop_words:
                all_words.append(word)

    # Step 4: Count unique word occurrences
    word_counts = Counter(all_words)
    print(f"Total unique words: {len(word_counts)}")
    
    # Get user input for number of top words to display
    inputN = input("Enter the amount of words you would like to see and how often they appear: ").strip()
    
    if inputN:
        try: 
            n = int(inputN)
        except ValueError:
            print("Invalid input, using default value of 20.")
            n = 20
    else:
        n = 20
    
    # Ensure n is within valid range
    while n <= 0 or n >= len(word_counts):
        if n <= 0:
            print("Please enter a number greater than 0.")
        elif n >= len(word_counts):
            print(f"Please enter a number less than {len(word_counts)}.")
        n = int(input("Enter a valid number: "))

    # Step 5: Create a DataFrame for plotting
    word_freq_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])
    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

    # Display the top n most frequent words
    print(word_freq_df.head(n).to_string(index=False))
    
    # Step 6: Plot the Histogram (Top N most common words)
    # plt.figure(figsize=(12, 6))
    # plt.bar(word_freq_df['Word'][:n], word_freq_df['Frequency'][:n], color='skyblue')
    # plt.xticks(rotation=45, ha='right')
    # plt.xlabel("Words")
    # plt.ylabel("Frequency")
    # plt.title(f"Top {n} Most Frequent Words in Sentences")

def getMostFreqWordsA_B(data):
    # Step 2: Get object_a and object_b from the csv file
    objects = data[['object_a', 'object_b']].astype(str)

    # Step 3: Tokenize words (split words in both columns)
    all_words = ' '.join(objects.values.flatten()).split()

    # Step 4: Count unique word occurrences
    word_counts = Counter(all_words)
    print(f"Total unique words: {len(word_counts)}")
    
    inputN = input("Enter the amount of words you would like to see and how often they appear: ").strip()
    n = 20
    
    if inputN:
        try: 
            n = int(inputN)
        except ValueError:
            print("Invalid input, using default value of 20.")
            
    while n <= 0 or n >= len(word_counts):
        if n <= 0:
            print("Please enter a number greater than 0.")
        elif n >= len(word_counts):
            print(f"Please enter a number less than {len(word_counts)}.")
        n = int(input("Enter a valid number: "))

        

    # Step 5: Create a DataFrame for plotting
    word_freq_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])
    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False) 

    print(word_freq_df.head(n).to_string(index=False))
    
def getSentimentOfA(data, topN=30):
    # Step 2: Extract relevant columns
    objects = data[['object_a', 'object_b', 'better_count', 'worse_count', 'none_count']].copy()

    # Step 3: Calculate sentiment percentage for each object_a
    objects['total_votes'] = objects['better_count'] + objects['worse_count'] + objects['none_count']

    # Avoid division by zero
    objects['better_percentage'] = (objects['better_count'] / objects['total_votes']).fillna(0) * 100
    objects['worse_percentage'] = (objects['worse_count'] / objects['total_votes']).fillna(0) * 100
    objects['neutral_percentage'] = (objects['none_count'] / objects['total_votes']).fillna(0) * 100

    # Step 4: Group by object_a to find average sentiment
    sentiment_summary = objects.groupby('object_a').agg({
        'better_percentage': 'mean',
        'worse_percentage': 'mean',
        'neutral_percentage': 'mean'
    }).reset_index()

   # Step 5: Find most positive and most negative objects
    top_positive = sentiment_summary.sort_values(by='better_percentage', ascending=False).head(topN)
    top_negative = sentiment_summary.sort_values(by='worse_percentage', ascending=False).head(topN)

    # Display the results
    print(f"Top {topN} Positive Objects:")
    print(top_positive[['object_a', 'better_percentage']].to_string(index=False))

    print(f"\nTop {topN} Negative Objects:")
    print(top_negative[['object_a', 'worse_percentage']].to_string(index=False))


    # Step 6: Display the sentiment summary table
    print("\nSentiment Summary (Top n):")
    print(sentiment_summary.head(topN).to_string(index=False))  # Top 20 objects

def main(): 
    # Step 1: Loads data from the csv file
    data = pd.read_csv("dataComparative.csv")
    
    getTotalFreqWords(data)
    getMostFreqWordsA_B(data)
    getSentimentOfA(data)
    
main()

print("Test One")