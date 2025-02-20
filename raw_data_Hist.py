# Program Purpose: This is a program that uses pands and matlab to vizulize the data in the research paper, "Comparting Comparative Sentences"
# Author: Daniel Chavez

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# Read stop words from the file
with open("stopwords.txt", "r") as file:
    stop_words = set(file.read().splitlines())

# print("Loaded stop words:", stop_words) DEBUGG LINE TO PRINT ALL STOP WORDS

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
    print(f"Total unique words: {len(word_counts)} in file from sentences column: dataComparative.csv")
    
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
    
def getSentimentOfA(data):
    # Step 2: Extract relevant columns
    objects = data[['object_a', 'object_b', 'better_count', 'worse_count', 'none_count']].copy()

    # Step 3: Calculate sentiment percentage for each object_a
    objects['total_votes'] = objects['better_count'] + objects['worse_count'] + objects['none_count']
    
    total_elements = len(objects)
    print(f"Total elements: {total_elements}")
    n = getValidNumber(total_elements)

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
    top_positive = sentiment_summary.sort_values(by='better_percentage', ascending=False).head(n)
    top_negative = sentiment_summary.sort_values(by='worse_percentage', ascending=False).head(n)

    # Display the results
    print(f"Top {n} Positive Objects:")
    print(top_positive[['object_a', 'better_percentage']].to_string(index=False))

    print(f"\nTop {n} Negative Objects:")
    print(top_negative[['object_a', 'worse_percentage']].to_string(index=False))


    # Step 6: Display the sentiment summary table
    print("\nSentiment Summary (Top n):")
    print(sentiment_summary.head(n).to_string(index=False))  # Top 20 objects
    
def getSentimentOfB(data):
    # Step 2: Extract relevant columns
    objects = data[['object_a', 'object_b', 'better_count', 'worse_count', 'none_count']].copy()

    # Step 3: Calculate sentiment percentage for each object_b
    objects['total_votes'] = objects['better_count'] + objects['worse_count'] + objects['none_count']
    
    total_elements = len(objects)
    print(f"Total elements: {total_elements}")
    n = getValidNumber(total_elements)

    # Avoid division by zero
    objects['better_percentage'] = (objects['better_count'] / objects['total_votes']).fillna(0) * 100
    objects['worse_percentage'] = (objects['worse_count'] / objects['total_votes']).fillna(0) * 100
    objects['neutral_percentage'] = (objects['none_count'] / objects['total_votes']).fillna(0) * 100

    # Step 4: Group by object_b to find average sentiment
    sentiment_summary = objects.groupby('object_b').agg({
        'better_percentage': 'mean',
        'worse_percentage': 'mean',
        'neutral_percentage': 'mean'
    }).reset_index()

    # Step 5: Find most positive and most negative objects
    top_positive = sentiment_summary.sort_values(by='better_percentage', ascending=False).head(n)
    top_negative = sentiment_summary.sort_values(by='worse_percentage', ascending=False).head(n)

    # Display the results
    print(f"Top {n} Positive Objects:")
    print(top_positive[['object_b', 'better_percentage']].to_string(index=False))

    print(f"\nTop {n} Negative Objects:")
    print(top_negative[['object_b', 'worse_percentage']].to_string(index=False))

    # Step 6: Display the sentiment summary table
    print("\nSentiment Summary (Top n):")
    print(sentiment_summary.head(n).to_string(index=False))  # Top n objects


def getUserInput(): 
    numOfFuncs = 4
    print("Welcome to the data anaysis of Comparting Comparative Sentences")
    print("To begin, here is a list of commands we can do to vizulize the data: ")
    print("1. Get the frequency of all unique words from the SENTENCES column")
    print("2. Get the frequency of all unique words from the OBJECT_A and OBJECT_B column")
    print("3. Get the sentiment of A in comparasion to B (Top n Positive and Negative)")
    print("4. Get the sentiment of B in comparasion to A (Top n Positive and Negative)")
    print("Q. To exit the program")
    # <------------- Below These functions are not yet done -------------> #
    # print("5. Create a vizulization of all the data") #Using streamLit, we will create a webpage of some kind to vizuluze the data
    while True:
        try:
                input_value = input("Please enter a number you would like to do: ").strip().lower()    
                # Default to 1
                if not input_value:
                    print("No input entered, exiting...")
                    return -1
                elif input_value == "q" or input_value == "quit":
                    print("Exiting...")
                    return -1

                n = int(input_value)
                # Ensure n is greater than 0 and less than or equal to numOfFuncs
                if 0 < n <= numOfFuncs:
                    return n
                else:
                    print(f"Invalid number! Please enter a number greater than 0 and less than or equal to {numOfFuncs}.")
        except ValueError:
            print("Invalid input! Please enter a valid integer.")

def getValidNumber(wordCount):
    while True:
        try:
            # Get user input and strip whitespace
            input_value = input("Enter the amount of words you would like to see and how often they appear (default 20): ").strip()
            
            # Default to 20 if input is empty
            if not input_value:
                print("No input entered, using default value of 20.")
                return 20

            # Convert input to an integer
            n = int(input_value)

            # Ensure n is greater than 0 and less than word_counts
            if 0 < n < wordCount:
                return n
            else:
                print(f"Invalid number! Please enter a number greater than 0 and less than {wordCount}.")

        except ValueError:
            print("Invalid input! Please enter a valid integer.")

def main(): 
    # Step 1: Loads data from the csv file
    data = pd.read_csv("dataComparative.csv")
    userChoice  = getUserInput()
    
    while userChoice > 0:
        if userChoice == 1:
                getTotalFreqWords(data)
        elif userChoice == 2:
            getMostFreqWordsA_B(data)
        elif userChoice == 3:
            getSentimentOfA(data)
        elif userChoice == 4:
            getSentimentOfB(data)
        
        userChoice  = getUserInput()
        
main()
