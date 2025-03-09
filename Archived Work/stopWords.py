import nltk
from nltk.corpus import stopwords

# Download the NLTK stopwords if not already downloaded
nltk.download('stopwords')

# List of stop words
stop_words = set(stopwords.words('english'))

with open("stopWords.txt", "w") as file:
    for word in stop_words:
        file.write(word + "\n")
        
print("Stop words saved to stopwords.txt")