# Program Purpose: This is a program that uses pands and matlab to vizulize the data in the research paper, "Comparting Comparative Sentences"
# Author: Daniel Chavez
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter, defaultdict

# You might need to download these resources first
# Uncomment these lines if you haven't downloaded them yet
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('vader_lexicon')

print("Analyzing data using super smart algorithms...")

def extract_data(csv_file_path):
    """
    Extract the 'sentence' and 'domain' columns from a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        tuple: (list of sentences, list of domains)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if required columns exist
        if 'sentence' not in df.columns:
            raise ValueError("The CSV file does not contain a 'sentence' column")
        if 'domain' not in df.columns:
            raise ValueError("The CSV file does not contain a 'domain' column")
        
        # Extract sentences and domains
        sentences = df['sentence'].tolist()
        domains = df['domain'].tolist()
        
        return sentences, domains
    
    except Exception as e:
        print(f"Error extracting data: {e}")
        return [], []

def clean_text(text_list):
    """
    Remove stop words and comparison-based words from a list of texts.
    
    Args:
        text_list (list): List of strings to clean
        
    Returns:
        list: List of cleaned strings
    """
    # Get standard English stop words
    stop_words = set(stopwords.words('english'))
    
    # Add comparison-based words to remove
    comparison_words = {
        'better', 'worse', 'more', 'less', 'greater', 'lesser',
        'higher', 'lower', 'like','bigger', 'smaller', 'faster', 'slower',
        'easier', 'harder', 'stronger', 'weaker', 'than', 'most',
        'least', 'best', 'worst', 'compared', 'compare', 'comparison',
        'versus', 'vs', 'against', 'prefer', 'preferable', 'preference',
        'superior', 'inferior', 'advantage', 'disadvantage', 'outperform',
        'outperforms', 'exceed', 'exceeds', 'surpass', 'surpasses', 'much',
        'decent', 'solid','cooler','good', 'use', 'even',
    }
    
    # Combine all words to remove
    words_to_remove = stop_words.union(comparison_words)
    
    cleaned_texts = []
    
    for text in text_list:
        if pd.isna(text):  # Handle NaN values (empty strings)
            cleaned_texts.append("")
            continue
            
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize the text
        words = nltk.word_tokenize(text)
        
        # Remove stop words and comparison words
        filtered_words = [word for word in words if word.lower() not in words_to_remove and word.isalnum()]
        
        # Join the words back into a string
        cleaned_text = ' '.join(filtered_words)
        
        cleaned_texts.append(cleaned_text)
    
    return cleaned_texts

def get_word_frequencies(text_list):
    """
    Calculate frequency of unique words across all texts in the provided list.
    
    Args:
        text_list (list): List of texts to analyze
        
    Returns:
        Counter: Counter object with word frequencies
    """
    # Initialize counter
    word_counter = Counter()
    
    # Process each text
    for text in text_list:
        if pd.isna(text) or text == "":
            continue
            
        # Tokenize the text
        words = nltk.word_tokenize(text.lower())
        
        # Only count alphanumeric words
        words = [word for word in words if word.isalnum()]
        
        # Update counter
        word_counter.update(words)
    
    return word_counter

def analyze_sentiment(sentence):
    """
    Analyze the sentiment of a sentence.
    
    Args:
        sentence (str): The input sentence to analyze
        
    Returns:
        tuple: (sentiment_text, is_positive)
            sentiment_text: A summary of the preference expressed in the sentence
            is_positive: Boolean indicating if sentiment is positive (1) or not (0)
    """
    # Initialize NLTK's sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Handle NaN or empty string
    if pd.isna(sentence) or sentence == "":
        return "No sentiment could be extracted.", 0
    
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(sentence)
    
    # Extract key elements from the sentence
    try:
        # Tokenize the sentence
        tokens = nltk.word_tokenize(sentence.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract nouns, which are typically the subjects/objects being compared
        nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
        
        # Check for comparative or preference terms
        comparative_terms = ['better', 'worse', 'easier', 'harder', 'faster', 'slower', 
                            'more', 'less', 'prefer', 'preferable', 'superior', 'inferior']
        
        found_comparatives = [word for word in tokens if word in comparative_terms]
        
        # Look for "than" to identify comparison structure
        than_index = -1
        if 'than' in tokens:
            than_index = tokens.index('than')
        
        # Extract entities being compared
        preferred_entity = ""
        less_preferred_entity = ""
        is_positive = 1 if sentiment_scores['compound'] > 0.1 else 0
        
        # If we found a "than" construction
        if than_index > 0:
            # Words before "than" are usually preferred in positive comparisons
            before_than = ' '.join(tokens[:than_index])
            after_than = ' '.join(tokens[than_index+1:])
            
            # Determine which part is preferred based on sentiment and comparison terms
            if any(term in before_than for term in ['better', 'easier', 'faster', 'more', 'prefer', 'preferable', 'superior']):
                preferred_entity = before_than
                less_preferred_entity = after_than
                is_positive = 1
            elif any(term in before_than for term in ['worse', 'harder', 'slower', 'less', 'inferior']):
                preferred_entity = after_than
                less_preferred_entity = before_than
                is_positive = 1
            else:
                # If no clear comparative terms, use overall sentiment
                if sentiment_scores['compound'] > 0:
                    preferred_entity = before_than
                    less_preferred_entity = after_than
                    is_positive = 1
                else:
                    preferred_entity = after_than
                    less_preferred_entity = before_than
                    is_positive = 0
                    
            # Construct summary
            if preferred_entity and less_preferred_entity:
                return f"{preferred_entity} is preferred over {less_preferred_entity}.", is_positive
        
        # If no clear comparison structure, extract key terms and sentiment
        key_terms = ' '.join(nouns[:3])  # Take up to first 3 nouns
        
        if sentiment_scores['compound'] > 0.1:
            return f"{key_terms} is viewed positively.", 1
        elif sentiment_scores['compound'] < -0.1:
            return f"{key_terms} is viewed negatively.", 0
        else:
            return f"No clear preference about {key_terms} was detected.", 0
            
    except Exception as e:
        # Fallback to simple sentiment analysis
        if sentiment_scores['compound'] > 0.1:
            return "The sentence expresses a positive sentiment or preference.", 1
        elif sentiment_scores['compound'] < -0.1:
            return "The sentence expresses a negative sentiment or preference.", 0
        else:
            return "No clear sentiment or preference could be detected.", 0

def track_domain_sentiment(sentences, domains, cleaned_sentences=None):
    """
    Analyze sentiments of sentences and track positive/negative counts by domain.
    
    Args:
        sentences (list): List of sentences to analyze
        domains (list): List of domains corresponding to each sentence
        cleaned_sentences (list, optional): List of cleaned sentences
        
    Returns:
        dict: Dictionary with domain sentiment statistics
    """
    # Initialize tracking dictionaries
    domain_counts = defaultdict(int)  # Total sentences per domain
    domain_positive = defaultdict(int)  # Positive sentiment count per domain
    domain_sentiment_texts = defaultdict(list)  # Store sentiment texts by domain
    
    # For word analysis by sentiment
    domain_positive_words = defaultdict(Counter)
    domain_negative_words = defaultdict(Counter)
    
    # Process each sentence with its domain
    for i, (sentence, domain) in enumerate(zip(sentences, domains)):
        if pd.isna(domain) or domain == "":
            domain = "unknown"
            
        # Analyze sentiment
        sentiment_text, is_positive = analyze_sentiment(sentence)
        
        # Update counters
        domain_counts[domain] += 1
        domain_positive[domain] += is_positive
        
        # Store sentiment text
        domain_sentiment_texts[domain].append((sentence, sentiment_text, is_positive))
        
        # Track words by sentiment
        if cleaned_sentences and i < len(cleaned_sentences):
            clean_text = cleaned_sentences[i]
            
            if clean_text:
                words = nltk.word_tokenize(clean_text.lower())
                words = [word for word in words if word.isalnum()]
                
                if is_positive:
                    domain_positive_words[domain].update(words)
                else:
                    domain_negative_words[domain].update(words)
    
    # Calculate statistics
    results = {}
    for domain in domain_counts:
        total = domain_counts[domain]
        positive = domain_positive[domain]
        negative = total - positive
        positive_percent = (positive / total * 100) if total > 0 else 0
        
        results[domain] = {
            'total': total,
            'positive': positive,
            'negative': negative,
            'positive_percent': positive_percent,
            'sentiment_examples': domain_sentiment_texts[domain][:3],  # Store first 3 examples
            'positive_words': domain_positive_words[domain],
            'negative_words': domain_negative_words[domain]
        }
    
    return results

def analyze_sentiment_keywords(domain_results, top_n=15):
    """
    Analyze and compare common words in positive vs negative sentiments for each domain.
    
    Args:
        domain_results (dict): Results from track_domain_sentiment function
        top_n (int): Number of top words to display for each category
        
    Returns:
        dict: Dictionary with keyword analysis results by domain
    """
    keyword_analysis = {}
    
    for domain, results in domain_results.items():
        pos_words = results['positive_words']
        neg_words = results['negative_words']
        
        # Get top words for each sentiment
        top_pos = pos_words.most_common(top_n)
        top_neg = neg_words.most_common(top_n)
        
        # Find distinctive words (higher in one sentiment than the other)
        distinctive_pos = []
        distinctive_neg = []
        
        # Create a set of all words to analyze
        all_words = set()
        for word, _ in top_pos:
            all_words.add(word)
        for word, _ in top_neg:
            all_words.add(word)
            
        # Find ratio for each word
        for word in all_words:
            pos_count = pos_words.get(word, 0)
            neg_count = neg_words.get(word, 0)
            
            # Calculate normalized counts (per 100 sentences)
            if results['positive'] > 0:
                pos_norm = (pos_count / results['positive']) * 100
            else:
                pos_norm = 0
                
            if results['negative'] > 0:
                neg_norm = (neg_count / results['negative']) * 100
            else:
                neg_norm = 0
            
            # Calculate ratio and difference
            if pos_norm > 0 and neg_norm > 0:
                if pos_norm > neg_norm:
                    ratio = pos_norm / neg_norm
                    diff = pos_norm - neg_norm
                    if ratio >= 1.5 and diff >= 2:  # At least 50% more common and 2% difference
                        distinctive_pos.append((word, pos_count, neg_count, ratio, diff))
                else:
                    ratio = neg_norm / pos_norm
                    diff = neg_norm - pos_norm
                    if ratio >= 1.5 and diff >= 2:  # At least 50% more common and 2% difference
                        distinctive_neg.append((word, pos_count, neg_count, ratio, diff))
            elif pos_norm > 0:
                distinctive_pos.append((word, pos_count, neg_count, float('inf'), pos_norm))
            elif neg_norm > 0:
                distinctive_neg.append((word, pos_count, neg_count, float('inf'), neg_norm))
        
        # Sort distinctive words by difference
        distinctive_pos = sorted(distinctive_pos, key=lambda x: x[4], reverse=True)[:top_n]
        distinctive_neg = sorted(distinctive_neg, key=lambda x: x[4], reverse=True)[:top_n]
        
        keyword_analysis[domain] = {
            'top_positive': top_pos,
            'top_negative': top_neg,
            'distinctive_positive': distinctive_pos,
            'distinctive_negative': distinctive_neg
        }
    
    return keyword_analysis

def print_first_15_sentences(sentences, domains, cleaned_sentences=None, sentiments=None):
    """
    Print the first 15 sentences from the dataset with domain and sentiment info.
    
    Args:
        sentences (list): List of original sentences
        domains (list): List of domains for each sentence
        cleaned_sentences (list, optional): List of cleaned sentences
        sentiments (list, optional): List of sentiment analyses (text, is_positive)
    """
    # Determine how many sentences to print (up to 15)
    num_to_print = min(15, len(sentences))
    
    print(f"Printing first {num_to_print} sentences:")
    print("-" * 80)
    
    for i in range(num_to_print):
        domain = domains[i] if i < len(domains) else "unknown"
        print(f"[{i+1}] Domain: {domain}")
        print(f"    Original: {sentences[i]}")
        
        if cleaned_sentences and i < len(cleaned_sentences):
            print(f"    Cleaned:  {cleaned_sentences[i]}")
            
        if sentiments and i < len(sentiments):
            sentiment_text, is_positive = sentiments[i]
            sentiment_label = "Positive" if is_positive else "Negative/Neutral"
            print(f"    Sentiment: {sentiment_text}")
            print(f"    Label: {sentiment_label}")
            
        print("-" * 80)

def print_domain_sentiment_summary(domain_results):
    """
    Print a summary of sentiment analysis by domain.
    
    Args:
        domain_results (dict): Results from track_domain_sentiment function
    """
    print("\nDOMAIN SENTIMENT SUMMARY")
    print("=" * 80)
    
    # Sort domains by positive percentage (descending)
    sorted_domains = sorted(domain_results.keys(), 
                           key=lambda k: domain_results[k]['positive_percent'], 
                           reverse=True)
    
    # Print header
    print(f"{'DOMAIN':<20} {'TOTAL':<10} {'POSITIVE':<10} {'NEGATIVE':<10} {'POS %':<10}")
    print("-" * 80)
    
    # Print each domain's stats
    for domain in sorted_domains:
        stats = domain_results[domain]
        print(f"{domain:<20} {stats['total']:<10} {stats['positive']:<10} {stats['negative']:<10} {stats['positive_percent']:.1f}%")
    
    print("\nEXAMPLES BY DOMAIN")
    print("=" * 80)
    
    # Print examples for each domain
    for domain in sorted_domains:
        print(f"\nDomain: {domain}")
        print("-" * 80)
        
        for i, (sentence, sentiment, is_positive) in enumerate(domain_results[domain]['sentiment_examples']):
            label = "POSITIVE" if is_positive else "NEGATIVE/NEUTRAL"
            print(f"Example {i+1} [{label}]:")
            print(f"Sentence: {sentence}")
            print(f"Sentiment: {sentiment}")
            print("-" * 40)

def print_sentiment_keyword_analysis(keyword_analysis, top_n=10):
    """
    Print the keyword analysis results showing which words are more common
    in positive vs negative sentiments for each domain.
    
    Args:
        keyword_analysis (dict): Results from analyze_sentiment_keywords function
        top_n (int): Number of top words to display
    """
    print("\nSENTIMENT KEYWORD ANALYSIS BY DOMAIN")
    print("=" * 80)
    
    for domain, analysis in keyword_analysis.items():
        print(f"\nDOMAIN: {domain}")
        print("=" * 80)
        
        # Print most frequent words in positive sentences
        print("\nMOST FREQUENT WORDS IN POSITIVE SENTENCES:")
        print(f"{'WORD':<15} {'COUNT':<10}")
        print("-" * 25)
        for word, count in analysis['top_positive'][:top_n]:
            print(f"{word:<15} {count:<10}")
        
        # Print most frequent words in negative sentences
        print("\nMOST FREQUENT WORDS IN NEGATIVE SENTENCES:")
        print(f"{'WORD':<15} {'COUNT':<10}")
        print("-" * 25)
        for word, count in analysis['top_negative'][:top_n]:
            print(f"{word:<15} {count:<10}")
        
        # Print distinctive positive words (more common in positive than negative)
        print("\nDISTINCTIVE POSITIVE WORDS (significantly more common in positive sentences):")
        print(f"{'WORD':<15} {'POS COUNT':<10} {'NEG COUNT':<10} {'RATIO':<10} {'DIFF':<10}")
        print("-" * 60)
        for word, pos_count, neg_count, ratio, diff in analysis['distinctive_positive'][:top_n]:
            ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "∞"
            print(f"{word:<15} {pos_count:<10} {neg_count:<10} {ratio_str:<10} {diff:.1f}")
        
        # Print distinctive negative words (more common in negative than positive)
        print("\nDISTINCTIVE NEGATIVE WORDS (significantly more common in negative sentences):")
        print(f"{'WORD':<15} {'POS COUNT':<10} {'NEG COUNT':<10} {'RATIO':<10} {'DIFF':<10}")
        print("-" * 60)
        for word, pos_count, neg_count, ratio, diff in analysis['distinctive_negative'][:top_n]:
            ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "∞"
            print(f"{word:<15} {pos_count:<10} {neg_count:<10} {ratio_str:<10} {diff:.1f}")
            
        print("\n" + "=" * 80)

# Example usage:
sentences, domains = extract_data("dataComparative.csv")
cleaned_sentences = clean_text(sentences)

# Analyze sentiments
sentiments = [analyze_sentiment(sentence) for sentence in sentences]

# Track domain sentiment
domain_results = track_domain_sentiment(sentences, domains, cleaned_sentences)

# Analyze sentiment keywords
keyword_analysis = analyze_sentiment_keywords(domain_results)

# Print first 15 sentences with domain and sentiment
print_first_15_sentences(sentences, domains, cleaned_sentences, sentiments)

# Print domain sentiment summary
print_domain_sentiment_summary(domain_results)

# Print keyword analysis by sentiment
print_sentiment_keyword_analysis(keyword_analysis)

