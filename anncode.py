# Import libraries

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Download nltk punk
nltk.download("punkt_tab")
nltk.download("wordnet")
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents.json file
intents = json.loads(open("intents.json").read())
# print("Intents loaded successfully")

# Build vocabulary and datasets
words = []  # This is the unique words
classes = []  # This is the tags
documents = []  # List if tokens,tags

ignore_characters = ["?", "!", ".", ","]

# Load and process each intents
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        # Add pattern and tag to documents
        documents.append((word_list, intent["tag"]))

        # Add tag to classes list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])


# Normalize and clean the words
words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in ignore_characters
]

# Remove duplicates and sort the words
words = sorted(set(words))

# Remove duplicates and sord classes
classes = sorted(set(classes))

# Save workds and classes to pickle files
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
