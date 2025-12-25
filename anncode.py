# Import libraries

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents.json file
intents = json.loads(open("intents.json").read())
# print("Intents loaded successfully")
