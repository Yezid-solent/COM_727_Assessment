# Import libraries

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

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


# Create training data

training = []

output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]

    # lemmatize each word in the pattern
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Create bag of words, 0 if word does not exists and 1 if it exists
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Create output row, i.e tag one-hot encoding
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])


# randomized the training data
random.shuffle(training)

# Convert training to numpy array
training = np.array(training, dtype=object)

# split training data to train and test i.e X,y
# train_x = list(training[:, 0])
# train_y = list(training[:, 1])

train_x = np.array([item[0] for item in training], dtype="float32")
train_y = np.array([item[1] for item in training], dtype="float32")


# Build the ANN model

model = Sequential()

# Set up the first input + first hidden layer
model.add(
    Dense(
        128,
        input_shape=(len(train_x[0]),),
        activation="relu",
    )
)
model.add(Dropout(0.5))

# Add second hidden layer
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))


# Add output layer
model.add(Dense(len(train_y[0]), activation="softmax"))

print("\n Model created")
model.summary()

# Add model optimizer
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# compile model
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train model
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# save model and history

model.save("chatbot_model.keras")

with open("training_history.json", "w", encoding="utf-8") as f:
    json.dump(history.history, f, ensure_ascii=False, indent=2)


print("Training completed and model saved to 'chatbot_mode.keras'")
