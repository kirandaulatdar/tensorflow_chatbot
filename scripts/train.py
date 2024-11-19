import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load Data
with open('../data/intents.json') as file:
    data = json.load(file)

# Preprocess Data
words = []
classes = []
documents = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words]))
classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

    # Print statements to debug the shapes
    print(f'bag: {bag}')
    print(f'output_row: {output_row}')

random.shuffle(training)
training = np.array(training, dtype=object)  # Ensure the array elements are objects

train_x = np.array(list(training[:, 0]), dtype=object)
train_y = np.array(list(training[:, 1]), dtype=object)

# Build Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(np.array(train_x.tolist()), np.array(train_y.tolist()), epochs=200, batch_size=5, verbose=1)
model.save('../models/chatbot_model.h5')
