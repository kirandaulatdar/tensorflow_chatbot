from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import mysql.connector
from datetime import datetime

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# Load model and data
model = load_model('models/chatbot_model.h5')
with open('data/intents.json') as file:
    data = json.load(file)

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="india"
)

words = []
classes = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words]))
classes = sorted(set(classes))

def preprocess_input(input_text):
    input_words = nltk.word_tokenize(input_text)
    input_words = [lemmatizer.lemmatize(word.lower()) for word in input_words]
    bag = [1 if w in input_words else 0 for w in words]
    return np.array(bag).reshape(1, len(words))

def get_response(prediction, user_input):
    index = np.argmax(prediction)
    tag = classes[index]
    print(f'Predicted tag: {tag}')  # Debugging line

    if tag == 'lead_query':
        return handle_lead_query(user_input)

    for intent in data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

def handle_lead_query(message):
    cursor = db.cursor(dictionary=True)
    if "how many leads" in message.lower():
        cursor.execute("SELECT COUNT(*) AS lead_count FROM business_lead")
        result = cursor.fetchone()
        return f"We have {result['lead_count']} leads."
    
    if "leads between" in message.lower():
        try:
            dates = message.lower().split("between")[1].strip().split("and")
            start_date = datetime.strptime(dates[0].strip(), '%Y-%m-%d')
            end_date = datetime.strptime(dates[1].strip(), '%Y-%m-%d')
            
            query = "SELECT COUNT(*) AS lead_count FROM business_lead WHERE lead_date BETWEEN %s AND %s"
            cursor.execute(query, (start_date, end_date))
            result = cursor.fetchone()
            return f"We have {result['lead_count']} leads between {start_date.date()} and {end_date.date()}."
        except Exception as e:
            return f"Error processing dates: {e}"
    
    return "I'm not sure how to answer that lead query."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['message']
    print(f'User input: {user_input}')  # Debugging line
    bag_of_words = preprocess_input(user_input)
    prediction = model.predict(bag_of_words)
    print(f'Prediction: {prediction}')  # Debugging line
    response = get_response(prediction, user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)