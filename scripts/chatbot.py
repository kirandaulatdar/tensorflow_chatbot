from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import mysql.connector
from datetime import datetime
import pickle

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Load model and necessary data
model = load_model('../models/chatbot_model.h5')
lemmatizer = WordNetLemmatizer()

with open('../data/intents.json') as file:
    intents = json.load(file)

words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="india"
)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, message, intents_json):
    tag = ints[0]['intent']
    
    # If the intent is 'lead_query', handle it separately
    if tag == 'lead_query':
        return handle_lead_query(message)
    
    # Otherwise, use predefined responses
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I don't understand!"

@app.route('/ask', methods=['POST'])
def ask():
    message = request.form['message']
    ints = predict_class(message, model)
    res = get_response(ints, message, intents)
    
    return jsonify({'response': res})

def handle_lead_query(message):
    cursor = db.cursor(dictionary=True)
    # Example: "How many leads do we have?"
    if "how many leads" in message.lower():
        cursor.execute("SELECT COUNT(*) AS lead_count FROM business_lead")
        result = cursor.fetchone()
        return f"We have {result['lead_count']} leads."
    
    # Example: "How many leads between 2023-01-01 and 2023-01-31?"
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
    
    return "I'm not sure how to answer that."

if __name__ == "__main__":
    app.run(debug=True)
