import json
import random as random
import pickle
import tensorflow as tf
import numpy as np
import translators as ts
from flask import Flask, request, render_template, redirect, url_for
from flask_restful import Resource, Api
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
api = Api(app)


with open('sarcasm_pickle','rb') as r:
    sarcasm_detector = pickle.load(r)

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

with open("sarcasm_v2_processed.json",'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

sentence = []
ts._google.language_map


verylow = ["This headline has a very low indication of sarcasm", "The writer is serious about it", "Not sarcastic, but is it based on fact?"]
mild = ["Mild level sarcasm has been detected","The writer had no clear intention while doing so"]
moderate = ["Moderate level of sarcasm has been detected"," Maybe it is sarcastic, maybe it isn't..."]
high = ["This headline is very sarcastic !", "Take it with a grain of salt"]

        
@app.route('/', methods=['GET','POST'])
def index():
    if request.method =='GET':
        return render_template('index.html',  sarc=101)



    elif request.method == 'POST':    

        phrase = request.form.get("phrase")
        t_phrase = ts.google(phrase, from_language='id', to_language='en')
        sentence.insert(0,t_phrase)
        sequences = tokenizer.texts_to_sequences(sentence)
        padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        hasil = sarcasm_detector.predict(padded).tolist()
        chance = hasil[0][0]
        persenan = round(chance*100,2)
        str_persenan = str(persenan)+"%"
        return render_template('index.html', translated=t_phrase, sarc=persenan,  str_sarc =str_persenan , 
        vl = random.choice(verylow),
        mi = random.choice(mild),
        mo = random.choice(moderate),
        h = random.choice(high),
        )

# @app.route('/hasil', methods = ['POST'])
# def hasil():
#     phrase = request.form.get("phrase")

#     t_phrase = ts.google(phrase, from_language='id', to_language='en')
#     sentence.insert(0,t_phrase)
#     sequences = tokenizer.texts_to_sequences(sentence)
#     padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
#     hasil = sarcasm_detector.predict(padded).tolist()
#     chance = hasil[0][0]
#     persenan = round(chance*100,2)
#     str_persenan = str(persenan)+"%"

#     return render_template('index.html', t_phrase=t_phrase, persenan=persenan, str_persenan=str_persenan)




if __name__ == '__main__':
    app.run(debug=False, threaded=True)
    
# class mainHome(Resource):
#     def get(self):
#         return render_template('index.html')

# api.add_resource(mainHome,'/')