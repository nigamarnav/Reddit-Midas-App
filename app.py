# Load libraries
from html.parser import HTMLParser
from urllib.request import urlopen
import urllib
import flask
from flask import Flask, jsonify, render_template, request
from werkzeug import secure_filename
import pandas as pd
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD
import nltk, re, string
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# instantiate flask 
app = flask.Flask(__name__)

@app.route("/")
def home():
	return render_template("index.html")
	
c_dict = {"ain't": "am not", "aren't": "are not", "can't": "cannot","can't've": "cannot have"}	
add_stop = ['', ' ', 'say', 's', 'u', 'ap', 'afp', '...', 'n', '\\']
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df = 1, max_df = .95)
stop_words = ENGLISH_STOP_WORDS.union(add_stop)
punc = list(set(string.punctuation))
c_re = re.compile('(%s)' % '|'.join(c_dict.keys()))
lemmatizer = WordNetLemmatizer() 
tokenizer = TweetTokenizer()
lsa = TruncatedSVD(n_components=10, n_iter=10, random_state=3)
	
def expandContractions(text, c_re=c_re):
	def replace(match):
		return c_dict[match.group(0)]
	return c_re.sub(replace, text)
	
def get_word_net_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return None
	
def casual_tokenizer(text): #Splits words on white spaces (leaves contractions intact) and splits out trailing punctuation
	tokens = tokenizer.tokenize(text)
	return tokens

def lemma_wordnet(tagged_text):
	final = []
	for word, tag in tagged_text:
		wordnet_tag = get_word_net_pos(tag)
		if wordnet_tag is None:
			final.append(lemmatizer.lemmatize(word))
		else:
			final.append(lemmatizer.lemmatize(word, pos=wordnet_tag))
	return final
		
def process_text(text):
	soup = BeautifulSoup(text, "lxml")
	tags_del = soup.get_text()
	no_html = re.sub('<[^>]*>', '', tags_del)
	tokenized = casual_tokenizer(no_html)
	lower = [item.lower() for item in tokenized]
	decontract = [expandContractions(item, c_re=c_re) for item in lower]
	tagged = nltk.pos_tag(decontract)
	lemma = lemma_wordnet(tagged)
	no_num = [re.sub('[0-9]+', '', each) for each in lemma]
	no_punc = [w for w in no_num if w not in punc]
	no_stop = [w for w in no_punc if w not in stop_words]
	return no_stop
	
# prediction function 
def ValuePredictor(to_predict_list): 
	texts = process_text(to_predict_list)
	to_predict = tfidf_vectorizer.fit_transform(texts) #features
	to_predict = lsa.fit_transform(to_predict)
	loaded_model = pickle.load(open("model10.pickle", "rb"))
	result = loaded_model.predict(to_predict)
	return result[0] 
  
@app.route('/result', methods = ['POST']) 
def result():
	if request.method == 'POST':
		to_predict_list = request.form['title'] + " " + request.form['Post']
		result = ValuePredictor(to_predict_list)
		if int(result)== 0:
			prediction='AskIndia'
		elif int(result)== 1:
			prediction ='Business/Finance'
		elif int(result)== 2:
			prediction ='Coronavirus'
		elif int(result)== 3:
			prediction ='Food'
		elif int(result)== 4:
			prediction ='Politics'
		elif int(result)== 5:
			prediction ='Photography'
		elif int(result)== 6:
			prediction ='Policy/Economy'
		elif int(result)== 7:
			prediction ='Non-Political'
		elif int(result)== 8:
			prediction ='Scheduled'
		elif int(result)== 9:
			prediction ='Science/Technology'
		elif int(result)== 10:
			prediction ='Sports'
		elif int(result)== 11:
			prediction ='[R]eddiquette'
		return render_template("result.html", prediction = prediction) 
		
		
def error_callback(*_, **__):
    pass

def is_string(data):
    return isinstance(data, str)

def is_bytes(data):
    return isinstance(data, bytes)

def to_ascii(data):
    if is_string(data):
        data = data.encode('ascii', errors='ignore')
    elif is_bytes(data):
        data = data.decode('ascii', errors='ignore')
    else:
        data = str(data).encode('ascii', errors='ignore')
    return data


class Parser(HTMLParser):
    def __init__(self, url):
        self.title = None
        self.rec = False
        HTMLParser.__init__(self)
        try:
            self.feed(to_ascii(urlopen(url).read()))
        except urllib.error.HTTPError:
            return
        except urllib.error.URLError:
            return
        except ValueError:
            return

        self.rec = False
        self.error = error_callback

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self.rec = True

    def handle_data(self, data):
        if self.rec:
            self.title = data

    def handle_endtag(self, tag):
        if tag == 'title':
            self.rec = False


def get_title(url):
    return Parser(url).title
		
@app.route('/automated_testing', methods = ['POST'])
def testing():
	result = []
	url = []
	if request.method == 'POST':
		file = request.files['files']
		filename = secure_filename(file.filename)
		
		with open(filename) as f:
			file_line = f.readline()
			to_predict_list = get_title(file_line)
			r = ValuePredictor(to_predict_list)
			url.append(file_line)
			result.append(r)
		return jasonify(key=url, value=result)
			
		
if __name__ == '__main__':
	app.run(debug=True)