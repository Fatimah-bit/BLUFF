# math and parse
import numpy as np
import pandas as pd
import json
#visualisation
import seaborn as sns
# plotting
import matplotlib.pyplot as plt
#natural language toolkit
import nltk
import gensim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

# from sklearn.preprocessing import LabelBinarizer
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from wordcloud import WordCloud,STOPWORDS
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize,sent_tokenize
# from bs4 import BeautifulSoup
# import re,string,unicodedata
# #from keras.preprocessing import text, sequence
# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# from sklearn.model_selection import train_test_split
# from string import punctuation
# from gensim.models import Word2Vec
# #import torchtext
# #from torchtext.data import get_tokenizer
# from nltk.tokenize import word_tokenize
# #from torchtext.data.utils import get_tokenizer
# #from torchtext.vocab import build_vocab_from_iterator 
# #from torchnlp.encoders.text import StaticTokenizerEncoder,stack_and_pad_tensors,pad_tensor
# from torch.nn.utils.rnn import pad_sequence

#import keras
#from keras.models import Sequential
#from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional,GRU
#import tensorflow as tf

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
#loading dataset 
data = pd.read_json("./Sarcasm_Headlines_Dataset_v2.json", lines=True)

#deleting article_link column
del data['article_link']

# number of non sarcastic headlines
#print (data.shape[0]-data.is_sarcastic.sum())

# number of sarcastic headlines
#print(data.is_sarcastic.sum())

# remove stopwords as they don't add any meanings to the sentece


data ['cleaned'] = data['headline'].apply (lambda x: gensim.utils.simple_preprocess(x))


# wordcloud for sarcastic text
'''
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(data[data.is_sarcastic == 1].headline))
plt.imshow(wc , interpolation = 'bilinear')
plt.title('Wordcloud for sarcastic headlines')
plt.savefig('Wordcloud for sarcastic headlines.png')

#wordcloud for non sarcastic text
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(data[data.is_sarcastic == 0].headline))
plt.imshow(wc , interpolation = 'bilinear')
plt.title('Wordcloud for non sarcastic headlines')
plt.savefig('Wordcloud for non sarcastic headlines.png')
'''

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')
x_train = df_train ['headline'].apply (lambda x: gensim.utils.simple_preprocess(x))
y_train = df_train ['is_sarcastic']
x_test = df_test ['headline'].apply (lambda x: gensim.utils.simple_preprocess(x))
y_test = df_test ['is_sarcastic']

    
# Train the word2vec model
w2v_model = gensim.models.Word2Vec(x_train,vector_size=200,window=5,min_count=1)

#most similar words for 'man'
#print (w2v_model.wv.most_similar('man'))


words = set(w2v_model.wv.index_to_key )
x_train_v = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in x_train])
x_test_v = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in x_test])



    
x_train_v_avg = []
x_test_v_avg = []

for v in x_train_v:
    if v.size:
        x_train_v_avg.append(v.mean(axis=0))
    else:
        x_train_v_avg.append(np.zeros(100, dtype=float))

for v in x_test_v:
    if v.size:
        x_test_v_avg.append(v.mean(axis=0))
    else:
        x_test_v_avg.append(np.zeros(100, dtype=float))


rf = RandomForestClassifier()
rf_model = rf.fit(x_train_v_avg, y_train.values.ravel())

#used trained model for predicting the unseen data (test data)
y_pred = rf_model.predict(x_test_v_avg)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))