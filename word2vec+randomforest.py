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
