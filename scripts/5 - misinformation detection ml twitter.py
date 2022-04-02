#!/usr/bin/env python
# coding: utf-8

# In[1]:


#utils
import emoji, re, string, time, os
from utils import getTestMetrics
import pandas as pd
import numpy as np
from scipy.stats import randint
import pickle

#nlp
import nltk
from nltk.corpus import stopwords
import spacy

#dataviz
import matplotlib.pyplot as plt
import seaborn as sns

#features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

#models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble  import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

#data balancing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

sns.set(style="darkgrid")

# In[2]:


base = 'twitter'
subset = 'twitter'
path_dir = '/results/2018/viral/ml twitter/'
path_dir



# In[20]:


experiments = ['ml-tfidf-unibitri_gram-random_oversampling',
 'ml-tfidf-unibi_gram-random_oversampling',
 'ml-tfidf-unibitri_gram-processed-random_oversampling',
 'ml-tfidf-unibitriquad_gram-processed-random_oversampling',
 'ml-tfidf-bigram-random_oversampling',
 'ml-bow-unibitri_gram-random_oversampling',
 'ml-tfidf-processed-smote',
 'ml-bow-processed-random_oversampling',
 'ml-tfidf-random_oversampling',
 'ml-tfidf-processed-random_oversampling',
 'ml-tfidf-smote',
 'ml-tfidf-undersampling',
 'ml-tfidf',
 'ml-tfidf-processed',
 'ml-bow-unibitri_gram-processed-random_oversampling',
 'ml-bow-random_oversampling',
 'ml-bow',
 'ml-bow-processed',
 'ml-bow-random_oversampling-processed',
 'ml-bow-random_oversampling-max_features',
 'ml-tfidf-random_oversampling-max_features',
 'ml-tfidf-processed-random_oversampling-max_features',
 'ml-bow-processed-random_oversampling-max_features',
 'ml-tfidf-trigram-random_oversampling',
 'ml-bow-processed-smote']



# # Begin experiment

# In[21]:

for experiment in experiments:
#%%
    #experiment = 'ml-tfidf-unibitri_gram-processed-random_oversampling'
    start_time = time.time()
    url_twitter = 'https://raw.githubusercontent.com/prc992/FakeTweet.Br/master/FakeTweetBr.csv'
    twitter = pd.read_csv(url_twitter)
    url_twitter_test = 'https://raw.githubusercontent.com/prc992/FakeTweet.Br/master/FakeTweetBr-Test.csv'
    twitter_test = pd.read_csv(url_twitter_test)
    
    
# In[18]:

    data_dir = 'data/' + str(base) #+ '/vis_processed_texts.p'
    preprocessed = False # the texts were already pre-processed
    processed_texts_filename = 'processed_texts-'+subset+'.p'
    for filename in os.listdir(data_dir):
        print(filename)
        if filename == processed_texts_filename:
            preprocessed = True
    preprocessed     

    
     
    # In[23]:
    #targets
    texts_train = twitter['text']
    texts_test = twitter_test['text']
    y_train = [1 if s == 'fake' else 0 for s in twitter['classificacao']]
    y_test = [1 if s == 'fake' else 0 for s in twitter_test['classificacao']]      
  
    # In[26]:
    
    
    print(len(texts_train))
    print(len(y_train))
    
    # In[27]:
    
    
    #emojis and punctuation
    emojis_list = list(emoji.UNICODE_EMOJI.keys())
    punct = list(string.punctuation)
    emojis_punct = emojis_list + punct
    
    def processEmojisPunctuation(text, remove_punct = True):
        '''
        Put spaces between emojis. Removes punctuation.
        '''
        #get all unique chars
        chars = set(text)
        #for each unique char in text, do:
        for c in chars:
            #remove punctuation
            if remove_punct:
                if c in emojis_list:
                    text = text.replace(c, ' ' + c + ' ')
                if c in punct:
                    text = text.replace(c, ' ')
                    
            #put spaces between punctuation
            else:
                if c in emojis_punct:
                    text = text.replace(c, ' ' + c + ' ')          
                
        text = text.replace('  ', ' ')
        return text
    
    #stop words removal
    stop_words = list(stopwords.words('portuguese'))
    new_stopwords = ['aí','pra','vão','vou','onde','lá','aqui',
                     'tá','pode','pois','so','deu','agora','todo',
                     'nao','ja','vc', 'bom', 'ai','kkk','kkkk','ta', 'voce', 'alguem', 'ne', 'pq',
                     'cara','to','mim','la','vcs','tbm', 'tudo']
    stop_words = stop_words + new_stopwords
    final_stop_words = []
    for sw in stop_words:
        sw = ' '+ sw + ' '
        final_stop_words.append(sw)
    
    def removeStopwords(text):
        for sw in final_stop_words:
            text = text.replace(sw,' ')
        text = text.replace('  ',' ')
        return text
    
    #lemmatization
    nlp = spacy.load('pt_core_news_sm')
    def lemmatization(text):
        doc = nlp(text)
        for token in doc:
            if token.text != token.lemma_:
                text = text.replace(token.text, token.lemma_)
        return text
        
    
    def domainUrl(text):
        '''
        Substitutes an URL in a text for the domain of this URL
        Input: an string
        Output: the string with the modified URL
        '''    
        if 'http' in text:
            re_url = '[^\s]*https*://[^\s]*'
            matches = re.findall(re_url, text, flags=re.IGNORECASE)
            for m in matches:
                domain = m.split('//')
                domain = domain[1].split('/')[0]
                text = re.sub(re_url, domain, text, 1)
            return text
        else:
            return text 
    
    def preprocess(text):
        text = text.lower().strip()
        text = domainUrl(text)
        text = processEmojisPunctuation(text)
        text = removeStopwords(text)
        text = lemmatization(text)
        return text
               
    
    
    # In[28]:
    
    
    #if experiment is with pre-processed text
    if 'processed' in experiment:
            #text was already pre-processed
            if preprocessed:
                pro_texts = pickle.load(open( "data/twitter/processed_texts-twitter.p", "rb" ))
                pro_texts_test = pickle.load(open( "data/twitter/processed_texts_test-twitter.p", "rb" ))
            else:
                pro_texts = [preprocess(t) for t in texts_train]
                pro_texts_test = [preprocess(t) for t in texts_test]
                pickle.dump(pro_texts, open( "data/twitter/processed_texts-twitter.p", "wb" ))
                pickle.dump(pro_texts_test, open( "data/twitter/processed_texts_test-twitter.p", "wb" ))
            
               
    else:
        #only use lowercase and separates emojis and punctuation
        pro_texts = [processEmojisPunctuation(t.lower(),remove_punct = False) for t in texts_train]
        pro_texts_test = [processEmojisPunctuation(t.lower(),remove_punct = False) for t in texts_test]
     
    

    
    # In[32]:
    
    
    max_feat = 500
    
    if 'tfidf' in experiment:
        if 'max_features' in experiment:
            vectorizer = TfidfVectorizer(max_features = max_feat)
        elif 'bigram' in experiment:
            vectorizer = TfidfVectorizer(ngram_range =(2,2))
        elif 'trigram' in experiment:
            vectorizer = TfidfVectorizer(ngram_range =(3,3)) 
        elif 'unibi_gram' in experiment:
            vectorizer = TfidfVectorizer(ngram_range =(1,2))
        elif 'unibitri_gram' in experiment:
            vectorizer = TfidfVectorizer(ngram_range =(1,3))       
        elif 'unibitriquad_gram' in experiment:
            vectorizer = TfidfVectorizer(ngram_range =(1,3))  
        else:
            vectorizer = TfidfVectorizer()
            
    elif 'bow' in experiment:
        if 'max_features' in experiment:
            vectorizer = CountVectorizer(max_features = max_feat, binary=True)
        elif 'bigram' in experiment:
            vectorizer = CountVectorizer(binary=True, ngram_range =(2,2))
        elif 'trigram' in experiment:
            vectorizer = CountVectorizer(binary=True, ngram_range =(3,3)) 
        elif 'unibi_gram' in experiment:
            vectorizer = CountVectorizer(binary=True, ngram_range =(1,2))
        elif 'unibitri_gram' in experiment:
            vectorizer = CountVectorizer(binary=True, ngram_range =(1,3))
        else:
            vectorizer = CountVectorizer(binary=True)
    
    vectorizer.fit(texts_train)   
    X_train = vectorizer.transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    
    # In[35]:
    
    
    if 'smote' in experiment:
        #oversampling with SMOTE
        sm = SMOTE(random_state = 42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    elif 'undersampling' in experiment:
        rus = RandomUnderSampler(random_state = 42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    elif 'random_oversampling' in experiment:
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        
    X_train.shape
    
    
    # In[36]:
    
    
    vocab_size = X_train.shape[1]
    vocab_size
    
    
    # ## Metrics
    
    # In[38]:
    
    
    #metrics
    scenario = []
    model = []
    accuracy_score = []
    precision_score = []
    precision_score_neg = []
    recall_score = []
    recall_score_neg = []
    f1_score = []
    f1_score_neg = []
    auc_score = []
    
    
    # ## Models training and test
    
    # In[39]:
    
    
    print('Logistic Regression')
    logreg = LogisticRegression().fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:,1]
    model.append('logistic regression')
    acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc = getTestMetrics(y_test, y_pred, y_prob, full_metrics = True, print_charts = False)
    accuracy_score.append(acc)
    precision_score.append(precision)
    precision_score_neg.append(precision_neg)
    recall_score.append(recall)
    recall_score_neg.append(recall_neg)
    f1_score.append(f1)
    f1_score_neg.append(f1_neg)
    auc_score.append(roc_auc)
    
    
    # In[25]:
    
    
    print('Bernoulli Naive-Bayes')
    bnb = BernoulliNB().fit(X_train, y_train)
    y_pred = bnb.predict(X_test)
    y_prob = bnb.predict_proba(X_test)[:,1]
    model.append('bernoulli naive-bayes')
    acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc = getTestMetrics(y_test, y_pred, y_prob, full_metrics = True, print_charts = False)
    accuracy_score.append(acc)
    precision_score.append(precision)
    precision_score_neg.append(precision_neg)
    recall_score.append(recall)
    recall_score_neg.append(recall_neg)
    f1_score.append(f1)
    f1_score_neg.append(f1_neg)
    auc_score.append(roc_auc)
    
    
    # In[40]:
    
    
    print('Multinomial Naive-Bayes')
    mnb = MultinomialNB().fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    y_prob = mnb.predict_proba(X_test)[:,1]
    model.append('multinomial naive-bayes')
    acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc = getTestMetrics(y_test, y_pred, y_prob, full_metrics = True, print_charts = False)
    accuracy_score.append(acc)
    precision_score.append(precision)
    precision_score_neg.append(precision_neg)
    recall_score.append(recall)
    recall_score_neg.append(recall_neg)
    f1_score.append(f1)
    f1_score_neg.append(f1_neg)
    auc_score.append(roc_auc)
    
    
    # In[41]:
    
    
    print('Linear Support Vector Machine')
    svm = LinearSVC(dual=False).fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    model.append('linear svm')
    acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc = getTestMetrics(y_test, y_pred, full_metrics = True, print_charts = False)
    accuracy_score.append(acc)
    precision_score.append(precision)
    precision_score_neg.append(precision_neg)
    recall_score.append(recall)
    recall_score_neg.append(recall_neg)
    f1_score.append(f1)
    f1_score_neg.append(f1_neg)
    auc_score.append(roc_auc)
    
    
    # In[42]:
    
    
    print('KNN')
    knn = KNeighborsClassifier().fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:,1]
    model.append('knn')
    acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc = getTestMetrics(y_test, y_pred, y_prob, full_metrics = True, print_charts = False)
    accuracy_score.append(acc)
    precision_score.append(precision)
    precision_score_neg.append(precision_neg)
    recall_score.append(recall)
    recall_score_neg.append(recall_neg)
    f1_score.append(f1)
    f1_score_neg.append(f1_neg)
    auc_score.append(roc_auc)
    
    
    # In[45]:
    
    
    print('Linear SVM with SGD training.')
    sgd = SGDClassifier().fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    model.append('sgd')
    
    acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc = getTestMetrics(y_test, y_pred, full_metrics = True, print_charts = False)
    
    accuracy_score.append(acc)
    precision_score.append(precision)
    precision_score_neg.append(precision_neg)
    recall_score.append(recall)
    recall_score_neg.append(recall_neg)
    f1_score.append(f1)
    f1_score_neg.append(f1_neg)
    auc_score.append(roc_auc)
    
    
    # In[43]:
    
    
    print('Random Forest')
    rf = RandomForestClassifier().fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:,1]
    model.append('random forest')
    acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc = getTestMetrics(y_test, y_pred, y_prob, full_metrics = True, print_charts = False)
    accuracy_score.append(acc)
    precision_score.append(precision)
    precision_score_neg.append(precision_neg)
    recall_score.append(recall)
    recall_score_neg.append(recall_neg)
    f1_score.append(f1)
    f1_score_neg.append(f1_neg)
    auc_score.append(roc_auc)
    
    
    # In[44]:
    
    
    print('Gradient Boosting')
    gb = GradientBoostingClassifier(n_estimators=200).fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    y_prob = gb.predict_proba(X_test)[:,1]
    model.append('gradient boosting')
    acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc = getTestMetrics(y_test, y_pred, y_prob, full_metrics = True, print_charts = False)
    accuracy_score.append(acc)
    precision_score.append(precision)
    precision_score_neg.append(precision_neg)
    recall_score.append(recall)
    recall_score_neg.append(recall_neg)
    f1_score.append(f1)
    f1_score_neg.append(f1_neg)
    auc_score.append(roc_auc)
    
    
    # In[46]:
    
    
    print('Multilayer perceptron')
    mlp = MLPClassifier(max_iter = 6, verbose=True, early_stopping= True).fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:,1]
    model.append('mlp')
    acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc = getTestMetrics(y_test, y_pred, y_prob, full_metrics = True, print_charts = False)
    accuracy_score.append(acc)
    precision_score.append(precision)
    precision_score_neg.append(precision_neg)
    recall_score.append(recall)
    recall_score_neg.append(recall_neg)
    f1_score.append(f1)
    f1_score_neg.append(f1_neg)
    auc_score.append(roc_auc)
    
    
    # ## Results
    
    # In[47]:
    
    
    end_time = time.time()
    ellapsed_time = end_time - start_time
    print('ellapsed time (min):', ellapsed_time/60)
    
    
    # In[48]:
    
    
    df_metrics = pd.DataFrame({'model':model,                                 
                                     'vocab':[vocab_size]*len(model),
                                     'auc score': auc_score,
                                     'accuracy':accuracy_score,
                                     'precision 1': precision_score,
                                     'recall 1': recall_score,
                                     'f1 score 1': f1_score,
                                     'precision 0': precision_score_neg,
                                     'recall 0': recall_score_neg,                                 
                                     'f1 score 0': f1_score_neg
                                     })
    
    df_metrics['precision avg'] = (df_metrics['precision 1'] + df_metrics['precision 0'])/2
    df_metrics['recall avg'] = (df_metrics['recall 1'] + df_metrics['recall 0'])/2
    df_metrics['f1 avg'] = (df_metrics['f1 score 1'] + df_metrics['f1 score 0'])/2

    
    
    # In[49]:    
    
    
    
    # In[34]:
    
    
    filepath = 'results/2018/viral/ml twitter/' + experiment + '.csv'
    print(filepath)
    
    
    # In[35]:
    
    
    df_metrics.to_csv(filepath, index = False)


# # Best results

