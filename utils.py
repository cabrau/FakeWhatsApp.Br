#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:36:44 2020

@author: cabrau
"""
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import time
import emoji, re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import RSLPStemmer
import spacy
from unidecode import unidecode

#%% 
# user classification
def get_user_features(df,users):
    '''
    
    '''
    virals = []
    texto = []
    media = []
    msg_str = []
    misinformation = []
    n_groups = []
    n_users = []
    strenght = []
    n_users_mis = []
    strenght_mis = []
    urls = []
    mis_ratio = []
    
    for user in users.index:
        virals.append(len(df[(df['id'] == user) & (df['viral']==1)]))
        texto.append(len(df[(df['id'] == user) & (df['midia']==0)]))
        media.append(len(df[(df['id'] == user)]) - len(df[(df['id'] == user) & (df['midia']==0)]))
        misinformation.append(len(df[(df['id'] == user) & (df['misinformation']==1)]))
        n_groups.append(df[df['id']==user]['group'].nunique())
        urls.append(len(df[(df['id'] == user) & (df['url']==1)]))
        mis_ratio.append(len(df[(df['id'] == user) & (df['misinformation']==1)])/len(df[df['id']==user]))
        
        u_degree = 0
        u_strenght = 0
        groups = df[df['id']==user]['group'].unique()
        
        u_degree_mis = 0
        u_strenght_mis = 0
        
        for g in groups:
            users_in_group = df[df['group']==g]['id'].nunique()
            messages_in_group = len(df[(df['id']==user) &(df['group']==g)])            
            u_strenght += users_in_group*messages_in_group
            u_degree += users_in_group
            
            # misinformations
            users_misinformed = 0
            misinformation_in_group = len(df[(df['id']==user) & (df['group']==g) & (df['misinformation']==1)])
            if misinformation_in_group > 0:
                users_misinformed = users_in_group                
            u_strenght_mis += users_misinformed*misinformation_in_group
            u_degree_mis += users_misinformed  
            
        n_users.append(u_degree)
        strenght.append(u_strenght)
        n_users_mis.append(u_degree_mis)
        strenght_mis.append(u_strenght_mis)
        
        msgs = df[(df['id']==user) & (df['midia']==0)]['preprocessed_text']
        msgs = list(msgs.values)
        msgs = " ".join(msgs)
        msg_str.append(msgs)
        
    topUsers = pd.DataFrame({'id':users.index, 'total_messages':users.values, 'viral_messages':virals, 
                             'texts':texto, 'midia':media, 'urls':urls,
                             'number_of_groups':n_groups, 'reached_users':n_users,'messages_for_reached_user':strenght,
                             'misinformations': misinformation, 'misinformation_ratio':mis_ratio,
                             'users_misinformed':n_users_mis, 'misinformation_for_reached_user':strenght_mis,                             
                             'messages':msg_str})    
    return topUsers
    

def getTopUsers(df,top=10):
    groupedByid = df.groupby(['id']).count()
    groupedByid = groupedByid.sort_values('date', ascending=False)[0:top]['date']
    return get_user_features(df,groupedByid)

#%%
# text preprocessing 

#emojis and punctuation
emojis_list = list(emoji.UNICODE_EMOJI.keys())
emojis_list += ['\n']
punct = list(string.punctuation) 
emojis_punct = emojis_list + punct



#stop words removal
stop_words = list(stopwords.words('portuguese'))
new_stopwords = ['aí','pra','vão','vou','onde','lá','aqui',
                 'tá','pode','pois','so','deu','agora','todo',
                 'nao','ja','vc', 'bom', 'ai','ta', 'voce', 'alguem', 'ne', 'pq',
                 'cara','to','mim','la','vcs','tbm', 'tudo']
stop_words = stop_words #+ new_stopwords

def processEmojisPunctuation(text, remove_punct = False, remove_emoji = False):
    '''
    Put spaces between emojis. Removes punctuation.
    '''
    #get all unique chars
    chars = set(text)
    #for each unique char in text, do:
    for c in chars:
        
        if remove_punct: #remove punctuation            
            if c in punct: 
                text = text.replace(c, ' ')
        else: #put spaces between punctuation
            if c in punct:
                text = text.replace(c, ' ' + c + ' ')
        
        if remove_emoji: #remove emojis
            if c in emojis_list:
                text = text.replace(c, ' ')
        else: #put spaces between emojis
            if c in emojis_list:
                text = text.replace(c, ' ' + c + ' ')                        
            
    text = re.sub(' +', ' ', text)
    return text


def removeStopwords(text):    
    text = [t for t in text.split(' ') if t not in stop_words]
    text = ' '.join(text)
    text = re.sub(' +',' ',text)
    return text

def remove_numbers(text):
    pattern = '[0-9]+'
    text = re.sub(pattern, '[num]', text)
    return text

#lemmatization
nlp = spacy.load('pt_core_news_sm')
def lemmatization(text):
    doc = nlp(text)
    for token in doc:
        if token.text != token.lemma_:
            text = text.replace(token.text, token.lemma_)
    return text

# stemmer
stemmer = RSLPStemmer()
def stemming(text):
    words = text.split(' ')
    words = [stemmer.stem(w) if len(w) > 1 else w for w in words]
    text = ' '.join(words)
    return text

    

def domainUrl(text):
    '''
    Substitutes an URL in a text for the domain of this URL
    Input: an string
    Output: the string with the modified URL
    '''    
    if 'http' in text:
        re_url = '[^\s]*https*://[^\s]*'
        text = text.replace('youtu.be','youtube')
        text = text.replace('.com.br','')
        matches = re.findall(re_url, text, flags=re.IGNORECASE)
        for m in matches:
            domain = m.split('//')
            domain = domain[1].split('/')[0]
            text = re.sub(re_url, domain, text, 1)
        text = text.replace('www','')        
        return text
    else:
        return text
    
def processLoL(text):
    re_kkk = 'kkk*'
    t = re.sub(re_kkk, "kkk", text, flags=re.IGNORECASE)
    return t

def firstSentence(text):
    list_s = sent_split(text)
    for s in list_s:
        if s is not None:
            return s

def sent_split(text):
    list_s = re.split('\. |\! |\? |\n',text)
    return list_s    

def preprocess(text, semi=False, rpunct = False, remoji = False, lemma = False, stem = False):
            
    text = text.lower().strip()    
    text = domainUrl(text)
    text = processLoL(text)
    text = processEmojisPunctuation(text,remove_punct = rpunct, remove_emoji=remoji)    
    
    if not remoji:
        for word in set(text.split()):        
            if word not in emojis_list:            
                text = text.replace(word,unidecode(word))            
            
    text = remove_numbers(text)
    
    if semi:        
        return text
    
    text = removeStopwords(text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    
    if stem:
        text = stemming(text)
    
    if lemma:
        text = lemmatization(text)
        
    return text  
#%%
# testing

# text = 'Como são as coisas.\nChefe do jacaré aparece no video baleado e pessoal no video falando que é morador, falam muita merda tentando colocar as forças armadas como vilão.\nQuando chegou no hospital foi reconhecido e preso pela PM.'

# text = text.lower().strip()    
# text = domainUrl(text)
# text = processLoL(text)
# text = remove_numbers(text)
# text = processEmojisPunctuation(text,remove_punct = False, remove_emoji=False)
# text_list = text.split(' ')
# text = removeStopwords(text)
# text = stemming(text)

# print(repr(text))
# print(repr(preprocess(text,stem=True)))


#%%
# metrics

def getTestMetrics(y_test, y_pred, y_prob = [], full_metrics = False, print_charts = True):
    '''
    Plot charts e print performance metrics
    Input: predictions and labels
    Output: charts and metrics
    '''
    #print(metrics.classification_report(y_test, y_pred))
    
    f1 = metrics.f1_score(y_test, y_pred, pos_label = 1, average = 'binary') #macro
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, pos_label = 1, average = 'binary')
    recall = metrics.recall_score(y_test, y_pred, pos_label = 1, average = 'binary')
    
    if full_metrics:
        f1_neg = metrics.f1_score(y_test, y_pred, pos_label = 0, average = 'binary')
        precision_neg = metrics.f1_score(y_test, y_pred, pos_label = 0, average = 'binary')
        recall_neg = metrics.f1_score(y_test, y_pred, pos_label = 0, average = 'binary')
    
    #confusion matrix
    if print_charts:
        print(metrics.classification_report(y_test, y_pred))
        cf_matrix = metrics.confusion_matrix(y_test, y_pred)
        group_names = ['TN','FP','FN','TP']
        group_counts = ['{0:0.0f}'.format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in
                             cf_matrix.flatten()/np.sum(cf_matrix)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
                  zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.title('Confusion matrix')
        sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    
    roc_auc = 0

    if len(y_prob) > 0:
        #auroc
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        print('AUC: ',roc_auc)
        #plot
        if print_charts:
            lw = 2
            plt.subplot(122)
            plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve')
            plt.legend(loc="lower right")
            plt.show()
    
    if not full_metrics:
        results = (acc, precision, recall, f1, roc_auc)
    else:
        results = (acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc)
    
    return results
