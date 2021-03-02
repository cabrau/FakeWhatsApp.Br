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
import emoji, re, string
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import spacy
from unidecode import unidecode

#%% 
# user classification

def get_text_content(df,user,column):
    msgs = df[(df['id']==user) & (df['midia']==0)][column]
    msgs = list(msgs.values)
    msgs = "<>".join(msgs)
    return msgs

def get_network_features(df, user, group, users_reached):
    users_in_group = df[df['group']==group]['id'].unique()
    n_messages_in_group = len(df[(df['group']==group) & (df['id']==user)])
    
    for u in users_in_group:
        if u in users_reached:
            users_reached[u] += n_messages_in_group
        else:
            users_reached[u] = n_messages_in_group
    return users_reached

def get_user_features(df,users):
    '''    
    '''
    # whatsapp features
    n_messages = []
    n_groups = [] 
    n_virals = []
    n_text = []
    n_midia = []
    viral_ratio = []
    midia_ratio = []
    text_ratio = []
    
    # content features
    viral_texts = []
    
    #labeled features
    n_misinformation = []
    message_misinformation_ratio = []
    viral_misinformation_ratio = []
    
    # network features: nodes are users #####################
    
    # messages network: links are messages sent by a user to other in the same group. 
    # weight is the number of messages
    messages_degree_centrality = [] # sum of links
    messages_strenght = [] # sum of weights in links
    #messages_eigencentrality = [] #eigencentrality
    
    # viral network: links are viral messages sent by a user to other in the same group. 
    # weight is the number of messages
    viral_degree_centrality = [] # sum of links
    viral_strenght = [] # sum of weights in links
    #viral_eigencentrality = [] #eigencentrality
    
    # misinformation network: links are misinformation sent by a user to other in the same group. 
    # weight is the number of messages
    mis_degree_centrality = [] # sum of links
    mis_strenght = [] # sum of weights in links
#     messages_eigencentrality = [] #eigencentrality   
    
    for user in users.index:       
        df_user = df[df['id']==user]
        
        #whatsapp features
        n_messages.append(len(df_user))
        n_text.append(len(df_user[df_user['midia']==0]))
        n_midia.append(len(df_user[df_user['midia']==1]))
        n_virals.append(len(df_user[(df_user['viral']==1)]))
        groups = df_user['group'].unique()
        n_groups.append(df_user['group'].nunique())
        #ratios
        viral_ratio.append(n_virals[-1]/n_messages[-1])
        midia_ratio.append(n_midia[-1]/n_messages[-1])
        text_ratio.append(n_text[-1]/n_messages[-1])
        
        #labelled features
        n_misinformation.append(len(df_user[df_user['misinformation']==1]))
        message_misinformation_ratio.append(n_misinformation[-1]/n_messages[-1])
        if n_virals[-1] > 0:
            viral_misinformation_ratio.append(n_misinformation[-1]/n_virals[-1])
        else:
            viral_misinformation_ratio.append(0)
        
        #content (lemmatized)
        col = 'preprocessed_text_lemma'
        virals = get_text_content(df[df['viral']==1],user,col)
        viral_texts.append(virals)     
        
        #graph features         
        users_reached = {}
        users_reached_viral = {}
        users_reached_mis = {}
        
        # iterate through groups where user interacted
        for g in groups:
            users_reached = get_network_features(df, user, g, users_reached)
            users_reached_viral = get_network_features(df[df['viral']==1], user, g, users_reached_viral)
            users_reached_mis = get_network_features(df[df['misinformation']==1], user, g, users_reached_mis)
        
        # message network
        users_reached = pd.Series(users_reached)[1:]
        messages_degree_centrality.append(users_reached.count())
        messages_strenght.append(users_reached.sum())
        
        # viral network
        users_reached_viral = pd.Series(users_reached_viral)[1:]
        viral_degree_centrality.append(users_reached_viral.count())
        viral_strenght.append(users_reached_viral.sum())
        
        # misinformation network
        users_reached_mis = pd.Series(users_reached_mis)[1:]
        mis_degree_centrality.append(users_reached_mis.count())
        mis_strenght.append(users_reached_mis.sum())  
        
        
    #credibility = 1-pd.Series(viral_misinformation_ratio)  
    df_users = pd.DataFrame({'id':users.index, 'groups':n_groups, 'number_of_messages':n_messages,
                            'texts':n_text,'text_ratio':text_ratio,
                             'midia':n_midia,'midia_ratio':midia_ratio,
                             'virals':n_virals,'viral_ratio':viral_ratio,                             
                             'degree_centrality':messages_degree_centrality,
                             'strenght':messages_strenght,
                             'viral_degree_centrality':viral_degree_centrality,
                             'viral_strenght':viral_strenght,
                             'misinformation':n_misinformation,
                             'misinformation_degree_centrality':mis_degree_centrality,
                             'misinformation_strenght':mis_strenght,
                             'misinformation_ratio': message_misinformation_ratio,
                             'viral_misinformation_ratio':viral_misinformation_ratio,
                             #'credibility':credibility,
                             'viral_messages':viral_texts,
                            })    
    return df_users

def get_top_users(df,top=10):
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

def get_test_metrics(y_test, y_pred, y_prob = [], full_metrics = False, print_charts = True):
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
