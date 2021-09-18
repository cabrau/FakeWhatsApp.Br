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
from gensim.models import Word2Vec

#%% 
# user identification
def get_degree_and_strenght(df_user, df, user):
    user_groups = df_user['group'].unique()
    degree = 0
    strenght = 0
    
    for g in user_groups:
        messages_in_g = len(df_user[df_user['group']==g])
        degree_in_g = len(df[df['group']==g]['id'].unique())
        strenght_in_g = degree_in_g*messages_in_g
        
        degree += degree_in_g
        strenght += strenght_in_g
    
    return degree, strenght

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
    repeated_messages = []
    repeated_messages_ratio = []
    
    # temporal features
    days_active = []
    daily_mean = []
    daily_std = []
    daily_median = []
    daily_95 = []
    daily_outliers = []
    daily_max = []
    
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
        n_groups.append(df_user['group'].nunique())
        #ratios
        viral_ratio.append(n_virals[-1]/n_messages[-1])
        midia_ratio.append(n_midia[-1]/n_messages[-1])
        text_ratio.append(n_text[-1]/n_messages[-1])
        
        #repeated messages
        originals = len(df_user[df_user['midia']==0].drop_duplicates(subset='text'))
        duplicates = len(df_user[df_user['midia']==0]) - originals
        repeated_messages.append(duplicates)
        if n_text[-1] > 0:
            repeated_messages_ratio.append(duplicates/n_text[-1])
        else:
            repeated_messages_ratio.append(0)
        
        #labelled features
        n_misinformation.append(len(df_user[df_user['misinformation']==1]))
        message_misinformation_ratio.append(n_misinformation[-1]/n_messages[-1])
        if n_virals[-1] > 0:
            viral_misinformation_ratio.append(n_misinformation[-1]/n_virals[-1])
        else:
            viral_misinformation_ratio.append(0)
            
        # temporal features
        frame = '24H'
        messages_by_day = df_user.groupby('timestamp').count()['id'].resample(frame).sum()        
        days_active.append(messages_by_day.count())
        daily_mean.append(messages_by_day.mean())
        daily_std.append(messages_by_day.std())
        daily_median.append(messages_by_day.median())
        daily_95.append(messages_by_day.quantile(0.95))
        daily_max.append(messages_by_day.max())
        daily_outliers.append((messages_by_day > daily_95[-1]).sum())
        
        #graph features
        degree, strenght = get_degree_and_strenght(df_user, df, user)
        v_degree, v_strenght = get_degree_and_strenght(df_user[df_user['viral']==1], df, user)
        m_degree, m_strenght = get_degree_and_strenght(df_user[df_user['misinformation']==1], df, user)
        
        messages_degree_centrality.append(degree)
        messages_strenght.append(strenght)
        
        viral_degree_centrality.append(v_degree)
        viral_strenght.append(v_strenght)
        
        mis_degree_centrality.append(m_degree)
        mis_strenght.append(m_strenght)    
        
           
    df_users = pd.DataFrame({'id':users.index, 'groups':n_groups, 'number_of_messages':n_messages,
                            'texts':n_text,'text_ratio':text_ratio,
                             'midia':n_midia,'midia_ratio':midia_ratio,
                             'virals':n_virals,'viral_ratio':viral_ratio,
                             'repeated_messages':repeated_messages, 
                             'repeated_messages_ratio':repeated_messages_ratio,
                             'days_active':days_active, 'daily_mean':daily_mean,
                             'daily_std':daily_std, 'daily_median': daily_median,
                             'daily_95':daily_95, 'daily_outliers':daily_outliers,
                             'daily_max':daily_max,
                             'degree_centrality':messages_degree_centrality,
                             'strenght':messages_strenght,
                             'viral_degree_centrality':viral_degree_centrality,
                             'viral_strenght':viral_strenght,
                             'misinformation':n_misinformation,
                             'misinformation_degree_centrality':mis_degree_centrality,
                             'misinformation_strenght':mis_strenght,
                             'misinformation_ratio': message_misinformation_ratio,
                             'viral_misinformation_ratio':viral_misinformation_ratio
                            })    
    return df_users

def get_top_users(df,top=10):
    groupedByid = df.groupby(['id']).count()
    groupedByid = groupedByid.sort_values('date', ascending=False)[0:top]['date']
    return get_user_features(df,groupedByid)

#%%
# text preprocessing 


#stop words removal
stop_words = list(stopwords.words('portuguese'))
new_stopwords = ['aí','pra','vão','vou','onde','lá','aqui',
                 'tá','pode','pois','so','deu','agora','todo',
                 'nao','ja','vc', 'bom', 'ai','ta', 'voce', 'alguem', 'ne', 'pq',
                 'cara','to','mim','la','vcs','tbm', 'tudo']
stop_words = stop_words #+ new_stopwords


def process_emojis(text):
    demoji = emoji.demojize(text, language = 'pt')
    demoji = demoji.replace(':',' ')
    for w in demoji.split():
        if '_' in w:
            prefix = w.split('_')[0:2]
            prefix = '_'.join(prefix)
            demoji = demoji.replace(w, prefix)
    return demoji

def process_punctuation(text):
    # important punctuation
    punct = ['*', '_', '!', "\n", '?']
    #get all unique chars
    chars = set(text)
    #for each unique char in text, do:
    for c in chars:
        if c in punct:
            text = text.replace(c, ' ' + c + ' ')
        elif c in string.punctuation:
            text = text.replace(c, ' ')
    text 
    return text    
    

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

def repeated_chars(text):
    chars = set(text)
    for c in chars:
        if c.isalpha():
            regex = c + c + '+'
            text = re.sub(regex, c+c, text)
    return text

def firstSentence(text):
    list_s = sent_split(text)
    for s in list_s:
        if s is not None:
            return s

def sent_split(text):
    list_s = re.split('\. |\! |\? |\n',text)
    return list_s


def preprocess(text, lemma = True):
            
    # normalization    
    text = text.lower().strip()    
    text = domainUrl(text)
    text = processLoL(text)
    text = repeated_chars(text)
    text = remove_numbers(text)
    # separet punctuation and emojis
    text = process_punctuation(text) 
    text = process_emojis(text)    
    # remove stopwords
    text = removeStopwords(text)    
    # remove exceeding spaces
    text = re.sub(' +', ' ', text)    
    text = text.strip()
    
    if lemma:
        text = lemmatization(text)
    
    # remove accents
    for word in set(text.split()):        
             if not word.startswith(':'):            
                 text = text.replace(word,unidecode(word))
        
    return text     

# def preprocess(text, semi=False, rpunct = True, remoji = False, lemma = True, stem = False):
            
#     text = text.lower().strip()    
#     text = domainUrl(text)
#     text = processLoL(text)
#     text = repeated_chars(text)
#     text = remove_numbers(text)    
#     text = processEmojisPunctuation(text,remove_punct = rpunct, remove_emoji=remoji)    
    
#     # remove punctuation
#     if not remoji:
#         for word in set(text.split()):        
#             if word not in emojis_list:            
#                 text = text.replace(word,unidecode(word))          
            
    
    
#     if semi:        
#         return text
#     # remove stopwords
#     text = removeStopwords(text)
#     # remove exceeding spaces
#     text = re.sub(' +', ' ', text)    
#     text = text.strip()
    
#     if stem:
#         text = stemming(text)
    
#     if lemma:
#         text = lemmatization(text)
        
#     return text 
 
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

def optimal_threshold(prob,y):
    best_thresold = 0
    best_score = 0
    
    for i in range(100):
        threshold = i/100
        y_pred = [1 if p >= threshold else 0 for p in prob]
        score = metrics.accuracy_score(y,y_pred)
        
        if score > best_score:
            #print(score)
            best_thresold = threshold
            best_score = score
            
    return best_thresold


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
        false_positive_rate = 1 - recall_neg
    
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
        results = (acc, false_positive_rate, precision, recall, f1, roc_auc)        
        for m in results:
            m = str(m).replace('.',',')[0:5]
            print(m)
    return results

#%%

# optization
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def random_layers():
    '''
    Create a tuple random of hidden_layer_sizes. 
    '''
    n_layers = np.random.randint(1,4)
    layers_list = []
    for i in range(n_layers):            
        hidden_neurons = np.random.randint(1,15)*25
        layers_list.append(hidden_neurons)
    layers_tuple = tuple(layers_list)
    return layers_tuple

def optimized_mlp(hl,bs,al,lri):
    clf = MLPClassifier(activation = 'relu', solver = 'adam',random_state = 42, 
                   tol = 1e-3, verbose = False, early_stopping = True, 
                   n_iter_no_change = 3, max_iter = 100,
                   hidden_layer_sizes = hl, alpha = al, 
                   learning_rate_init = lri, batch_size = bs)
    return clf

def random_search_mlp(X_train,y_train,n_iter=10):
    
    # hyperparams to optimize
    hidden_layers = []
    alphas = []
    batch_sizes = []
    learning_rate_inits = []
    # sample
    np.random.seed(0)
    for i in range(n_iter):   
        hl = random_layers()
        #print(hl,end = '; ')
        hidden_layers.append(hl)
        ap = 10**np.random.uniform(-6,-2)
        #print(ap, end = '; ')
        alphas.append(ap)
        learning = 10**np.random.uniform(-4,-1)
        #print(learning, end = '; ')
        learning_rate_inits.append(learning)
        batch = np.random.randint(1,7)*50 #math.floor(10**np.random.uniform(1.5,2.6)) #np.random.randint(2,30)*10
        #print(batch)
        batch_sizes.append(batch)

    # tunning
    X_train_v, X_val, y_train_v, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    best_score = 0
    i = 0
    for hl,bs,al,lri in zip(hidden_layers,batch_sizes,alphas,learning_rate_inits):
        
        clf = optimized_mlp(hl,bs,al,lri)

        print(i, end= ' ')
        i+=1
        print()
        print('hidden layers: {a}; alpha: {b:.5f}; learning rate: {c:.5f}; batch: {d}'.format(a=hl,b=al,c=lri,d=bs))    
        clf.fit(X_train_v, y_train_v)
        y_pred = clf.predict(X_val)        
        y_prob = clf.predict_proba(X_val)[:,1]
        
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_prob, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        roc_auc = metrics.accuracy_score(y_val, y_pred)
        
        print('ACC: {a:.3f}'.format(a=roc_auc))       
        if roc_auc > best_score:
            best_score = roc_auc
            best_params = (hl,bs,al,lri)

        #print('validation rmse: {a:.3f}'.format(a=rmse))


    hl,bs,al,lri = best_params
    print()
    print('--------------------')
    print('BEST PARAMETERS (validation AUC = {a:.3f})'.format(a=best_score))
    print('hidden layers: {a}; alpha: {b:.5f}; learning rate: {c:.5f}; batch: {d}'.format(a=hl,b=al,c=lri,d=bs))
    print('--------------------')
    return hl,bs,al,lri

#%%

# word embeddings

def vectorize_text(model, text, method='mean'):
        """
        Convert all words in a text to their embedding vector
        and calculate a vector for the text, by the mean or the sum of word vectors
        Parameters
        ----------
        text: str
        Text from wich the word vector's will be calculated    
        
        method: str
        'mean' or 'sum'
            
        Returns
        -------
        vec: numpy.ndarray 
        Array of the word embeddings from the given text 
        """
        n = model.wv.vector_size
        X = np.empty(shape=[0, n])
        words = text.split()
        for word in words:
            try:
                vec = model.wv[word]
                X = np.append(X,[vec], axis = 0)
            # if oov:    
            except:
                #print('word not in vocabulary: ', word)
                continue
        if X.size == 0:
            vec = np.zeros(n)
        elif method == 'mean':
            vec = np.mean(X,axis=0)
        elif method == 'sum':
            vec = np.sum(X,axis=0)
        return vec
    
def vectorize_corpus(model, corpus, method='mean'):
        """
        Convert all texts in a corpus to vectors
        Parameters
        ----------
        corpus: list
        List of texts    
        
        method: str
        'mean' or 'sum'
            
        Returns
        -------
        X: numpy.ndarray 
        2D Array of vectors from each text in corpus
        """
        X = [vectorize_text(model, text, method=method) for text in corpus]
        X = np.concatenate(X, axis=0).reshape(len(X),len(X[0]))
        return X


#%%
# data processing

def z_score(X_train,X_test):
    '''
    Normalization z    

    Parameters
    ----------
    X_train : numpy
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train : numpy
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.

    '''
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu ) / sigma
    X_test = (X_test - mu ) / sigma
    return X_train,X_test