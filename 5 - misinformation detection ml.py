#!/usr/bin/env python
# coding: utf-8

# # Experiments with textual misinformation detection using machine learning models
# In this experiment, we train and test machine learning models for detecting misinformation based purely on the text. We evaluated the performance of 5 classification models with wide use in text classification problems.
# 
# **Models:**
# * Logistic regression
# * Bernoulli Naive-Bayes
# * Multinomial Naive-Bayes
# * Linear SVM
# * KNN
# * Random forest
# * Gradient Boosting
# * Multilayer Perceptron
# 
# # NOTES TO MYSELF
# 
# **To do:**
# * linguistic features
# * hyperparamater tunning; validation
# 
# 
# **References:**
# * https://towardsdatascience.com/text-classification-in-python-dd95d264c802
# * https://github.com/miguelfzafra/Latest-News-Classifier/tree/master/0.%20Latest%20News%20Classifier/04.%20Model%20Training
# 

# In[1]:


#utils
import emoji, re, string, time, os
from utils import getTestMetrics
import pandas as pd
import numpy as np
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

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="darkgrid")

start_time = time.time()


# ## Experiments
# 
# ## Subsets:
# * Viral
# * All
# 
# ### Feature engineering
# * bow: bag of words
# * tfidf: term frequency–inverse document frequency
# * max_features: limits to 500
# 
# ### Pre-processing
# * processed: convert url in just the domain, separate emojis, remove punctuation, downcase, lemmatization, remove stop words
# 
# ### Data balancing
# * smote: Synthetic Minority Oversampling Technique
# * undersampling: random undersampling
# * random_oversampling
# 
# 1. **ml-bow**<br> 
# One of the best so far. Best F1: 0.6429 random forest<br>
# 
# * **ml-tfidf** <br>
# Comparable with ml-bow. Best F1: 0.6278 random forest<br>
# 
# * **ml-tfidf-processed**<br>
# Comparable with ml-bow and ml-tfidf. The pre-processing didn't had great impact. Precision slightly better.
# Best F1: 0.6184 random forest<br>
# 
# * **ml-bow-processed**<br>
# Same as ml-tfidf-processed. Best F1: 0.6254 random forest  <br>
# 
# * **ml-tfidf-smote**<br>
# Good result. Smote improves F1 when using tf-idf. Best F1: 0.6426 random forest<br>
# 
# * **ml-tfidf-undersampling** <br>
# Very poor approach. Undersampling is a bad ideia. Increases recall but greatly reduces accuracy. Best F1: 0.3570 bernoulli naive-bayes<br>
# 
# * **ml-tfidf-processed-smote**<br>
# Good result. Best F1: 0.6470 random forest <br>
# 
# * **ml-bow-processed-smote**<br>
# Poor. Smote does not goes well with bow. Best F1: 0.4037 bernoulli naive-bayes<br>
# 
# * **ml-bow-random_oversampling**<br>
# ~~Holy shit!!! This approuch overcame all the expectations! Best F1: 0.9658 random forest~~<br>
# *Edit:* well, when things look too good to be true, they probably are. The good results were just a case of data leaking.
# 
# * **ml-tfidf-random_oversampling**<br>
# Similar to *ml-bow-random_oversampling*
# 
# * **ml-bow-processed-random_oversampling**<br>
# Similar to *ml-bow-random_oversampling*
# 
# * **ml-tfidf-processed-random_oversampling**<br>
# Similar to *ml-bow-random_oversampling*
# 
# * **ml-bow-processed-random_oversampling-max_features**<br>
# Poor.
# 
# * **ml-tfidf-processed-random_oversampling-max_features**<br>
# Poor.
# 
# 
# ### Conclusions:
# * Smote is not the best oversampling technique for text, specially for BOW features
# * Pre-processing the data didn't appear to have a great impact, despite the great reduction of dimentionality
# * BOW features are comparably to TF-IDF features
# * Random oversampling is a good oversampling technique
# * Use a maximum number of features wasn't a good approach

# In[2]:


base = '2018'
subset = 'viral'
path_dir = 'results/' + str(base) + '/' + subset + '/ml/'
path_dir


# In[3]:


# best results analysis
df_best = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1 score', 'auc score','vocab'])
#iterates over files
exp = []
for filename in os.listdir(path_dir):
    exp.append(str(filename).replace('.csv',''))
    file_path = path_dir + filename
    #print(filename)
    df_temp = pd.read_csv(file_path)
    best_ix = df_temp['f1 score'].argmax() #f1 score
    best = df_temp.iloc[best_ix]
    df_best = df_best.append(best)
df_best['experiment'] = exp
cols = df_best.columns.tolist()
cols = cols[-2:] + cols[:-2]
df_best = df_best[cols]
df_best = df_best.reset_index()
df_best = df_best.drop(columns = ['index'])
df_best = df_best.sort_values(by='f1 score',ascending=False)
df_best.style.background_gradient(cmap='Blues')


# # Begin experiment
experiments =  ['ml-bow-random_oversampling',
 'ml-bow-processed-random_oversampling',
 'ml-tfidf-random_oversampling',
 'ml-tfidf-processed-random_oversampling',
 'ml-tfidf-processed-smote',
 'ml-bow',
 'ml-tfidf-smote',
 'ml-tfidf',
 'ml-bow-processed',
 'ml-tfidf-processed',
 'ml-tfidf-random_oversampling-max_features',
 'ml-bow-random_oversampling-max_features',
 'ml-tfidf-processed-random_oversampling-max_features',
 'ml-bow-processed-random_oversampling-max_features',
 'ml-bow-processed-smote',
 'ml-tfidf-undersampling']


# In[4]:
for experiment in experiments:
    
    pre_processed = True # the texts were already pre-processed
    filepath = 'data/' + str(base) + '/fakeWhatsApp.BR_' + str(base) + '.csv'
    df = pd.read_csv(filepath)
    
    if subset == 'viral':
        df = df[df['viral']==1]
        
    df.head(5)
    
    
    # # Corpus statistics
    
    # In[5]:
    
    
    df.describe()[['characters','words','sharings']]
    
    
    # In[6]:
    
    
    texts = df[df['midia']==0]['text']
    y = df[df['midia']==0]['misinformation']
    
    
    # In[7]:
    
    
    print('total data')
    pos_mask = y == 1 
    pos = y[pos_mask]
    neg_mask = y == 0 
    neg = y[neg_mask]
    values = [pos.shape[0],neg.shape[0]]
    keys = ['misinformation', 'non-misinformation']
    g = sns.barplot(x = keys, y = values)
    for p in g.patches:
        g.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', 
                   va = 'center', xytext = (0, 5), textcoords = 'offset points')
    
    
    # In[8]:
    
    
    #removing duplicates
        
    df = df.drop_duplicates(subset=['text'])    
    texts = df[df['midia']==0]['text']
    y = df[df['midia']==0]['misinformation']
    
    print('data after remove duplicates')
    pos_mask = y == 1 
    pos = y[pos_mask]
    neg_mask = y == 0 
    neg = y[neg_mask]
    values = [pos.shape[0],neg.shape[0]]
    keys = ['misinformation', 'non-misinformation']
    g = sns.barplot(x = keys, y = values)
    for p in g.patches:
        g.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', 
                   va = 'center', xytext = (0, 5), textcoords = 'offset points')
    
    
    # In[9]:
    
    
    print(len(texts))
    print(len(y))
    
    
    # # Pre-processing
    # * convert url in just the domain
    # * separate emojis
    # * punctuation
    
    # [Some suggestions in this work](https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/03.%20Feature%20Engineering/03.%20Feature%20Engineering.ipynb)
    # 
    # * **Special character cleaning**
    # 
    # * **Upcase/downcase**
    # 
    # * **Punctuation signs** 
    # 
    # * **Possessive pronouns**
    # 
    # * **Stemming or Lemmatization**
    # 
    # * **Stop words**
    
    # In[10]:
    
    
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
               
    
    
    # In[11]:
    
    
    #if experiment is with pre-processed text
    if 'processed' in experiment:
            #text was already pre-processed
            if pre_processed:
                if subset != 'viral':
                    pro_texts = pickle.load(open( "data/2018/processed_texts.p", "rb" ))
                else:
                    pro_texts = pickle.load(open( "data/2018/processed_texts-viral.p", "rb" ))
            else:
                pro_texts = [preprocess(t) for t in texts]
                if subset != 'viral':
                    pickle.dump(pro_texts, open( "data/2018/processed_texts.p", "wb" ))
                else:
                    pickle.dump(pro_texts, open( "data/2018/processed_texts-viral.p", "wb" ))
    else:
        pro_texts = [t for t in texts]
    
    
    # In[12]:
    
    
    list(zip(pro_texts[0:10], texts[0:10]))
    
    
    # In[13]:
    
    
    print(len(pro_texts))
    print(len(y))
    
    
    # ## Train-test split
    
    # In[14]:
    
    
    #random state = 42 for reprudictibility
    texts_train, texts_test, y_train, y_test = train_test_split(pro_texts, y, test_size=0.2, 
                                                                        stratify = y, random_state=42)
    
    full_texts_train, full_texts_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, 
                                                                        stratify = y, random_state=42)
    
    
    # ## Vectorization
    
    # In[15]:
    
    
    max_feat = 500
    
    if 'tfidf' in experiment:
        if 'max_features' in experiment:
            vectorizer = TfidfVectorizer(max_features = max_feat) 
        else:
            vectorizer = TfidfVectorizer()
            
    elif 'bow' in experiment:
        if 'max_features' in experiment:
            vectorizer = CountVectorizer(max_features = max_feat, binary=True) 
        else:
            vectorizer = CountVectorizer(binary=True)
    
    vectorizer.fit(texts_train)   
    X_train = vectorizer.transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    X = vectorizer.transform(pro_texts)
    
    
    # ## SVD visualization
    
    # In[16]:
    
    
    n_components = 2
    title = "SVD decomposition"
    # Creation of the model
    mod = TruncatedSVD(n_components=n_components)
    # Fit and transform the features
    principal_components = mod.fit_transform(X)
    
    # Put them into a dataframe
    df_features = pd.DataFrame(data=principal_components,
                     columns=['1', '2'])
    
    df_features['label'] = y
    
    # Plot
    plt.figure(figsize=(10,10))
    sns.scatterplot(x='1',
                    y='2',
                    hue="label", 
                    data=df_features,
                    alpha=.8).set_title(title);
    
    
    # ## Data balancing
    
    # In[17]:
    
    
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
    
    
    # In[18]:
    
    
    vocab_size = X_train.shape[1]
    vocab_size
    
    
    # ## Metrics
    
    # In[19]:
    
    
    scenario = []
    model = []
    accuracy_score = []
    precision_score = []
    recall_score = []
    f1_score = []
    auc_score = []
    
    
    # ## Models training and test
    
    # In[20]:
    
    
    print('Logistic Regression')
    logreg = LogisticRegression().fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:,1]
    acc, precision, recall, f1, roc_auc = getTestMetrics(y_test, y_pred, y_prob)
    
    model.append('logistic regression')
    accuracy_score.append(acc)
    precision_score.append(precision)
    recall_score.append(recall)
    f1_score.append(f1)
    auc_score.append(roc_auc)
    
    
    # In[21]:
    
    
    print('Bernoulli Naive-Bayes')
    bnb = BernoulliNB().fit(X_train, y_train)
    y_pred = bnb.predict(X_test)
    y_prob = bnb.predict_proba(X_test)[:,1]
    acc, precision, recall, f1, roc_auc = getTestMetrics(y_test, y_pred, y_prob)
    
    model.append('bernoulli naive-bayes')
    accuracy_score.append(acc)
    precision_score.append(precision)
    recall_score.append(recall)
    f1_score.append(f1)
    auc_score.append(roc_auc)
    
    
    # In[22]:
    
    
    print('Multinomial Naive-Bayes')
    mnb = MultinomialNB().fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    y_prob = mnb.predict_proba(X_test)[:,1]
    acc, precision, recall, f1, roc_auc = getTestMetrics(y_test, y_pred, y_prob)
    
    model.append('multinomial naive-bayes')
    accuracy_score.append(acc)
    precision_score.append(precision)
    recall_score.append(recall)
    f1_score.append(f1)
    auc_score.append(roc_auc)
    
    
    # In[23]:
    
    
    print('Linear Support Vector Machine')
    svm = LinearSVC().fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    #y_prob = svm.predict_proba(X_test)[:,1]
    acc, precision, recall, f1, roc_auc = getTestMetrics(y_test, y_pred)
    
    model.append('linear svm')
    accuracy_score.append(acc)
    precision_score.append(precision)
    recall_score.append(recall)
    f1_score.append(f1)
    auc_score.append(roc_auc)
    
    
    # In[24]:
    
    
    print('KNN')
    rf = KNeighborsClassifier().fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:,1]
    acc, precision, recall, f1, roc_auc = getTestMetrics(y_test, y_pred, y_prob)
    
    model.append('knn')
    accuracy_score.append(acc)
    precision_score.append(precision)
    recall_score.append(recall)
    f1_score.append(f1)
    auc_score.append(roc_auc)
    
    
    # In[25]:
    
    
    print('Random Forest')
    rf = RandomForestClassifier().fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:,1]
    acc, precision, recall, f1, roc_auc = getTestMetrics(y_test, y_pred, y_prob)
    
    model.append('random forest')
    accuracy_score.append(acc)
    precision_score.append(precision)
    recall_score.append(recall)
    f1_score.append(f1)
    auc_score.append(roc_auc)
    
    
    # In[26]:
    
    
    print('Gradient Boosting')
    gb = GradientBoostingClassifier().fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    y_prob = gb.predict_proba(X_test)[:,1]
    acc, precision, recall, f1, roc_auc = getTestMetrics(y_test, y_pred, y_prob)
    
    model.append('gradient boosting')
    accuracy_score.append(acc)
    precision_score.append(precision)
    recall_score.append(recall)
    f1_score.append(f1)
    auc_score.append(roc_auc)
    
    
    # In[38]:
    
    
    print('Multilayer perceptron')
    mlp = MLPClassifier(max_iter=10,verbose=True).fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:,1]
    acc, precision, recall, f1, roc_auc = getTestMetrics(y_test, y_pred, y_prob)
    
    model.append('mlp')
    accuracy_score.append(acc)
    precision_score.append(precision)
    recall_score.append(recall)
    f1_score.append(f1)
    auc_score.append(roc_auc)
    
    
    # ## Results
    
    # In[28]:
    
    
    end_time = time.time()
    ellapsed_time = end_time - start_time
    print('ellapsed time (min):', ellapsed_time/60)
    
    
    # In[29]:
    
    
    df_metrics = pd.DataFrame({'model':model,
                                     'accuracy':accuracy_score,
                                     'precision': precision_score,
                                     'recall': recall_score,
                                     'f1 score': f1_score,
                                     'auc score': auc_score})
    
    df_metrics['vocab'] = [vocab_size]*len(df_metrics)
    df_metrics
    
    
    # In[30]:
    
    
    filepath = 'results/' + base + '/' + subset + '/ml/' + experiment + '.csv'
    filepath
    
    
    # In[31]:
    
    
    df_metrics.to_csv(filepath, index = False)