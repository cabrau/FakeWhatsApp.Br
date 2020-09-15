#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:36:44 2020

@author: cabrau
"""
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import time

def getTestMetrics(y_test, y_pred, y_prob = []):
    '''
    Plot charts e print performance metrics
    Input: predictions and labels
    Output: charts and metrics
    '''
    print(metrics.classification_report(y_test, y_pred))
    
    f1 = metrics.f1_score(y_test, y_pred, pos_label = 1, average = 'binary') #macro
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, pos_label = 1, average = 'binary')
    recall = metrics.recall_score(y_test, y_pred, pos_label = 1, average = 'binary')
    
    #confusion matrix
    cf_matrix = metrics.confusion_matrix(y_test, y_pred)
    group_names = ['VN','FP','FN','VP']
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
        #plt.figure()
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
    return acc, precision, recall, f1, roc_auc

#MLP shallow para classificação binária
class Shallow_MLP():
    
    def __init__(self,Nx,Nh,seed = 42):
        ''' 
        Inicializa a rede com o número Nx neurônios de entrada (número de atributos), 
        número Nh de neurônios da camada oculta 
        '''
        np.random.seed(seed)
        
        self.W = np.random.randn(Nh,Nx)*np.sqrt(1/Nx) 
        b_h = np.ones((Nh,1))*0.01
        self.W = np.concatenate((self.W,b_h),axis=1)
        
        self.M = np.random.randn(1,Nh)*np.sqrt(1/Nx)
        b_o = np.zeros((1,1))
        self.M = np.concatenate((self.M,b_o),axis=1)

        
    #funções de ativação e derivadas
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def d_sigmoid(self,z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def relu(self,z):
        return np.maximum(0,z)
    
    def d_relu(self,z):
        return np.where(z >= 0, 1, 0)
    
    def feedfoward(self,x):
        #coloca o 1 do bias na última coluna
        x = x.reshape(1,x.shape[0])
        
        x = np.concatenate((x,np.ones((1,1))),axis=1)
        #soma ponderada na camada oculta
        u = self.W @ x.T
        #saída da função de ativação da camada oculta
        z = self.relu(u)
        z = np.concatenate((z,np.ones((1,1))),axis=0)        
        #soma ponderada na camada de saída
        r = self.M @ z
        #saída da função de ativação da camada de saída
        o = self.sigmoid(r)
        return o
    
    def predict(self,X):
        y_pred = []
        y_prob = []
        for x in X:
            prob = self.feedfoward(x)[0][0]
            if prob > 0.5:
                y_hat = 1
            else:
                y_hat = 0                
            y_pred.append(y_hat)
            y_prob.append(prob)
            
        return y_pred, y_prob
    
    #acurácia
    def score(self,X,y):
        y_pred, prob = self.predict(X)
        return float(sum(y_pred == y)) / float(len(y))  
                
    def backpropagation(self,x,y,alpha):
        #feedfoward
        x = x.reshape(1,x.shape[0])
        x = np.concatenate((x,np.ones((1,1))),axis=1)
        #soma ponderada na camada oculta
        u = self.W @ x.T
        #saída da função de ativação da camada oculta
        z = self.relu(u)
        z = np.concatenate((z,np.ones((1,1))),axis=0)        
        #soma ponderada na camada de saída
        r = self.M @ z
        #saída da função de ativação da camada de saída
        o = self.sigmoid(r)       
        
        #volta o erro e atualiza os pesos
        error = y-o
        delta = error * self.d_sigmoid(r)
        zeta = self.d_relu(u) * (self.M[:,0:-1].T @ delta)
        delta_M = delta @ z.T
        delta_W = zeta @ x
        return delta_M, delta_W
        
    def minibatch_backpropagation(self,X,y,alpha,lambda_l2):
        delta_M = 0
        delta_W = 0
        
        for i,x in enumerate(X):
            dM, dW = self.backpropagation(x,y[i],alpha)
            delta_M += dM
            delta_W = dW
        
        delta_M = delta_M/len(X)
        delta_W = delta_W/len(X)
        
        self.updateWeights(delta_M,delta_W,alpha,lambda_l2)
        
        
    def updateWeights(self, delta_M, delta_W, alpha, lambda_l2):
        #atualização dos pesos com regularização L2
        self.M = self.M + alpha * delta_M - alpha*lambda_l2*self.M 
        self.W = self.W + alpha * delta_W - alpha*lambda_l2*delta_W
        
    #SGD com minibatch
    def fit(self, X, y, alpha, epochs, minibatch_size, validation = False, lambda_l2 = 0, X_val = None, y_val = None):
        
        X = X.toarray()
        
        loss = []
        val_loss = []
        epoch_i = 0
        first_run = True
        
        while epoch_i < epochs:
            
            if first_run:
                t0 = time.time()
            

            #shuffle
            indices = np.random.permutation(X.shape[0])
            X = X[indices]
            y = y[indices]
            
            #minibatch
            for i in range(0, X.shape[0], minibatch_size):

                y_pred, y_prob = self.predict(X)
                loss_i = metrics.log_loss(y,y_prob)
                loss.append(loss_i)
                
                if validation == True:
                    y_pred, y_prob = self.predict(X_val)
                    loss_i = metrics.log_loss(y_val,y_prob)
                    val_loss.append(loss_i)                    
                
                X_train_mini = X[i:i + minibatch_size]
                y_train_mini = y[i:i + minibatch_size]
                self.minibatch_backpropagation(X_train_mini, y_train_mini, alpha, lambda_l2)
                
                epoch_i += 1
                
                if first_run:
                    t1 = time.time()
                    t = t1-t0
                    t = t/60
                    total = t*epochs
                    print('Tempo de execução da primeira época (min): %0.2f' % t)
                    print('Previsão de tempo total de execução: %0.2f' % total)
                    first_run = False
                    
                if epoch_i > epochs:
                    break
                    
        if validation == True:
            return loss, val_loss
        else:
            return loss
        

class KNNClassifier():
    #inicialize com dados de treino, saídas desejadas, quantidade K de vizinhos e a métrica de distância
    #opções de métricas: euclidean, mahalanobis 
    def __init__(self,X_train,y_train, K , dist_metric = 'euclidean'):
        
        self.X_train = X_train
        self.y_train = y_train
        self.K = K   
        self.dist_metric = dist_metric
        
        if dist_metric == 'mahalanobis':
            cov = np.cov(X_train)
            self.inv_covmat = sp.linalg.inv(cov)
        
    def euclidian(self, a, b):
        return np.sqrt(np.sum((a - b)**2, axis=1))
    
    def manhattan(self, a, b):
        return np.sqrt(np.sum((a - b), axis=1))
    
    def chebyshev(self, a, b):
        return np.max(np.absolute(a - b), axis=1)
    
    def mahalanobis(self, a, b):        
        left_term = np.dot((a-b).T, self.inv_covmat)
        right_term = (a - b)
        mahal = np.dot(left_term,right_term)
        mahal = np.sqrt(mahal)
        return mahal

        
    def get_distance(self, a, b):
        if self.dist_metric == 'euclidean':
            return self.euclidian(a,b)
        
        elif self.dist_metric == 'mahalanobis':
            return self.mahalanobis(a,b)
        
        elif self.dist_metric == 'chebyshev':
            return self.chebyshev(a,b)
        
        elif self.dist_metric == 'manhattan':
            return self.manhattan(a,b)
        
    def kneighbors(self, X_test, return_distance=False):

        dist = []
        neigh_ind = []
        
        distances = [self.get_distance(x_test, self.X_train) for x_test in X_test]

        for row in distances:
            #enumera distancias
            enum_neigh = enumerate(row)
            #ordena distâncias e seleciona as K menores
            sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:self.K]
            #seleciona os índices dos K vizinhos mais próximos
            ind_list = [tup[0] for tup in sorted_neigh]
            #seleciona as distâncias dos K vizinhos mais próximos
            dist_list = [tup[1] for tup in sorted_neigh]

            dist.append(dist_list)
            neigh_ind.append(ind_list)

        #retorna as as distâncias para debug e os índices
        if return_distance:
            return np.array(dist), np.array(neigh_ind)
        
        #retorna apenas os índices
        return np.array(neigh_ind)
        
    
    def predict(self, X_test):        
        neighbors = self.kneighbors(X_test)
        #alteração para prever um valor contínuo (dividindo a quantidade de vizinhos de cada classe pelo K)
        #y_pred = np.array([np.argmax(np.bincount(y_train[neighbor].astype(int))) for neighbor in neighbors])
        prob_classes = [np.bincount(self.y_train[neighbor].astype(int))/self.K for neighbor in neighbors]
        y_pred = []
        prob_pos = []
        for bc in prob_classes:
            y_pred.append(np.argmax(bc))
            prob_pos.append(bc[0])
        prob_pos = [1 - p for p in prob_pos]
        return y_pred, prob_pos
    
    def score(self, X_test, y_test):
        y_pred, prob = self.predict(X_test)
        return float(sum(y_pred == y_test)) / float(len(y_test))