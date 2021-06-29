# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:42:43 2019

@author: atjeehad
"""
import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from numpy import ndarray
from sklearn.metrics import accuracy_score


data=np.genfromtxt("perceptron_data.csv",delimiter=",")

x=data[:,:-1]
y=data[:,data.shape[1]-1]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)
abc=xTrain.shape[1]
#print(abc,xTrain.shape)
XC=abc
#a= np.ndarray((1,abc+1),float)
a=np.random.rand(x.shape[1]+1,1)
#print(a.shape)
#print(a.T.shape)


class PerceptronN:
    Max_iteration=500
   # w=np.random.rand(n+1,1)
    def fit(self,x,y):
        n=x.shape[1]
        Max_iteration=500
        w=np.random.rand(n+1,1)
        #print(y)
        for i in range(Max_iteration):
            missclasified=[]
            delta=[]
            #wTx=0
            featurexallall=np.zeros((np.size(x,1)+1,1))
            #ew_X = x[i].tolist()
            #ew_X.append(1)
            for j in range( x.shape[0] ):
                for k in range(np.size(x,1)+1):
                    if k==60:
                        featurexallall[k]=1
                    elif k<=59:
                        featurexallall[k]=x[j][k]
                #print( featurexallall[59] )
                wTx=np.matmul(w.T,featurexallall)

                #new_w = w.T
                #xxy = np.dot( new_w, featurexallall )
                #print(xxy)
                xxy=wTx[0,0]
                #print(xxy)
                if xxy >= 0 and y[j] != 0:
                    #print("In If")
                    missclasified.append(j)
                    delta.append(-1)
                elif xxy < 0 and y[j]==1:

                    #print("In elIf")
                    missclasified.append(j)
                    delta.append(1)
            for d in range (len(missclasified)):
                w=w+(delta[d]*x[missclasified[d]])
        return w
       
    def predict(self,x,w):
        prediction=np.zeros(len(x))
        featureallx=np.zeros((np.size(x,1)+1,1))
        #print(3)
        for i in range(len(x)):
            #print(i)
            wTx=0
            new_X = x.tolist()
            new_X.append(1)
            for k in range(np.size(x,1)+1):
                #print(5)
                if k==60:
                    featureallx[k]=1
                    #print(6)
                elif k<=59:
                    featureallx[k]=x[i][k]
            wTx=np.matmul(w.T,featureallx)
           # print("wtx:",wTx)
           # print(featureallx.shape)
            xxy=wTx[0,0]
            if xxy>=0:
                prediction[i]=1
            elif xxy<=0:
                #print(0)
                prediction[i]=0
        return prediction
                    
                
                        
pre=PerceptronN()
w=pre.fit(xTrain,yTrain)
#print(w)
pred=pre.predict(xTest,w)
#print(np.size(x,1))
#print(len(xTest))
print(pred)
#print(yTest)
acfun=accuracy_score(yTest,pred)*100
#print(acfun)                    
        
    
            
                        
                        
                    
                
                
            
            
        
        
        
        
        