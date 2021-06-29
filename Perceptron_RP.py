from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import numpy as np
import collections
import math as m
import random as rnd
from sklearn import metrics


class Perceptrons:
    def __init__(self, max_iteration):
        self.max_iteration = max_iteration

   
    


    def train_data_set(self, X_train, Y_train):
        new_w = self.w.tolist()
        
        #missmatch_list = list

        #new_x = list
         
        
        for i in range(self.max_iteration):
            
            missmatch_list = []
            jog_biyog_select = []
            

            for i in range( X_train.shape[0] ):

                #new_X_train = fix_X_train(X_train)

                new_X = X_train[i].tolist()
                new_X.append(1)
                X_array = np.array(new_X)
                
                value = np.dot(new_w, X_array)
                
                if value >= 0:
                    if Y_train[i] == 0:
                        
                        new_w = new_w - X_array

                        

                else:
                    if Y_train[i] == 1:

                        new_w = new_w + X_array





            self.w = new_w
            if len(missmatch_list) == 0:
                break


            
            
            #print ( self.w ) 


        #print (missmatch_list)
        #print (jog_biyog_select)



            
            #print(value)    
        #print ( self.w ) 
             


    def fit(self, X_train, Y_train):
        
        self.w = np.random.rand( 1, X_train.shape[1] + 1)
        #print(self.w)

        self.train_data_set(X_train, Y_train)

        #print( self.w )


      


    def predict(self, X_test, Y_test):
        predicted_data = []
        for i in range ( X_test.shape[0] ):
            new_X = X_test[i].tolist()
            new_X.append(1)
            array_new_X = np.array( new_X )
            value = np.dot( self.w, array_new_X )

            if value >= 0:
                predicted_data.append(1)
            else:
                predicted_data.append(0)
        
        #print ( predicted_data )
        accu = metrics.accuracy_score(predicted_data, Y_test)
        print ( "Local: ",accu * 100, "%" )
    
    
def sklearn_perceptron(max_iterr, X_train, Y_train, X_test, Y_test):

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    percept = Perceptron(max_iter=max_iterr, alpha=0.1, random_state=1)
    percept.fit(X_train_std, Y_train)

    y_pred = percept.predict(X_test_std)

    accu = metrics.accuracy_score( y_pred, Y_test )
    print ( "SkLearn Library: ",accu * 100, "%" )
        
 

def main():

    data = np.genfromtxt("perceptron_data.csv", delimiter=",")
    #print( (data.shape[1]) )
    
    X = data[ :, :data.shape[1] - 1 ]
    Y = data[ :, data.shape[1] - 1 ]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1, stratify=Y)
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1)

    #print(  type( X_train[0].tolist() )  )

    max_iter = int( input("Enter Max Iteration: ") )

    obj1 = Perceptrons(max_iter)  


    obj1.fit(X_train, Y_train)
    obj1.predict( X_test, Y_test )

    sklearn_perceptron(max_iter, X_train, Y_train, X_test, Y_test)



if __name__ == "__main__":
    main()