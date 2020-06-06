from pandas import read_csv
from numpy import array, concatenate, repeat, append, zeros
from numpy.random import shuffle
from random import randint

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

def prediction(classifiers,classifiers_names,cost_matrix,data):
    for i,j in zip(classifiers, classifiers_names):
        #10-fold cross validation
        predicted_labels = cross_val_predict(i,data[:,0:13],data[:,13],cv=10)

        #confusion matrix
        conf_matrix = confusion_matrix(data[:,13],predicted_labels)
        print(conf_matrix)
        TN, FP, FN, TP = conf_matrix.ravel()

        print("Total cost for",j,"...",TN*cost_matrix[0,0]+FP*cost_matrix[0,1]+FN*cost_matrix[1,0]+TP*cost_matrix[1,1])



def main():
    #read file
    data = read_csv('./data/heart.csv',sep=" ",header=None)
    data_array = data.to_numpy()

    #define the cost matrix
    cost_matrix = array([[0,1],[5,0]])

    #define the classifiers to be used
    classifiers = [make_pipeline(StandardScaler(), svm.LinearSVC(max_iter=5000)), make_pipeline(StandardScaler(), RandomForestClassifier()), make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))]
    classifiers_with_weights = [make_pipeline(StandardScaler(), svm.LinearSVC(class_weight={1:cost_matrix[0,1],2:cost_matrix[1,0]},max_iter=5000)), make_pipeline(StandardScaler(),RandomForestClassifier(class_weight={1:cost_matrix[0,1],2:cost_matrix[1,0]})), make_pipeline(StandardScaler(), LogisticRegression(class_weight={1:cost_matrix[0,1],2:cost_matrix[1,0]},max_iter=3000))]
    classifiers_names = ["Linear SVM","Random Forest","Logistic Regression"]

    print("---------------------No cost method---------------------")
    prediction(classifiers,classifiers_names,cost_matrix,data_array)

    #############oversampling####################
    indexes_of_class_2 = data.index[data.iloc[:,13] == 2] #find the indexes of class 2, thats the class with the higher cost if predicted wrong
    indexes_of_class_1 = data.index[data.iloc[:,13] == 1] #find the indexes of class 1, thats the class with the lower cost if predicted wrong

    #the instances from the class with the higher cost must be n times more than the instances of lower misclassification cost, where n is the misclassification cost of the higher cost class
    new_data = repeat(data.values[indexes_of_class_2,:],cost_matrix[1,0]+1,axis=0)
    new_data = append(new_data,data.values[indexes_of_class_1,:],axis=0)

    #shuffle the data
    oversampling_data = shuffle(new_data)

    print("---------------------Oversampling data---------------------")
    prediction(classifiers,classifiers_names,cost_matrix,oversampling_data)

    #############undersampling####################
    #the istances of the lower misclassification class must be n time less than the instances of lower miclassification cost, where n is the misclassification cost of the higher cost class
    shuffle(indexes_of_class_1) # shuffle the indexes to get a random permutation which is gonna me the first 24 indexes (if misclassification cost is 5), these indexes will be different every time the algorithm start from the beginning

    new_data = data.values[indexes_of_class_1[0:24],:]
    new_data = append(new_data,data.values[indexes_of_class_2,:],axis=0)

    #shuffle the data
    undersampling_data = shuffle(new_data)

    print("---------------------Undersampling data---------------------")
    prediction(classifiers,classifiers_names,cost_matrix,undersampling_data)

    ###################CSRoulette##########################
    #algorithm according to CSRoulette paper presented in class
    #weights of classes
    #weight of positives = FN_cost*instances / (positive_instances*FN_cost + negative_instances*FP_cost)
    #weight of negatives = FP_cost*instances / (positive_instances*FN_cost + negative_instances*FP_cost)
    weight_positive = (cost_matrix[1,0]* (indexes_of_class_1.size+indexes_of_class_2.size)) / (indexes_of_class_2.size * cost_matrix[1,0] + indexes_of_class_1.size * cost_matrix[0,1])
    weight_negative = (cost_matrix[0,1]* (indexes_of_class_1.size+indexes_of_class_2.size)) / (indexes_of_class_2.size * cost_matrix[1,0] + indexes_of_class_1.size * cost_matrix[0,1])

    weights_matrix = zeros((indexes_of_class_1.size+indexes_of_class_2.size,1))

    weights_matrix[indexes_of_class_1,0] = weight_negative
    weights_matrix[indexes_of_class_2,0] = weight_positive

    #sum_of_weights is equal to dataset instances
    sum_of_weights = weight_positive*indexes_of_class_2.size + weight_negative*indexes_of_class_1.size

    #number of instances to draw with replacement sample
    number_of_instances = 400
    new_data = zeros((number_of_instances,14))

    #we want to take instances according to their weights
    #imagine all the weights are in a line
    #the length of this line equals the number of instances
    #now we draw a random number in range [0,number_of_instances]
    #we iterate through the instances and we keep the instance which its distance (remember the line...) is equal or greater this particular random number
    #repeat the above process until we reach the desirable number of instances we want to keep 

    k=0
    while k < number_of_instances:
        sum = 0
        n = randint(0,int(sum_of_weights))
        for ind, i in enumerate(weights_matrix):
            if sum < n:
                sum = sum + i
            else:
                break
        new_data[k,:] = data.iloc[ind,:]
        k = k+1

    roulette_data = shuffle(new_data)

    #the paper suggest a bagging classifier after the sampling of data
    #for the purpose of this task we use the algorithms that we have used above
    print("---------------------CSRoulette---------------------")
    prediction(classifiers,classifiers_names,cost_matrix,roulette_data)

    ###################Class weight##########################
    print("---------------------Class weight---------------------")
    prediction(classifiers_with_weights,classifiers_names,cost_matrix,data_array)



main()