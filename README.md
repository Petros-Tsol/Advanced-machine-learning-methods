# Advanced-machine-learning-methods
Implementation in Python of some advanced ML methods

## Descption
In this repository can be found some advanced machine learning methods implemented in Python. We used numpy, pandas and scikit-learn libraries.

## Cost sensitive learning
The dataset contains samples divided in two classes, either someone has not a heart disease (class 1) or has a heart disease (class 2). Along the data, its included the following cost matrix:

| Actual class / Predicted class | class 1 | class 2 |
| -------------| -----------| -----------------|
| class 1 | 0 | 1 |
| class 2 | 5 | 0 |

We implemented the oversampling, undersampling and CSRoulette method and we used the class weight method from scikit-learn. The methods were tested using 10-fold cross validation using SVM or Random Forest or Linear Regression as classifier.

## Multi-instance learning
The dataset is consisted from documents with varied number of sentences. Every document is assigned to one or more classes from the twenty available classes. You can find more info here: https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels.

We found the dominant class from the training data and classify every document with 1 or 0 whether or not this document belongs to the dominant class. After that we constructed a feature vector for each sentence with the frequency for every word in the document. Then we clustered the data using k-means and transformed the problem so that each cluster correspond to a feature. We applied ML algorithms to the transformed data to classify them.


## Multi-label learning
We used the same dataset as multi-instance learning
We constructed a feature vector for each document, with the frequency for every word in the document. We applied the classifier chains algorithm combined with the random forest algorithm.

## Libraries
We used Python with scikit-learn and scikit-multilearn along with numpy and pandas.
