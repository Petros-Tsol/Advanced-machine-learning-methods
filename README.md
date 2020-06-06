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
