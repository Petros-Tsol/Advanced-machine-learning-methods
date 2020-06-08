import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score, precision_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def preprocess_data(data):

    arrayWithWords = []
    arrayPointerOfDocuments = []
    counterSentences = -1
    counterDocuments = 0
    for index, row in data.iterrows():

        splitted_words_array = data.iloc[index, 0].split()
        # removing first array
        splitted_words_array.pop(0)

        for eachWord in splitted_words_array:
            #print(eachWord)

            if "<" in eachWord:
                counterSentences += 1

                arrayWithWords.append("")
                arrayPointerOfDocuments.append(0)

            else:
                arrayWithWords[counterSentences] += eachWord + " "
                arrayPointerOfDocuments[counterSentences] = counterDocuments

        counterDocuments += 1


    # remove last space in each last character of arayWithWords
    count = 0
    for eachWord in arrayWithWords:
        arrayWithWords[count] = arrayWithWords[count][:-1]
        count += 1

    return arrayWithWords, arrayPointerOfDocuments, counterDocuments
    
    
def main():
    # read files
    train_data = pd.read_csv('.data/train-data.dat', delimiter='\n', header=None)
    test_data = pd.read_csv('.data/test-data.dat', delimiter='\n', header=None)
    train_labels = pd.read_csv('.data/train-label.dat', sep=' ', header=None)
    test_labels = pd.read_csv('.data/test-label.dat', sep=' ', header=None)



    # initialize array to find the most popular class
    all_classes_count = []
    for eachClass in range(0, 20):
        all_classes_count.append(0)

    # simply iterate through all train rows
    for index, row in train_labels.iterrows():
        all_classes_count += row

    most_popular_class = all_classes_count.idxmax()
    print("Most popular class index:", most_popular_class)


    train_labels[20] = 0    # initialize 20th column on train_labels

    # if the most popular class exists or not add in the 20th column of train_labels
    for index, row in train_labels.iterrows():
        if train_labels.iloc[index, most_popular_class] == 1:
            train_labels.iloc[index, 20] = 1
        else:
            train_labels.iloc[index, 20] = 0


    #print(train_labels)


    test_labels[20] = 0     # initialize 20th column on test_labels

    # if the most popular class exists or not add in the 20th column of test_labels
    for index, row in test_labels.iterrows():
        if test_labels.iloc[index, most_popular_class] == 1:
            test_labels.iloc[index, 20] = 1
        else:
            test_labels.iloc[index, 20] = 0


    #print(test_labels)


    ################### DATA PREPROCESS #######################


    arrayWithWords_train, arrayPointerOfDocuments_train, counterDocuments_train = preprocess_data(train_data)
    arrayWithWords_test, arrayPointerOfDocuments_test, counterDocuments_test = preprocess_data(test_data)



    ################### VECTORIZE DATA #######################


    vectorizer = CountVectorizer()

    train_data_vector = vectorizer.fit_transform(arrayWithWords_train)
    test_data_vector = vectorizer.transform(arrayWithWords_test)

    print("train_data_vector shape:", train_data_vector.shape)
    print("test_data_vector shape:", test_data_vector.shape)


    ################### CLUSTER DATA #######################

    cluster_means = [5, 10, 50, 100, 150]

    for cluster_number in cluster_means:

        print("starting kmeans... with cluster:", cluster_number)

        kmeans_model = KMeans(n_clusters=cluster_number)

        training_label_kmeans = kmeans_model.fit_predict(train_data_vector)
        test_labels_kmeans = kmeans_model.predict(test_data_vector)

        print("finished kmeans...")

        # prepare new feature array training
        new_feature_array_training = np.zeros((counterDocuments_train, cluster_number))

        counter = 0
        for index, cluster in np.ndenumerate(training_label_kmeans):

            # get each document
            document = arrayPointerOfDocuments_train[counter]

            # if document exists in cluster
            if new_feature_array_training[document, cluster] == 0:
                new_feature_array_training[document, cluster] = 1

            counter += 1

        print("new_feature array shape:", new_feature_array_training.shape)

        # prepare new feature array test
        new_feature_array_testing = np.zeros((counterDocuments_test, cluster_number))

        counter = 0
        for index, cluster in np.ndenumerate(test_labels_kmeans):

            # get each document
            document = arrayPointerOfDocuments_test[counter]

            # if document exists in cluster
            if new_feature_array_testing[document, cluster] == 0:
                new_feature_array_testing[document, cluster] = 1

            counter += 1

        print("new_feature array test shape:", new_feature_array_testing.shape)

        ################### CLASSIFIY TO GET SCORES #######################

        names = ["Linear SVM", "RBF SVM", "Decision Tree", "Random Forest"]

        classifiers = [SVC(kernel="linear"), SVC(kernel="rbf"), DecisionTreeClassifier(),  RandomForestClassifier()]

        for index, clf in enumerate(classifiers):
            clf.fit(new_feature_array_training, train_labels[20])
            predictions = clf.predict(new_feature_array_testing)
            scores = clf.score(new_feature_array_testing, test_labels[20])

            print("Classifier:", names[index], " with Accuracy:", scores * 100, "%")

            recall = recall_score(test_labels[20], predictions)
            print("Recall:", recall * 100, "%")
            
            precision = precision_score(test_labels[20], predictions)
            print("Precision:", precision * 100, "%")
            print('--------------------------')


main()