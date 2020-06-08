import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, coverage_error, label_ranking_loss, label_ranking_average_precision_score, hamming_loss

#read files
train_data = pd.read_csv('.data/train-data.dat',delimiter = '\n', header=None)
test_data = pd.read_csv('.data/test-data.dat',delimiter = '\n', header=None)
train_labels = pd.read_csv('.data/train-label.dat', sep = ' ', header=None)
test_labels = pd.read_csv('.data/test-label.dat', sep = ' ', header=None)

#replace <D> with nothing from data
train_data = train_data.iloc[:,0].str.replace('<\d+>','')
test_data = test_data.iloc[:,0].str.replace('<\d+>','')

#count the frequency of every word in vocabulary in each document
vectorizer = CountVectorizer()
train_data_vector = vectorizer.fit_transform(train_data)
test_data_vector = vectorizer.transform(test_data)

#train the classifier
model = ClassifierChain(RandomForestClassifier(n_jobs=-1, verbose=1))
model.fit(train_data_vector,train_labels)

#test the classifier
predicted_labels = model.predict(test_data_vector)
predicted_labels_train = model.predict(train_data_vector)
predicted_probabilities = model.predict_proba(test_data_vector)

#test accuracy 
#~7% with random forest and binary relevance
#~7% with random forest and classifier chain
#~5% with random forest and label powerset
#~4% with multilabel knn
test_acc = accuracy_score(test_labels,predicted_labels)
train_acc = accuracy_score(train_labels,predicted_labels_train)
test_hamm_loss = hamming_loss(test_labels,predicted_labels)
test_cov_err = coverage_error(test_labels,predicted_probabilities.toarray())
test_rank_loss = label_ranking_loss(test_labels,predicted_probabilities.toarray())
test_avr_prec = label_ranking_average_precision_score(test_labels,predicted_probabilities.toarray())

print("Train accuracy: ",train_acc)
print("Test accuracy: ",test_acc)
print("Hamming loss: ",test_hamm_loss)
print("Coverage error: ",test_cov_err)
print("Ranking loss: ",test_rank_loss)
print("Average precision: ",test_avr_prec)
