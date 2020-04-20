from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np



# Iris dataset is taken as input
iris_dataset = load_iris()
print(iris_dataset)
input()

# These are the target
print("Target names: {}".format(iris_dataset['target_names']))
input()

#The features that will be feed into the model
print("Feature names: {}".format(iris_dataset['feature_names']))
input()

#The dataset
print(" Data: {}".format(iris_dataset['data']))
input()

#Spliting the dataset into training and testing 
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
input()

# Same for the test samples
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
input()


#The classifier 
knn = KNeighborsClassifier(n_neighbors=1)


#Training the classifier
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

#Prediction
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
input()
y_pred = knn.predict(X_test)
print("Actual Labels:\n {}".format(y_test))

#printing the predictions
for i in range(0,len(y_pred)):
    test=iris_dataset['target_names'][y_test[i]]
    pred=iris_dataset['target_names'][y_pred[i]]
    print("Test dataset id:{0} \t Actual Label: {1} \t Predicted label: {2}".format(i+1,test,pred))
input()

print("Test set score (np.mean): {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score (knn.score): {:.2f}".format(knn.score(X_test, y_test)))