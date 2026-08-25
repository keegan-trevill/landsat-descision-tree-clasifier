import numpy as np
import pandas as pd
import rasterio
import joblib
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from rasterio.io import MemoryFile
import tempfile
import os

col_names = ["class","Band_1","Band_2","Band_3","Band_4","Band_5","Band_6","Band_7"]
data = pd.read_csv(r"C:\\Users\\keega\\OneDrive\\Documents\\GitHub\\LandSat_Classifier\\Training_Updated.csv", skiprows=1, header = None, names = col_names)
#data['class'] = data['class'].map({'NonVeg': 0, 'Veg': 1})
print(data.head(10))
print("Hello")
# %%
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain = None, value=None):
        #for decision Node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

# %%
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        # initialize the root of the tree
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
   
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth< self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values) #RETURNED VALUES OF feature
            # loop over all the unique values of the feature
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:,-1], dataset_left[:,-1], dataset_right[:,-1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        
        return best_split

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y): # find most occuring element in Y
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    def make_prediction(self, x, tree):
        if tree.value!=None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

# These are the feature columns that will be used to train the model. The features are the Landsat bands 1-7, which are used to classify the pixels into vegetation and non-vegetation classes.
feature_columns = [
    "Band_1",
    "Band_2",
    "Band_3",
    "Band_4",
    "Band_5",
    "Band_6",
    "Band_7"
]

X = data[feature_columns].to_numpy()
Y = data["class"].to_numpy().reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train, Y_train) 
classifier.print_tree()

#shows accuracy of the model
Y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test.ravel(), Y_pred)

from sklearn.metrics import confusion_matrix, classification_report

print("Predicted:", Y_pred)
print("Actual:", Y_test.ravel())

print(confusion_matrix(Y_test.ravel(), Y_pred))

print(
    classification_report(
        Y_test.ravel(),
        Y_pred,
        target_names=["Non-vegetation", "Vegetation"]
    )
)

print("Total pixels:", len(data))
print("Training pixels:", len(X_train))
print("Testing pixels:", len(X_test))
print()

print("Training classes:")
print(pd.Series(Y_train.ravel()).value_counts())

print()

print("Testing classes:")
print(pd.Series(Y_test.ravel()).value_counts())

# Column 0 = class
# Columns 1-7 = Landsat B1-B7
X = data.iloc[:, 1:8].to_numpy()
y = data.iloc[:, 0].to_numpy()

print(X)
print(y)

print("\nClass counts:")
print(data.iloc[:, 0].value_counts())


#Train-test split with stratification to ensure that the class distribution is preserved in both the training and testing sets. The test size is set to 20% of the data, and a random state is provided for reproducibility.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=17
)


#Set up the Random Forest Classifier with a random state for reproducibility and n_jobs=-1 to use all available CPU cores for parallel processing.
rf = RandomForestClassifier(
    random_state=17,
    n_jobs=-1
)

rf.fit(X_train, y_train)

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "landsat_random_forest.pkl"
)

accuracy = rf.score(X_test, y_test)

model_data = {
    "model": rf,
    "accuracy": accuracy
}

joblib.dump(
    model_data,
    "landsat_random_forest.pkl"
)

print(f"Model accuracy: {accuracy:.2%}")

print("\nModel saved!")
print("Model location:", model_path)

y_pred = rf.predict(X_test)

print(
    rf.score(X_test, y_test),
    "is the accuracy of the model"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#Find the misclassified feature rows, actual labels, and predicted labels. This is done to identify which samples were incorrectly classified by the model.

wrong_mask = y_pred != y_test

print("\nMisclassified feature rows:")
print(X_test[wrong_mask])

print("\nActual labels:")
print(y_test[wrong_mask])

print("\nPredicted labels:")
print(y_pred[wrong_mask])


#Feature importances are calculated by the Random Forest model to determine which bands are most important for making predictions. The importance of each feature is a measure of how much that feature contributes to the model's decision-making process.

feature_names = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7"
]

print("\nFeature Importances:")

for feature, importance in zip(
    feature_names,
    rf.feature_importances_
):
    print(feature, round(importance, 3))


print("Model saved as landsat_random_forest.pkl")