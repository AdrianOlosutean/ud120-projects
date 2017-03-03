#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import precision_score, recall_score

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                            test_size=0.3,
                                                                                            random_state=42)
clf = DecisionTreeClassifier()


clf.fit(features_train, labels_train)

print len([e for e in labels_test if e == 1.0])

pred = clf.predict(features_test)

print precision_score(pred, labels_test)
print recall_score(pred, labels_test)
