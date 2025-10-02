from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def tree(X_train,X_test,y_train,y_test):
    clf = DecisionTreeClassifier(max_depth=3, random_state=42,criterion='gini',min_samples_leaf=10,min_samples_split=2)
    clf.fit(X_train, y_train)
    print("Decision Tree Accuracy:", clf.score(X_test, y_test))
    print("Tree depth:", clf.get_depth())
    print("Number of leaves:", clf.get_n_leaves())
    return clf

def predict(clf,in_model):
    pred = clf.predict(in_model)
    print("Predicted:", "Productive" if pred[0]==1 else "Unproductive")
