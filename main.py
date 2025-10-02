from data_preprocess import get_data
from model import tree,predict
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def run_model():
    X,X_train,X_test,y_train,y_test = get_data()
    model_tree = tree(X_train,X_test,y_train,y_test)
    
    plt.figure(figsize=(12,8))
    plot_tree(model_tree, filled=True, feature_names=X.columns, class_names=["Unproductive","Productive"], rounded=True, proportion=True)
    plt.savefig("decision_tree.png", dpi=300)

    while True:
        Hours_Worked_Per_Week = int(input("Enter Hours worked per Week : "))
        Well_Being_Score = int(input("Enter your Well being score : "))
        Employment_Type_Remote = int(input("Enter Employment Type (1 : Remote, 0 : In-Office) : "))

        in_model = pd.DataFrame({
            "Hours_Worked_Per_Week": [Hours_Worked_Per_Week],
            "Well_Being_Score": [Well_Being_Score],
            "Employment_Type_Remote": [Employment_Type_Remote]  
        })
        predict(model_tree,in_model)
        want_to_continue = int(input("Want to continue, Yes-1 & No-0 : "))
        if want_to_continue==0: break
        

if __name__=="__main__":
    run_model()