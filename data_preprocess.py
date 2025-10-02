import pandas as pd
from sklearn.model_selection import train_test_split

def get_data():
    df = pd.read_csv('remote_work_productivity.csv')

    df["Productive"] = (df["Productivity_Score"] > 70).astype(int)
    df = pd.get_dummies(df, columns=["Employment_Type"], drop_first=True)
    df["Balance_Ratio"] = df["Well_Being_Score"] / (df["Hours_Worked_Per_Week"] + 1) 
    df["Hours_WellBeing_Interaction"] = df["Hours_Worked_Per_Week"] * df["Well_Being_Score"]

    X = df.drop(["Employee_ID", "Productivity_Score", "Productive"], axis=1)
    y = df["Productive"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

    return X,X_train,X_test,y_train,y_test