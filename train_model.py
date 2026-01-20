import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def LoadDataset():
    return pd.read_csv("dataset/high_diamond_ranked_10min.csv")


def PreprocessData(df):
    df = df.dropna()

    X = df.drop(columns=[
        'gameId', 'blueWins', 'blueGoldDiff', 'blueExperienceDiff',
        'redGoldDiff', 'redExperienceDiff', 'redGoldPerMin',
        'blueGoldPerMin', 'redAvgLevel', 'blueAvgLevel',
        'blueCSPerMin', 'redCSPerMin',
        'redWardsPlaced', 'blueWardsPlaced',
        'redWardsDestroyed', 'blueWardsDestroyed',
        'redTotalJungleMinionsKilled', 'blueTotalJungleMinionsKilled'
    ])

    y = df['blueWins']
    return X, y


def TrainAndSaveModel(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=10000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))

    joblib.dump(model, "backend/model.pkl")
    
    with open("backend/features.json", "w") as f:
        json.dump(list(X.columns), f)

    print("Model and features saved successfully.")


if __name__ == "__main__":
    df = LoadDataset()
    X, y = PreprocessData(df)
    TrainAndSaveModel(X, y)
