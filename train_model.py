import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, matthews_corrcoef,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def LoadDataset():
    df = pd.read_csv('dataset/high_diamond_ranked_10min.csv')
    return df

def PreprocessData(df):
    df = df.dropna()
    '''
    df['killDiff'] = df['blueKills'] - df['redKills']
    df['deathDiff'] = df['blueDeaths'] - df['redDeaths']
    df['assistDiff'] = df['blueAssists'] - df['redAssists']

    df['goldPerKillBlue'] = df['blueTotalGold'] / (df['blueKills'] + 1)
    df['goldPerKillRed']  = df['redTotalGold'] / (df['redKills'] + 1)

    df['csDiff'] = df['blueTotalMinionsKilled'] - df['redTotalMinionsKilled']

    df['objectiveScoreBlue'] = (
        df['blueDragons'] * 2 +
        df['blueHeralds'] * 3 +
        df['blueTowersDestroyed'] * 4
    )

    df['objectiveScoreRed'] = (
        df['redDragons'] * 2 +
        df['redHeralds'] * 3 +
        df['redTowersDestroyed'] * 4
    )
    '''


    x = df.drop(columns=['gameId', 'blueWins', 'blueGoldDiff', 'blueExperienceDiff', 'redGoldDiff', 'redExperienceDiff',
                         'redGoldPerMin', 'blueGoldPerMin', 'redAvgLevel', 'blueAvgLevel', 'blueCSPerMin', 'redCSPerMin',
                         'redWardsPlaced', 'blueWardsPlaced', 'redWardsDestroyed', 'blueWardsDestroyed', 'redTotalJungleMinionsKilled', 'blueTotalJungleMinionsKilled'
                         #'blueKills', 'redKills', 'blueDeaths', 'redDeaths', 'blueAssists', 'redAssists',
                         #'blueTotalGold', 'blueKills', 'redTotalGold', 'redKills', 'blueTotalMinionsKilled', 'redTotalMinionsKilled',
                         #'blueDragons', 'redDragons', 'blueHeralds', 'redHeralds', 'blueTowersDestroyed', 'redTowersDestroyed'
                         ], axis=1)

    y = df['blueWins']
    return x, y

def SplitData(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def TrainModel(X_train, X_test, y_test, y_train, model):
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    mcc_train = matthews_corrcoef(y_train, y_pred_train)
    mcc_test = matthews_corrcoef(y_test, y_pred_test)

    confusionMatrix = confusion_matrix(y_test, y_pred_test, labels=model.classes_)

    print(f"Training Accuracy: {accuracy_train}, F1 Score: {f1_train}, MCC: {mcc_train}")
    print(f"Testing Accuracy: {accuracy_test}, F1 Score: {f1_test}, MCC: {mcc_test}")
    dips = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=model.classes_)
    dips.plot(cmap=plt.cm.Blues)
    plt.show()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = X_train.columns
    
        feat_imp = (
            pd.Series(importances, index=features)
            .sort_values(ascending=False)
        )
    
        print("\nTop 15 Feature Importances:")
        print(feat_imp.head(30))

def PlotCorrelation(X, dataset, features=None):
    dataFrame = pd.DataFrame(X, columns=features if features.any() else [f"Feature_{i}" for i in range(X.shape[1])])
    correlation = dataFrame.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=False, cmap='coolwarm')
    plt.title(f"Feature Correlation Heatmap - {dataset}")
    plt.show()



if __name__ == "__main__":
    data = LoadDataset()
    x,y = PreprocessData(data)
    X_train, X_test, y_train, y_test = SplitData(x, y)


    models = [
        ('Logistic Regression', LogisticRegression(solver='lbfgs', max_iter=10000)),
        ('Decision Tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=50, max_depth=5)),
        #('SVC', SVC(probability=True, kernel='linear', C=0.5)),
        ('Gaussian NB', GaussianNB())
    ]
    PlotCorrelation(x, "High Diamond Ranked 10min", features=x.columns)


    for model_name, model in models:
        print(f"Training {model_name}...")
        TrainModel(X_train, X_test, y_test, y_train, model)        
        print("\n")