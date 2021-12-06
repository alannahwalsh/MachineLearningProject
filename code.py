import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve


def Qi_a(x1, x2, X, y):
    posX1 = []
    posX2 = []
    negX1 = []
    negX2 = []
    i = 0
    for val in y:
        if y[i] == "1":
            posX1.append(x1[i])
            posX2.append(x2[i])
        else:
            negX1.append(x1[i])
            negX2.append(x2[i])
        i = i + 1
    plt.scatter(posX1, posX2, c='Red', label='men')
    plt.scatter(negX1, negX2, c='Blue', label='women')
    plt.xlabel("input 1")
    plt.ylabel("input 2")
    plt.legend()
    plt.show()  #comparing Targets with +1 in red to Targets with -1 in blue

    polyFeat = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    meanList = []
    devList = []
    for val in polyFeat:
        estimates = []
        k = KFold(n_splits=5)
        tModel = LogisticRegression(penalty="l2", C=1)
        for data, test in k.split(X):
            poly = PolynomialFeatures(degree=val)
            polyFit = poly.fit_transform(X[data])
            tModel.fit(polyFit, y[data])
            fitData = poly.fit_transform(X[test])
            prediction = tModel.predict(fitData)
            estimates.append(mean_squared_error(prediction, y[test]))

        meanList.append(np.mean(estimates))
        devList.append(np.std(estimates))

    plt.errorbar(polyFeat, meanList, yerr=devList, capsize=5)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Prediction error mean and standard deviation')
    plt.show()


    cVal = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    meanList = []
    devList = []
    degree = 1  #whatever the optimal degree value is from cross-val above
    for val in cVal:
        estimates = []
        k = KFold(n_splits=5)
        tModel = LogisticRegression(penalty="l2", C=val)
        for data, test in k.split(X):
            poly = PolynomialFeatures(degree=degree)
            polyFit = poly.fit_transform(X[data])
            tModel.fit(polyFit, y[data])
            fitData = poly.fit_transform(X[test])
            prediction = tModel.predict(fitData)
            estimates.append(mean_squared_error(prediction, y[test]))

        meanList.append(np.mean(estimates))
        devList.append(np.std(estimates))

    plt.errorbar(np.log10(cVal), meanList, yerr=devList, capsize=5)
    plt.title("Polynomial degree " + str(degree))
    plt.xlabel('C-Value, log10(c)')
    plt.ylabel('Prediction error mean and standard deviation')
    plt.show()


def Qi_b(X, y):
    kList = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    meanList = []
    devList = []
    for val in kList:
        estimates = []
        k = KFold(n_splits=5)
        tModel = KNeighborsClassifier(n_neighbors=val, weights="uniform")
        for data, test in k.split(X):
            newTModel = tModel.fit(X[data], y[data])
            prediction = newTModel.predict(X[test])
            estimates.append(mean_squared_error(prediction, y[test]))

        meanList.append(np.mean(estimates))
        devList.append(np.std(estimates))

    plt.errorbar(kList, meanList, yerr=devList, capsize=5)
    plt.xlabel('Number of neighbours')
    plt.ylabel('Prediction error mean and standard deviation')
    plt.show()


def Qi_c(X, y):
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
    c = 0.0001
    degree = 1
    # Logistic Regression
    tModel = LogisticRegression(penalty="l2", C=c)
    poly = PolynomialFeatures(degree=degree)
    polyFit = poly.fit_transform(xTrain)
    tModel.fit(polyFit, yTrain)
    fitData = poly.fit_transform(xTest)
    prediction = tModel.predict(fitData)
    print(confusion_matrix(yTest, prediction))

    # kNN
    tModel = KNeighborsClassifier(n_neighbors=20, weights="uniform")
    newTModel = tModel.fit(xTrain, yTrain)
    prediction = newTModel.predict(xTest)
    print(confusion_matrix(yTest, prediction))

    # baseline most freq
    dummyModel = DummyClassifier(strategy="most_frequent").fit(xTrain, yTrain)
    yDummy = dummyModel.predict(xTest)
    print(confusion_matrix(yTest, yDummy))


def Qi_d(X, y):
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
    tModel = LogisticRegression(penalty="l2", C=0.0001)
    poly = PolynomialFeatures(degree=1)
    tModel.fit(poly.fit_transform(xTrain), yTrain)
    polyFit = poly.fit_transform(xTest)
    fpr, tpr, _ = roc_curve(yTest, tModel.decision_function(polyFit), pos_label='M')
    plt.plot(fpr, tpr)
    plt.plot(1, 1, label='Baseline - Most Frequent', marker='o')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC for Logistic Regression')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.show()

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
    tModel = KNeighborsClassifier(n_neighbors=20, weights="uniform")
    newTModel = tModel.fit(xTrain, yTrain)
    prob = newTModel.predict_proba(xTest)
    fpr, tpr, _ = roc_curve(yTest, prob[:, 1], pos_label='M')
    plt.plot(fpr, tpr)
    plt.plot(1, 1, label='Baseline - Most Frequent', marker='o')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC for kNN')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("InstagramData.csv")
    X1 = df.iloc[:, 0] #change X1 and X2 values for different features
    X2 = df.iloc[:, 5]
    y = df.iloc[:, 3]
    i = 0
    for value in y:
        if math.isnan(X2[i]):
            X2[i] = 0
        if y[i] == "M":
            y[i] = '1'
        elif y[i] == "F":
            y[i] = '-1'
        i += 1
    X = np.column_stack((X1, X2))
    print(X)
    Qi_a(X1, X2, X, y)
    Qi_b(X, y)
    Qi_c(X, y)
    j = 0
    for value in y:
        if y[j] == '1':
            y[j] = "M"
        elif y[j] == '-1':
            y[j] = "F"
        j += 1
    Qi_d(X, y)
