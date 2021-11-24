from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier

# Read in data
df = pd.read_csv("Instagram_Data.csv")
#Posts vs Followers
#X1=df.iloc[:,0]
#X2=df.iloc[:,1]
#X=np.column_stack((X1,X2))
#y=df.iloc[:,3]

#Followers vs Following
X1=df.iloc[:,1]
X2=df.iloc[:,2]
X=np.column_stack((X1,X2))
y=df.iloc[:,3]

#polynomials
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X)
print("printing in order: tn, fp, fn, tp")

#train models
knnModel = KNeighborsClassifier(n_neighbors=1, weights='uniform').fit(X,y)
knn_pred = knnModel.predict(X)
tn,fp,fn,tp = confusion_matrix(y,knn_pred).ravel()
#print("knn_model of no. Posts vs no. Followers:", tn, fp, fn, tp)
print("knn_model of no. Followers vs no. Following", tn, fp, fn, tp)
lgModel = LogisticRegression(C=8).fit(X_poly,y)
lg_pred = lgModel.predict(X_poly)
tn,fp,fn,tp = confusion_matrix(y, lg_pred).ravel()
#print("logistic_regression_model of no. Posts vs no. Followers:", tn, fp, fn, tp)
print("logistic_regression_model of no. Followers vs no. Following:", tn, fp, fn, tp)
randomModel = DummyClassifier(strategy="uniform")
mostFreqModel = DummyClassifier(strategy="most_frequent")
randomModel.fit(X,y)
mostFreqModel.fit(X,y)

#random
rqndom_pred = randomModel.predict(X)
tn, fp, fn, tp = confusion_matrix(y, rqndom_pred).ravel()
#print("random_model of no. Posts vs no. Followers:", tn, fp, fn, tp)
print("random_model of no. Followers vs no. Following:", tn, fp, fn, tp)
# Uniform
most_freq_pred = mostFreqModel.predict(X)
tn, fp, fn, tp = confusion_matrix(y,most_freq_pred).ravel()
#print("most_freq_value_model of no. Posts vs no. Followers:", tn, fp, fn, tp)
print("most_freq_value_model of no. Followers vs no. Following:", tn, fp, fn, tp)

#logistic roc
fpr, tpr, _ = roc_curve(y, lgModel.decision_function(X_poly),pos_label='F')
# knn roc
knn_proba = knnModel.predict_proba(X)
knn_fpr, knn_tpr, thresh = roc_curve(y, knn_proba[:,1],pos_label='F')
#most freq val roc
most_freq_proba = mostFreqModel.predict_proba(X)
most_freq_fpr, most_freq_tpr, thresh = roc_curve(y,most_freq_proba[:,1],pos_label='F')
#random roc
rand_proba = randomModel.predict_proba(X)
rand_fpr, rand_tpr, thresh = roc_curve(y,rand_proba[:,1],pos_label='F')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(fpr, tpr, color='yellow')
ax.plot(knn_fpr, knn_tpr, color='orange')
ax.plot(most_freq_fpr, most_freq_tpr, color='blue')
ax.plot(rand_fpr, rand_tpr, color='red')
ax.plot([0,1], [0,1], color='green', linestyle='--')
ax.set_ylabel('True positive rate',fontsize=10)
ax.set_xlabel('False positive rate',fontsize=10)
#ax.set_title("Cross-validation of neighbors of no. Posts vs no. Followers " + "range for a kNN model",fontsize=10)
ax.set_title("Cross-validation of neighbors of no. Followers vs no. Following " + "range for a kNN model",fontsize=10)
plt.legend(["Logistic Regression", "kNN", "Most Frequent value ", "Random"],fontsize=9)
plt.show()