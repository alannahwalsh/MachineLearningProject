import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd

#read in data
df = pd.read_csv("Instagram_Data.csv")
#Posts vs Followers
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,3]

#Followers vs Following
#X1=df.iloc[:,1]
#X2=df.iloc[:,2]
#X=np.column_stack((X1,X2))
#y=df.iloc[:,3]

#data on own
plt.scatter(X1[y=='M'],X2[y=='M'],color='blue',marker="o")
plt.scatter(X1[y=='F'],X2[y=='F'],color='red',marker="+")
plt.title('Plot of no. Posts vs no. Followers data on own')
plt.xlabel('no. Posts->')
plt.ylabel('no. Followers->')
plt.title('Plot of no. Followers vs no. Following data on own')
plt.xlabel('no. Followers->')
plt.ylabel('no. Following->')
plt.legend(["M","F"],loc='upper right',ncol=2,fontsize=8)
plt.show()

#train model logistic regression
model = LogisticRegression(penalty='none',solver='lbfgs').fit(X, y)
print("Intercept %f"%(model.intercept_))
print("Coefficients",model.coef_)
print("Accuracy",model.score(X, y))
predic = model.predict(X)

#decision boundary
#extract the slope to display from model coefs
#line_bias = model.intercept_
#line_w = model.coef_.T
#points = [(line_w[0]*x+line_bias)/(-1*line_w[1])for x in X1]

m = plt.scatter(X1[y=='M'],X2[y=='M'],color='blue',marker="o")
f = plt.scatter(X1[y=='F'],X2[y=='F'],color='red',marker="o")

m_pred = plt.scatter(X1[predic=='M'],X2[predic=='M'],color='yellow',marker='.') 
f_pred = plt.scatter(X1[predic=='F'],X2[predic=='F'],color='green',marker='.')

plt.rcParams['figure.constrained_layout.use'] = True 
#decision_boundary = plt.plot(X1, points, color='black', linewidth=1) 
plt.title('Logistic Regression of no. Posts vs no. Followers')
#plt.title('Logistic Regression of no. Followers vs no. Following')
plt.xlabel('no. Followers->')
plt.ylabel('no. Following->')
#plt.legend(["decision_boundary","m", "f", "m_pred", "f_pred"]
plt.legend(["m", "f", "m_pred", "f_pred"],loc='upper right',ncol=2,fontsize=7)
plt.show()
