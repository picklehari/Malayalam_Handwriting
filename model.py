import pandas
from sklearn import svm
from sklearn.metrics import confusion_matrix
import numpy
import pickle
import matplotlib.pyplot as plt


train_data = pandas.read_csv("Dataset/handwritten/Handwritten_V2_train.csv",header=None)
test_data = pandas.read_csv("Dataset/handwritten/Handwritten_V2_test.csv",header=None)

y_train = train_data[0]
x_train = train_data.drop([0],axis=1,inplace=False)

y_test = test_data[0]
x_test = test_data.drop([0],axis=1,inplace=False)

classifier=svm.SVC(gamma='auto')
classifier.fit(x_train,y_train)

with open("model",'ab') as f:
    pickle.dump(classifier,f) 

s = classifier.predict(x_test)
cm = confusion_matrix(s,y_test)

x = []
y = []
for i in range(85):
    for j in range(85):
        if cm[i][j] != 0:
            x.append(i)
            y.append(j)

plt.scatter(x,y)
plt.show()

print(len(x)/len(y_test))