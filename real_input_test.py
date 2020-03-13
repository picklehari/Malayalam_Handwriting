import pandas
from sklearn import svm
import numpy
import joblib
from sklearn.metrics import confusion_matrix


test_data = pandas.read_csv("Dataset/handwritten/Handwritten_V2_test.csv",header=None)
y_test = test_data[0]
x_test = test_data.drop([0],axis=1,inplace=False)

classifier = joblib.load("model_file.joblib")

s = classifier.predict(x_test)
cm = confusion_matrix(s,y_test)
print(cm)