import os
import pandas
from sklearn import svm
import PIL
import numpy
import joblib
from sklearn.metrics import confusion_matrix
from PIL import Image
import ast

images = []
f = open("real_inputs/output_new.csv" ,"r+") 
img = []
lines = f.readlines()
count = 0
for l in lines:
    l = l.replace('\n' ,"")
    l = l.replace(" ",",")
    l = ast.literal_eval(l)
    if count ==0:
        images.append(img)
        img = []
        img.extend(l)
    else:
        img.extend(l)
    count = (count + 1)%32

real_data = []
for img in images:
    if img == []:
        continue
    real_data.append(img)

# print(real_data[0])
# print(len(real_data[0]))
real_data_points = pandas.DataFrame(real_data)
print(real_data_points.head())

#images = []
# path = "real_inputs/"
# for img in os.listdir(path):
#     img = Image.open(path + img)
#     data = numpy.asarray(img)
#     data_2 = []
#     for x in data:
#         for y in data:  
#             data_2.append(sum(y))
#     data_2 = numpy.array(data_2)
#     # data_2 = [0 if set(list(x)).Intersect([68,1,84,255]) else 1 for x in data]
#     # data_2 = list(map(lambda x: 1 if x==[68,1,84,255] else 0 ,data))
#     print(numpy.unique(data_2)  )
#     print(len(data_2))


# test_data = pandas.read_csv("Dataset/handwritten/Handwritten_V2_test.csv",header=None)
# y_test = test_data[0]
# x_test = test_data.drop([0],axis=1,inplace=False)



classifier = joblib.load("model_file.joblib")

s = classifier.predict(real_data_points)
# cm = confusion_matrix(s,y_test)
alphabet = "്  ാ  ി  ീ  ു  ൂ  െ ൃ  െ  ൌ  ം അ ആ ഇ ഉ ഋ എ ഏ ഒ ക ഖ ഗ ഘ ങ ച ഛ ജ ഝ ഞ ട ഠ ഢ ഡ ണ ത ഫ ദ ധ ന പ ഫ ബ ഭ മ യ ര റ ല ള ഴ വ ശ ഷ സ ഹ ൺ ൻ ർ ൽ ൾ ക്ക ക്ഷ ങ്ക ങ്ങ ച്ച ഞ്ച ഞ്ഞ ട്ട ണ്ട ണ്ണ ത്ത ദ്ധ ന്ത ന്ദ ന്ന പ്പ മ്പ മ്മ യ്യ ല്ല ള്ള  ്യ   ്ര  ്വ"
alphabet = alphabet.split(" ")
alphabets = []
for x in alphabet:
    if x!= '':
        alphabets.append(x)

# for i in s:
#     print(alphabets[i])
for i in s:
    print(alphabets[int(i)-1])