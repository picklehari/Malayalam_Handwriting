import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = pandas.read_csv("Dataset/handwritten/Handwritten_V2_test.csv",header=None)
data_2 = pandas.read_csv("Dataset/handwritten/Handwritten_V2_train.csv",header=None)
data_3 = pandas.read_csv("Dataset/handwritten/Handwritten_V2_valid.csv",header=None)

data = data.transpose()
file_names = list(data.iloc[0])
filepath = "Dataset/Data/"
data = data.drop(index=0)
data = data.transpose()
data = data.to_numpy()
x = 0
for rows in data:
    img = rows
    img = numpy.reshape(img,(32,32))
    img = img.transpose()
    
    name = "Dataset/Data/" +str(file_names[x]) +"_" + str(x) + ".svg"
    x = x+1
    plt.imsave(name,img,cmap=cm.gray)

data_2 = data_2.transpose()
file_names = list(data_2.iloc[0])
data_2 = data_2.drop(index=0)
data_2 = data_2.transpose()
data_2 = data_2.to_numpy()
for rows in data_2:
    img = rows
    img = numpy.reshape(img,(32,32))
    img = img.transpose()
    
    name = "Dataset/Data/" +str(file_names[x]) +"_" + str(x) + ".svg"
    x = x+1
    plt.imsave(name,img,cmap=cm.gray)

data_3 = data_3.transpose()
file_names = list(data_3.iloc[0])
data_3 = data_3.drop(index=0)
data_3 = data_3.transpose()
data_3 = data_3.to_numpy()

for rows in data_3:
    img = rows
    img = numpy.reshape(img,(32,32))
    img = img.transpose()
    
    name = "Dataset/Data/" +str(file_names[x]) +"_" + str(x) + ".svg"
    x = x+1
    plt.imsave(name,img,cmap=cm.gray)

