import pandas
import matplotlib.pyplot as plt

a = pandas.read_csv("Dataset/handwritten/Handwritten_V2_train.csv",header=None)
e = pandas.read_csv("Dataset/handwritten/Handwritten_V2_test.csv",header=None)
x = pandas.read_csv("Dataset/handwritten/Handwritten_V2_valid.csv",header=None)


b = list(a[0])
f = list(e[0])
y = list(x[0])

# z = a.drop(0,axis=1,inplace=False)
# dens = []
# for i in range(len(z)):
#     s = sum(z.iloc[i,:])
#     dens.append(s/1024)

# plt.hist(dens)

# plt.hist(b,color="red",label="Train")
# plt.hist(f,color="blue",label="Test")
# plt.hist(y,color="green",label="Valid")
plt.show()