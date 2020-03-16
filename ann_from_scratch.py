from keras import Sequential
from keras.layers import Dense,Activation,Conv1D,Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD
import pandas


model = Sequential()
#model.add(Conv1D(128,kernel_size=1,activation="relu",input_shape=(17236,86)))
model.add(Flatten())
model.add(Dense(64,input_shape=(1024,2)))
model.add(Activation("relu"))
model.add(Dense(86,activation="relu"))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer="rmsprop",loss='categorical_crossentropy',metrics=['accuracy'])

def load_dataset():
    train_data = pandas.read_csv("Dataset/handwritten/Handwritten_V2_train.csv",header=None)
    test_data = pandas.read_csv("Dataset/handwritten/Handwritten_V2_test.csv",header=None)

    y_train = train_data[0]
    x_train = train_data.drop([0],axis=1,inplace=False)

    y_test = test_data[0]
    x_test = test_data.drop([0],axis=1,inplace=False)

    y_train = to_categorical(y_train)
    x_train = to_categorical(x_train)

    y_test = to_categorical(y_test)
    x_test = to_categorical(x_test)

    return x_train , y_train , x_test ,y_test

x_train,y_train,x_test,y_test = load_dataset()

model.fit(x_train,y_train,epochs=10,batch_size=128)

model.evaluate(x_test,y_test,batch_size=128)
classes = model.predict(x_test, batch_size=128)
print(classes)