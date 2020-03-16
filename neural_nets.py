# baseline cnn model for mnist
from numpy import mean
from numpy import std

from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import pandas

def load_dataset():
    train_data = pandas.read_csv("Dataset/handwritten/Handwritten_V2_train.csv",header=None)
    test_data = pandas.read_csv("Dataset/handwritten/Handwritten_V2_test.csv",header=None)

    y_train = train_data[0]
    x_train = train_data.drop([0],axis=1,inplace=False)

    y_test = test_data[0]
    x_test = test_data.drop([0],axis=1,inplace=False)

    y_train = to_categorical(y_train)
    x_train = to_categorical(x_train)
    return x_train , y_train , x_test ,y_test


def define_model():

    model = Sequential()
    model.add(Flatten())
    model.add(Flatten())
    model.add(Dense(None, activation='relu'))
    model.add(Dense(85, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()
 
# summarize model performance
def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    pyplot.boxplot(scores)
    pyplot.show()
 
# run the test harness for evaluating a model
def run_test_harness():
	
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)
 
# entry point, run the test harness
run_test_harness()