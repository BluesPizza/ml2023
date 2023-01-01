import numpy as np
import mnist
import nn
import pickle

X_train, Y_train, X_test, Y_test = mnist.load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)
X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test = X_test.reshape(X_test.shape[0],1,28,28)

# 调用训练好的模型进行测试 10000张图相当于625个batch 大概需要十来分钟跑完
model = nn.LeNet5()
weights = pickle.load(open("./model30000/weights96.pkl", "rb"))
model.set_params(weights)
print("测试开始")
Y_pred = model.forward(X_test)
result = np.argmax(Y_pred, axis=1) - Y_test
result = list(result)
print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(result.count(0)/X_test.shape[0]))