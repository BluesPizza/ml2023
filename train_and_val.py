import numpy as np
import mnist
import matplotlib.pyplot as plt
import util
import layer
import nn
import optimizer
import pickle
import loss

X_train, Y_train, X_test, Y_test = mnist.load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)
# 传入第一层后会被自动resize成 60000*1*32*32
X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test = X_test.reshape(X_test.shape[0],1,28,28)

batch_size = 32  # 16
D_out = 10  # 分组数目
lr = 0.0001  # 0.00003

model = nn.LeNet5()
losses = []
optim = optimizer.SGD(model.get_params(), lr=lr)
criterion = loss.SoftmaxLoss()

# Train 30000非常耗时 大概要10h 正确率95%左右
# 如果只是想测试能不能用还请将iter调到几百就行 但是正确率会很低 可能只有70%+ 3000个iter大概需要60min 有92%+正确率
# 只测试模型性能请使用model_test.py 测试目前得到的最高性能模型
ITER = 30000  # 30000
for i in range(ITER):
    # get batch, make onehot
    X_batch, Y_batch = util.get_batch(X_train, Y_train, batch_size)
    Y_batch = util.MakeOneHot(Y_batch, D_out)

    # forward, loss, backward, step
    Y_pred = model.forward(X_batch)
    loss, dout = criterion.get(Y_pred, Y_batch)
    model.backward(dout)
    optim.step()

    print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
    losses.append(loss)
    """
    if i % 100 == 0:
        print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
        losses.append(loss)
    """

# save params
weights = model.get_params()
with open("weights.pkl","wb") as f:
    pickle.dump(weights, f)

with open("losses.pkl","wb") as f:
    pickle.dump(losses, f)

print("测试开始")
# Test  用训练集测试 不建议解注释 会很慢
# TRAIN SET ACC
# Y_pred = model.forward(X_train)
# result = np.argmax(Y_pred, axis=1) - Y_train
# result = list(result)
# print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(result.count(0)/X_train.shape[0]))

# TEST SET ACC 用训练集测试
Y_pred = model.forward(X_test)
result = np.argmax(Y_pred, axis=1) - Y_test
result = list(result)
print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(result.count(0)/X_test.shape[0]))
