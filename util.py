import numpy as np
import matplotlib.pyplot as plt
import random
'''as 
训练的组件
'''
def MakeOneHot(Y, D_out):
    """
    独热编码
    """
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

def draw_losses(losses):
    """
    plt刻画损失
    """
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

def get_batch(X, Y, batch_size):
    """
    随机抽取一个batch
    """
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]
