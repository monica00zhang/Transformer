import numpy as np

def softmax(x):
    """ 使得概率分布，每个数是非负，且相加为1
    np.exp e的x次幂是为了反向传播的可导性质，而且能放数值之间的差异，且曲线是平滑可导 """
    shifted_v = x-np.max(x)
    exp_v = np.exp(shifted_v)
    return exp_v/np.sum(exp_v)

""" non-linear func """
def sigmoid(x):
    x = np.array(x)
    return 1/(1+np.exp(-x))


def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))


def relu(x):
    return max(0,x)






