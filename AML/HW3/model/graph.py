import numpy as np
from abc import ABC, abstractmethod

class Graph(ABC):
    def __init__(self, label_num, feature_num):
        self.label_num = label_num
        self.feature_num = feature_num
       
        self.thetaP = np.zeros((label_num, label_num))
        self.thetaI = np.zeros((label_num, feature_num))
        self.potentialI = None

    @property
    def potP(self):
        return np.exp(self.thetaP)

    @property
    def potI(self):
        return self.potentialI

    @potI.setter
    def potI(self, seq_x):
        self.potentialI = np.exp(seq_x.dot(self.thetaI.T))

    @abstractmethod
    def compute_ll(self, seq_x, seq_y):
        pass

    def compute_batch_ll(self, X, Y):
        batch_ll = 0
        for i in range(len(X)):
            batch_ll += self.compute_ll(X[i], Y[i])
        return batch_ll/len(X)

    @abstractmethod 
    def compute_grad(self, seq_x, seq_y, lamda):
        pass

    def compute_batch_grad(self, X, Y, lamda):
        grad_p, grad_i = [],[]
        for i in range(len(X)):
            tmp1, tmp2 = self.compute_grad(X[i], Y[i], lamda)
            grad_p.append(tmp1)
            grad_i.append(tmp2)
        grad_p = sum(grad_p)/len(grad_p)
        grad_i = sum(grad_i)/len(grad_i)
        return grad_p, grad_i

