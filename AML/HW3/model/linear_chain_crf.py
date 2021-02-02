import argparse,random
import numpy as np
from .graph import Graph

class LinearChainCRF(Graph):
    def __init__(self, label_num, feature_num, label_dict):
        super().__init__(label_num, feature_num)
        self.label_dict = label_dict

    def compute_message(self, seq_x):
        self.potI = seq_x
        forward_message, backward_message = np.empty((seq_x.shape[0]-1, self.label_num)), np.empty((seq_x.shape[0]-1, self.label_num))
        for i in range(forward_message.shape[0]):
            forward_message[i] = self.potI[i].dot(self.potP.T)*(1 if i==0 else forward_message[i-1])
        for i in range(backward_message.shape[0]-1,-1,-1):
            backward_message[i] = self.potI[i+1].dot(self.potP.T)*(1 if i==backward_message.shape[0]-1 else backward_message[i+1])
        return forward_message, backward_message
    
    def compute_fweight(self, seq_x, seq_y, apply_exp=False):
        res = seq_x.dot(self.thetaI.T)[:,seq_y].sum()
        res += self.thetaP[seq_y[:-1], seq_y[1:]].sum()
        if apply_exp:
            return np.exp(res)
        return res

    def compute_pair_dist(self, seq_x, fm, bm):
        res = []
        N = seq_x.shape[0]-1
        for i in range(N):
            left = (self.potI[i] * (1 if i==0 else fm[i]) )[np.newaxis,:]
            right = (self.potI[i+1] * (1 if i==N-1 else bm[i+1]) )[:,np.newaxis]
            tmp = left * self.potP * right
            res.append((tmp/tmp.sum())[np.newaxis,:])
        return np.concatenate(res)
    
    def compute_marginal_dist(self, seq_x, fm, bm):
        res = []
        N = seq_x.shape[0]
        for i in range(N):
            tmp = self.potI[i] * (1 if i==0 else fm[i-1]) * (1 if i==N-1 else bm[i])
            res.append((tmp/tmp.sum())[np.newaxis,:])
        return np.concatenate(res)
    
    def compute_grad(self, seq_x, seq_y, lamda):
        grad_p, grad_i = np.zeros_like(self.thetaP), np.zeros_like(self.thetaI) 
        fm, fb = self.compute_message(seq_x)
        pair_dist = self.compute_pair_dist(seq_x, fm, fb)
        marginal_dist = self.compute_marginal_dist(seq_x,fm, fb)
        for i in range(0, seq_x.shape[0]-1):
            grad_p[seq_y[i], seq_y[i+1]] += 1
            grad_p -= pair_dist[i].T
        for i in range(seq_x.shape[0]):
            grad_i[seq_y[i]] += seq_x[i]
            grad_i -= marginal_dist[i].reshape(self.label_num,1) * seq_x[i]
        grad_p -= lamda*self.thetaP
        grad_i -= lamda*self.thetaI
        return grad_p, grad_i

    def compute_ll(self, seq_x, seq_y):
        *_, bm = self.compute_message(seq_x)
        logz = np.log((self.potI[0]*bm[0]).sum())
        ll = self.compute_fweight(seq_x, seq_y) - logz
        return ll
    
    def inference(self, seq_x):
        fm, bm = self.compute_message(seq_x)
        pair_dist = self.compute_pair_dist(seq_x, fm, bm) 
        marginal_dist = self.compute_marginal_dist(seq_x,  fm, bm)
        pred = []
        label = 0
        for i in range(seq_x.shape[0]):
            label = np.argmax((1 if i==0 else pair_dist[i-1,:,label]) * marginal_dist[i])
            pred.append(label)
        label_string = ""
        for label in pred:
            label_string += self.label_dict.inverse[label]
        return pred, label_string

    def fit_params(self, lr, grad_p, grad_i):
        self.thetaP += lr*grad_p
        self.thetaI += lr*grad_i






