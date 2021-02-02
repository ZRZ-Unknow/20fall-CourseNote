import random 
import matplotlib.pyplot as plt
import numpy as np
import time

class Trainer(object):
    def __init__(self, lr, batch_size, epoch, lamda):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.lamda = lamda

    def gradient_ascent(self, model, batch_x, batch_y):
        *grad, = model.compute_batch_grad(batch_x, batch_y, self.lamda)
        model.fit_params(self.lr, *grad)
    
    def train(self, model, train_x, train_y, test_x, test_y):
        order = [i for i in range(len(train_x))]
        ll, acc = [], []
        start_time = time.time()
        for i in range(self.epoch):
            random.shuffle(order)
            for j in range(0, len(order), self.batch_size):
                tmp = order[j: j+self.batch_size]
                batch_x = [train_x[k] for k in tmp]
                batch_y = [train_y[k] for k in tmp]
                self.gradient_ascent(model, batch_x, batch_y)
            ll.append(model.compute_batch_ll(train_x, train_y))
            acc.append(self.eval(model, test_x, test_y))
            print(f"iter:{i},loglikelihood:{ll[-1]}, acc:{acc[-1]}")
        total_time = time.time()-start_time
        print(f"Time:{total_time}")
        tmp = np.array([ll, acc])
        plt.title('LogLikeliHood')
        plt.xlabel('iterate times')
        plt.ylabel('loglikelihood')
        plt.plot(range(len(ll)), ll)
        plt.show()
        plt.title('Accuracy')
        plt.xlabel('iterate times')
        plt.ylabel('accuracy')
        plt.plot(range(len(acc)), acc)
        plt.show()

    def eval(self, model, test_x, test_y):
        p1, p2 = 0, 0
        for i in range(len(test_x)):
            pred, *_ = model.inference(test_x[i])
            tmp = (np.array(pred) == np.array(test_y[i]))
            p1 += tmp.sum()
            p2 += tmp.shape[0]
        return p1/p2 
