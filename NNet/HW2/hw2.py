import numpy as np

share_prices = [55.22, 56.34, 55.52, 55.53, 56.94,
                58.88, 58.18, 57.09, 58.38, 58.54,
                57.72, 58.02, 57.81, 58.71, 60.84,
                61.08, 61.74, 62.16, 60.80, 60.87]
np.random.seed(1)

class NN():
    def __init__(self,N):
        self.w = np.random.rand(N)

    def fit(self,X,Y,alpha,max_iter_num):
        for _ in range(int(max_iter_num)):
            for x,y in zip(X,Y):
                pred = self.w.dot(x)
                delta_w = alpha*(y-pred)*x
                self.w += delta_w

    def params(self):
        return self.w

def get_data(N):
    x,y = [], []
    for i in range(0,len(share_prices)-N):
        x.append([share_prices[j] for j in range(i,i+N)]+[1])
        y.append(share_prices[i+N])
    return np.array(x),np.array(y)
    '''
def get_data(N):
    x,y = [], []
    for i in range(1,len(share_prices)-N+1):
        x.append([j for j in range(i,i+N)]+[1])
        y.append(share_prices[i+N-1])
    return np.array(x),np.array(y)'''

def main_loop(N):
    x,y = get_data(N)
    print(x,y)
    nn = NN(N+1)
    nn.fit(x,y,0.0001,1e4)
    print(x.shape,y.shape)
    w = nn.params()
    pred_y = x.dot(w)
    print("MSEloss:",np.sum((pred_y-y)**2)/y.shape[0])
    print("weight:",w)

main_loop(4)