import numpy as np
import matplotlib.pyplot as plt
from Net import *
import argparse

def generate_data(N, r, w, d):
    theta1 = np.random.uniform(0, np.pi, size=N)
    theta2 = np.random.uniform(-np.pi, 0, size=N)
    w1 = np.random.uniform(-w/2, w/2, size=N)
    w2 = np.random.uniform(-w/2, w/2, size=N)
    one = np.ones_like(theta1)
    zeros = np.zeros_like(theta1) 
    data_A = np.array([(r+w1)*np.cos(theta1), (r+w1)*np.sin(theta1), one]).T
    data_B = np.array([r + (r+w2)*np.cos(theta2), -d + (r+w2)*np.sin(theta2), zeros]).T
    return data_A, data_B
 
class Data:
    def __init__(self, n, r=10, w=6, d=-4):
        self.n = n          
        self.r = r         
        self.w = w         
        self.d = d          
        self.data_A = []    
        self.data_B = []    
        self.data_AB = []   
    
    def get_data(self):
        self.data_A, self.data_B = generate_data(self.n, self.r, self.w, self.d)
        all_data = np.vstack([self.data_A,self.data_B])
        np.random.shuffle(all_data)
        self.data_AB = all_data
        return all_data[:,:-1],all_data[:,-1:]
 
    def plot(self):
        fig = plt.figure()
        plt.scatter(self.data_A[:, 0], self.data_A[:, 1], marker='x')
        plt.scatter(self.data_B[:, 0], self.data_B[:, 1], marker='+')
        plt.title('Double Moon Data')
        plt.show()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter_nums',type=int,default=200)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--lr',type=float,default=0.0003)
    parser.add_argument('--hidden_dim',type=int,default=32)
    parser.add_argument('--hidden_layer_num',type=int,default=3)
    parser.add_argument('--input_dim',type=int,default=2)
    parser.add_argument('--output_dim',type=int,default=1)
    parser.add_argument('--activation',type=str,default='relu')
    parser.add_argument('--output_activation',type=str,default='sigmoid')
    parser.add_argument('--data_nums',type=int,default=2000)
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    data_set = Data(int(args.data_nums/2))
    X, Y = data_set.get_data()
    #data_set.plot()
    nn = NN(args.input_dim, args.hidden_dim, args.output_dim, args.activation, args.output_activation, args.hidden_layer_num, args.lr)
    criterion = CrossEntropyLoss()
    nn.train(X,Y,args.iter_nums,criterion)
    
    pred = nn.predict(X)
    pred = np.array(pred>0.5,dtype=int)
    acc = (pred==Y).mean()
    print(f'Acc:{acc}')
    index0 = [i for i in range(pred.shape[0]) if pred[i]==0]
    index1 = [i for i in range(pred.shape[0]) if pred[i]==1]
    fig = plt.figure()
    plt.scatter(X[index1,0],X[index1,1],marker='x')
    plt.scatter(X[index0,0],X[index0,1],marker='+')
    plt.title('Iter '+str(args.iter_nums))
    plt.show()

if __name__ == "__main__":
    main()