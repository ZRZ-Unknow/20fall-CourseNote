import numpy as np
import argparse
from matplotlib import pyplot as plt
from Net import *

def generate_data(N):
    N = int(np.sqrt(N))
    X = np.random.uniform(-5,5,size=(N,2))
    X = np.sort(X,axis=0)
    x1, x2 = np.meshgrid(X[:,0],X[:,1])
    Y = np.sin(x1)-np.cos(x2)
    ax = plt.gca(projection='3d')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.title('y=sin(x1)-cos(x2)')
    ax.plot_surface(x1,x2,Y,cmap='rainbow')
    plt.show()
    X1, X2 = x1.reshape(-1,1), x2.reshape(-1,1)
    flatten_Y = Y.reshape(-1,1)
    flatten_X = np.hstack([X1,X2])
    return flatten_X, flatten_Y

#sigmoid 1e-2 tanh 1e-3 relu 1e-3
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter_nums',type=int,default=36)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--hidden_dim',type=int,default=16)
    parser.add_argument('--hidden_layer_num',type=int,default=1)
    parser.add_argument('--input_dim',type=int,default=2)
    parser.add_argument('--output_dim',type=int,default=1)
    parser.add_argument('--activation',type=str,default='sigmoid')
    parser.add_argument('--data_nums',type=int,default=10000)
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    X, Y = generate_data(args.data_nums)
    index = np.arange(0,X.shape[0])
    np.random.shuffle(index)
    train_X, train_Y = X[index], Y[index]
    nn = NN(args.input_dim, args.hidden_dim, args.output_dim, args.activation, args.hidden_layer_num, args.lr)
    criterion = MSELoss()
    nn.train(train_X,train_Y,args.iter_nums,criterion)
    
    pred = nn.predict(X)

    ax = plt.gca(projection='3d')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.title('iter '+str(args.iter_nums))
    N = int(np.sqrt(args.data_nums))
    x1, x2 = X[:,0].reshape(N,N), X[:,1].reshape(N,N)
    pred = pred.reshape(N,N)
    ax.plot_surface(x1,x2,pred,cmap='rainbow')
    plt.show()

if __name__=='__main__':
    main()