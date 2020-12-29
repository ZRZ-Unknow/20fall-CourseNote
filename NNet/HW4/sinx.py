import numpy as np
import argparse
from matplotlib import pyplot as plt
from Net import *

def generate_data_plot(N):
    X = np.random.uniform(-2*np.pi,2*np.pi,size=(N,1))
    X = np.sort(X,axis=0)
    Y = np.sin(X)
    fig = plt.figure()
    plt.plot(X,Y)
    plt.title('y=sin(x)')
    plt.show()
    return X, Y

def generate_data(N):
    X = np.random.uniform(-2*np.pi,2*np.pi,size=(N,1))
    Y = np.sin(X)
    return X, Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter_nums',type=int,default=500)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--lr',type=float,default=0.003)
    parser.add_argument('--hidden_dim',type=int,default=5)
    parser.add_argument('--hidden_layer_num',type=int,default=1)
    parser.add_argument('--input_dim',type=int,default=1)
    parser.add_argument('--output_dim',type=int,default=1)
    parser.add_argument('--activation',type=str,default='relu')
    parser.add_argument('--output_activation',type=str,default='tanh')
    parser.add_argument('--data_nums',type=int,default=100)
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    X, Y = generate_data(args.data_nums)
    nn = NN(args.input_dim, args.hidden_dim, args.output_dim, args.activation, args.output_activation, args.hidden_layer_num, args.lr)
    criterion = MSELoss()
    nn.train(X,Y,args.iter_nums,criterion)
    
    X_sorted = np.sort(X,axis=0)
    pred = nn.predict(X_sorted)
    for i,layer in enumerate(nn.nnSequence):
        print(f"layer{i}",layer.w,layer.b)
    out = X
    for i,layer in enumerate(nn.nnSequence):
        out = layer(out)
        out = layer.activate(out,nn.activation if i==0 else nn.last_activation)
        np.savetxt(f'./data/layer{i}_output.txt',out)
    '''fig = plt.figure() 
    plt.plot(X_sorted, pred)
    plt.title('Iter '+str(args.iter_nums))
    plt.show()'''

if __name__=='__main__':
    main()
