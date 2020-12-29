import numpy as np
import matplotlib.pyplot as plt
from Net import *
import argparse
import pandas as pd

def load_data(feature_nums,output_nums):
    train_data = pd.read_csv('./data/optdigits.tra',header=None)
    test_data = pd.read_csv('./data/optdigits.tes',header=None)
    train_x, train_y_ = train_data[np.arange(feature_nums)].values, train_data[feature_nums].values
    test_x, test_y_ = test_data[np.arange(feature_nums)].values, test_data[feature_nums].values
    train_y = np.zeros((train_y_.shape[0], output_nums))
    test_y = np.zeros((test_y_.shape[0], output_nums))
    for i in range(train_y.shape[0]):
        label = train_y_[i]   
        train_y[i,label] = 1
    for i in range(test_y.shape[0]):
        label = test_y_[i]   
        test_y[i,label] = 1
    return train_x, train_y, test_x, test_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter_nums',type=int,default=100)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--hidden_dim',type=int,default=128)
    parser.add_argument('--hidden_layer_num',type=int,default=1)
    parser.add_argument('--input_dim',type=int,default=64)
    parser.add_argument('--output_dim',type=int,default=10)
    parser.add_argument('--activation',type=str,default='tanh')
    parser.add_argument('--output_activation',type=str,default='line')
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    train_x, train_y, test_x, test_y = load_data(args.input_dim,args.output_dim)
    nn = NN(args.input_dim, args.hidden_dim, args.output_dim, args.activation, args.output_activation, args.hidden_layer_num, args.lr)
    criterion = SoftmaxCrossEntropyLoss()
    nn.train(train_x, train_y, args.iter_nums, criterion)

    pred = Softmax(nn.predict(test_x))
    acc = (np.argmax(pred,axis=-1)==np.argmax(test_y, axis=-1)).mean()
    print(f"Iter {args.iter_nums}, Acc on test data {acc}")


if __name__ == '__main__':
    main()
    