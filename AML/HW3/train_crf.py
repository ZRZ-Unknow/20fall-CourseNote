import argparse,random
import numpy as np
from model.linear_chain_crf import LinearChainCRF
from utils.trainer import Trainer
from utils.utils import *

def run(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_data, train_data_label, test_data, test_data_label, label_dict = load_data(args.shared_nums)\
                                                                           if args.not_use_c else load_data_c(args.shared_nums)
    print(f"Feature nums:{train_data[0].shape[-1]}, Label nums:{len(label_dict.keys())}")
    crf = LinearChainCRF(len(label_dict.keys()), train_data[0].shape[-1], label_dict)
    
    trainer = Trainer(args.lr, args.batch_size, args.epoch, args.lamda) 
    trainer.train(crf, train_data, train_data_label, test_data, test_data_label)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--lr', default=0.01, type=float)
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--epoch', default=100, type=int)
    p.add_argument('--not_use_c', action='store_true')
    p.add_argument('--lamda', default=0, type=float)
    p.add_argument('--shared_nums', default=1, type=int)

    args = p.parse_args()
    run(args)