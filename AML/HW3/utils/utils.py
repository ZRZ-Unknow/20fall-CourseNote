import numpy as np
import os
from bidict import bidict

def load_data(shared_nums):
    """ Return:
    data: list, each of it is a numpy array of shape (k,321)
    label: list, each of it is a numpy array of shape (k,)
    """
    train_path, test_path = './Dataset/train/', './Dataset/test/'
    train_data, train_data_label = [], []
    test_data, test_data_label = [], []
    label_dict = bidict() 
    label_num = 0
    for file in os.listdir(train_path):
        with open(train_path+file,'r') as f:
            p = f.read().split('\n')
            if '' in p:
                p.remove('')
            label_string = p[0]
            tmp_x, tmp_y = [], []
            for i in range(1,len(p)):
                label = label_string[i-1]
                if label not in label_dict.keys():
                    label_dict[label] = label_num
                    label_num += 1 
                tmp_y.append(label_dict[label])
                nums = [int(j) for j in p[i].split()]
                nums = nums[1:]
                nums_ = []
                for i in range(0, len(nums), shared_nums):
                    nums_.append( sum(nums[i:i+shared_nums]))
                tmp_x.append(nums_)
            train_data.append(np.array(tmp_x,dtype=int))
            train_data_label.append(np.array(tmp_y,dtype=int))
    for file in os.listdir(test_path):
        with open(test_path+file,'r') as f:
            p = f.read().split('\n')
            if '' in p:
                p.remove('')
            assert len(p[0])==len(p)-1
            label_string = p[0]
            tmp_x, tmp_y = [], []
            for i in range(1,len(p)):
                label = label_string[i-1]
                tmp_y.append(label_dict[label])
                nums = [int(j) for j in p[i].split()]
                nums = nums[1:]
                nums_ = []
                for i in range(0, len(nums), shared_nums):
                    nums_.append( sum(nums[i:i+shared_nums]))
                tmp_x.append(nums_)
            test_data.append(np.array(tmp_x,dtype=int))
            test_data_label.append(np.array(tmp_y,dtype=int))
    return train_data, train_data_label, test_data, test_data_label, label_dict

def load_data_c(shared_nums):
    train_path, test_path = './Dataset/train/', './Dataset/test/'
    train_data, train_data_label = [], []
    test_data, test_data_label = [], []
    label_dict = bidict() 
    label_num = 0
    for file in os.listdir(train_path):
        with open(train_path+file,'r') as f:
            p = f.read().split('\n')
            if '' in p:
                p.remove('')
            label_string = p[0]
            tmp_x, tmp_y = [], []
            for i in range(1,len(p)):
                label = label_string[i-1]
                if label not in label_dict.keys():
                    label_dict[label] = label_num
                    label_num += 1 
                tmp_y.append(label_dict[label])
                nums = [int(j) for j in p[i].split()]
                nums = nums[1:]
                nums_ = []
                for i in range(0, len(nums), shared_nums):
                    nums_.append( sum(nums[i:i+shared_nums]))
                nums = nums_
                add = np.zeros(10)
                add[label_dict[label]] = 1
                nums = np.hstack([nums,add])
                tmp_x.append(nums)
            train_data.append(np.array(tmp_x,dtype=int))
            train_data_label.append(np.array(tmp_y,dtype=int))
    for file in os.listdir(test_path):
        with open(test_path+file,'r') as f:
            p = f.read().split('\n')
            if '' in p:
                p.remove('')
            assert len(p[0])==len(p)-1
            label_string = p[0]
            tmp_x, tmp_y = [], []
            for i in range(1,len(p)):
                label = label_string[i-1]
                tmp_y.append(label_dict[label])
                nums = [int(j) for j in p[i].split()]
                nums = nums[1:]
                nums_ = []
                for i in range(0, len(nums), shared_nums):
                    nums_.append( sum(nums[i:i+shared_nums]))
                nums = nums_
                add = np.zeros(10)
                add[label_dict[label]] = 1
                nums = np.hstack([nums,add])
                tmp_x.append(nums)
            test_data.append(np.array(tmp_x,dtype=int))
            test_data_label.append(np.array(tmp_y,dtype=int))
    return train_data, train_data_label, test_data, test_data_label, label_dict