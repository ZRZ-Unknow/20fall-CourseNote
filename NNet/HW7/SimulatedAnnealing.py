import math
import random
import matplotlib.pyplot as plt
import argparse


def func(x1, x2):
    return 0.2 + math.pow(x1,2) + math.pow(x2,2) - 0.1*math.cos(6*math.pi*x1) - 0.1*math.cos(6*math.pi*x2)

def calc(x1_set, x2_set):
    y_min,y_max = 10000, -10000
    y_min_x, y_max_x = (None, None), (None, None)
    for i in x1_set:
        for j in x2_set:
            y = func(i,j)
            if y < y_min:
                y_min = y
                y_min_x = (i, j)
            if y > y_max:
                y_max = y
                y_max_x = (i, j)
    print(f"Min of y:{y_min}, x:{y_min_x}")

def get_range(index_x1, index_x2, x1_set, x2_set, neighbor_num):
    index_x1_low = 0 if (index_x1-neighbor_num)<0 else index_x1-neighbor_num
    index_x1_high = len(x1_set) if (index_x1+neighbor_num)>len(x1_set) else index_x1+neighbor_num
    index_x2_low = 0 if (index_x2-neighbor_num)<0 else index_x2-neighbor_num
    index_x2_high = len(x2_set) if (index_x2+neighbor_num)>len(x2_set) else index_x2+neighbor_num
    return index_x1_low, index_x1_high, index_x2_low, index_x2_high

def plot_y(y_list):
    fig = plt.figure()
    x_index = [i for i in range(len(y_list))]
    plt.plot(x_index, y_list, color='r', linestyle='--', marker='*', linewidth=1.0, label='Objective')
    plt.axis([0, 60, -0.1, 3])
    plt.legend(loc='upper right')
    plt.show()

def plot_x(x1_list, x2_list):
    fig = plt.figure()
    x_index = [i for i in range(len(x1_list))]
    plt.plot(x_index, x1_list, color='b', linestyle='--', marker='*', linewidth=1.0, label='x1')
    plt.plot(x_index, x2_list, color='g', linestyle='--', marker='*', linewidth=1.0, label='x2')
    plt.axis([0, 60, -2, 2])
    plt.legend(loc='upper right')
    plt.show()

class SA(object):
    def __init__(self, T, T_final, inner_iters, gamma, neighbor_num):
        self.T = T
        self.T_final = T_final
        self.inner_iters = inner_iters
        self.gamma = gamma
        self.neighbor_num = neighbor_num

    def train(self, x1, x2, x1_set, x2_set):
        steps = 0
        x1_list, x2_list, y_list = [], [], []
        while self.T >= self.T_final:
            for i in range(self.inner_iters):
                y = func(x1, x2)
                index_x1, index_x2 = x1_set.index(x1), x2_set.index(x2)
                index_x1_low, index_x1_high, index_x2_low, index_x2_high = get_range(index_x1, index_x2, x1_set, x2_set, self.neighbor_num)
                x1_new = random.choice(x1_set[index_x1_low : index_x1_high])
                x2_new = random.choice(x2_set[index_x2_low : index_x2_high])
                y_new = func(x1_new, x2_new)
                if y_new < y:
                    x1 = x1_new
                    x2 = x2_new
                else:
                    prob = math.exp((y-y_new)/self.T)
                    if prob > random.uniform(0,1):
                        x1 = x1_new
                        x2 = x2_new
            x1_list.append(x1)
            x2_list.append(x2)
            y_list.append(func(x1,x2))
            print(f"timesteps:{steps}, x1:{x1}, x2:{x2}, y:{func(x1,x2)}")
            steps += 1
            self.T *= self.gamma
        return y_list, x1_list, x2_list

def run(args):
    random.seed(args.seed)
    x1_set = [i/100 for i in range(-100,101)]
    x2_set = [i/100 for i in range(-100,101)]
    x1, x2 = 0.8, -0.5

    sa = SA(args.T, args.T_final, args.inner_iters, args.gamma, args.neighbor_num)
    y_list, x1_list, x2_list = sa.train(x1, x2, x1_set, x2_set)
    calc(x1_set, x2_set)
    plot_y(y_list)
    plot_x(x1_list, x2_list)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed',default=0,type=int)
    p.add_argument('--T',default=2,type=float)
    p.add_argument('--T_final',default=0.01,type=float)
    p.add_argument('--inner_iters',default=50,type=int)
    p.add_argument('--neighbor_num',default=15,type=int)
    p.add_argument('--gamma',default=0.9,type=float)
    args = p.parse_args()
    run(args)