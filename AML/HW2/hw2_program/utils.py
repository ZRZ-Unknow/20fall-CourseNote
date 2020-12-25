import numpy as np
import random


def load_data():
    data = []
    with open("./assets/news.txt") as f:
        for line in f.readlines():
            if line!='' and line!='\n':
                data.append(line.strip('\n'))
    return data

def choose_from_dist(d):
    s = sum(d)
    for i in range(len(d)):
        d[i] = d[i]/s*1.0
    r = random.random()
    index = -1
    while (r > 0):
        r = r - d[index]
        index = index + 1
    return index