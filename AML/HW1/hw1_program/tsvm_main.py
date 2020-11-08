import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt
import random,copy

random.seed(3)

class TSVM:
    def __init__(self):
        self.svm = svm.SVC(kernel='linear')
    
    def fit(self, X_l, y, X_u):
        """
        训练函数
        :param X_l: 有标记数据的特征
        :param y: 有标记数据的标记
        :param X_u: 无标记数据的特征
        """
        self.svm.fit(X_l,y)
        y_u = self.svm.predict(X_u)
        num_p = y_u[y_u>0].shape[0]
        C_l ,C_u = 1, 1e-3
        X = np.vstack((X_l,X_u))
        Y = np.hstack((y,y_u))
        weight = np.ones(X.shape[0])
        weight[len(X_l):] = C_u
        assert(X.shape[0]==X_l.shape[0]+X_u.shape[0])
        assert(Y.shape[0]==y.shape[0]+y_u.shape[0])
        while C_u < C_l:
            self.svm.fit(X,Y,sample_weight=weight)
            while True:
                margin = self.svm.decision_function(X_u)
                slack_variable = 1-margin*y_u
                positive_index, negetive_index = [i for i in range(y_u.shape[0]) if y_u[i]>0], \
                                                 [i for i in range(y_u.shape[0]) if y_u[i]<0]
                positive_sv, negetive_sv = slack_variable[positive_index], slack_variable[negetive_index]
                if positive_sv.size==0 or negetive_sv.size==0:
                    target = positive_sv if negetive_sv.size==0 else negetive_sv
                    sv_i = np.max(target)
                    if sv_i>2:
                        i = np.where(slack_variable==sv_i)[0][0]
                        y_u[i] *= -1
                        Y = np.hstack((y,y_u))
                        self.svm.fit(X,Y,sample_weight=weight)
                        continue
                    else:
                        break
                sv_i, sv_j = np.max(positive_sv), np.max(negetive_sv)
                if sv_i>0 and sv_j>0 and sv_i+sv_j>2:
                    i, j = np.where(slack_variable==sv_i)[0][0],np.where(slack_variable==sv_j)[0][0]
                    y_u[i] *= -1
                    y_u[j] *= -1
                    Y = np.hstack((y,y_u))
                    self.svm.fit(X,Y,sample_weight=weight)
                else:
                    break
                
            C_u = min(2*C_u,C_l)
            weight[len(X_l):] = C_u

    def predict(self, X):
        """
        预测函数
        :param X: 预测数据的特征
        :return: 数据对应的预测值
        """
        return self.svm.predict(X)
    
    def decision_function(self, X):
        return self.svm.decision_function(X)

def load_data():
    label_X = np.loadtxt('label_X.csv', delimiter=',')
    label_y = np.loadtxt('label_y.csv', delimiter=',').astype(np.int)
    unlabel_X = np.loadtxt('unlabel_X.csv', delimiter=',')
    unlabel_y = np.loadtxt('unlabel_y.csv', delimiter=',').astype(np.int)
    test_X = np.loadtxt('test_X.csv', delimiter=',')
    test_y = np.loadtxt('test_y.csv', delimiter=',').astype(np.int)
    label_y[label_y==0] = -1
    unlabel_y[unlabel_y==0] = -1
    test_y[test_y==0] = -1 
    return label_X, label_y, unlabel_X, unlabel_y, test_X, test_y


def plot_roc(y_true,y_score,title):
    fpr, tpr, thr = roc_curve(y_true,y_score)
    auc_score = roc_auc_score(y_true,y_score)
    plt.plot(fpr,tpr,label='auc:%f'%auc_score)
    plt.legend(loc='lower right')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title(title)
    plt.show()

def accuracy(y_true,y_pred):
    return (y_true==y_pred).mean()

def main_loop():
    label_X, label_y, unlabel_X, unlabel_y, test_X, test_y = load_data()
    tsvm = TSVM()
    tsvm.fit(label_X, label_y, unlabel_X)
    
    unlabel_y_pred = tsvm.predict(unlabel_X)
    unlabel_y_score = tsvm.decision_function(unlabel_X)
    test_y_pred = tsvm.predict(test_X)
    test_y_score = tsvm.decision_function(test_X)
    
    #plot_roc(unlabel_y,unlabel_y_score,"roc curve of unlabel_y")
    #plot_roc(test_y,test_y_score,"roc curve of test_y")
    print("===================no LVW===================") 
    print("acc of unlabel_X:",accuracy(unlabel_y,unlabel_y_pred))  
    print("acc of test_X:",accuracy(test_y,test_y_pred))
    print("============================================")

def LVW(T):
    '''Implementation of LVW algorithm'''
    label_X, label_y, unlabel_X, unlabel_y, test_X, test_y = load_data()
    N = label_X.shape[1]
    index_all = [i for i in range(N)]
    
    d = N
    acc = -1
    best_index = index_all
    t = 0
    best_tsvm = None
    tsvm = TSVM()
    while t<T:
        m = random.randint(1,N)
        subset_index = random.sample(index_all,m)
        subset_index.sort()
        subset_label = label_X[:,subset_index]
        subset_unlabel = unlabel_X[:,subset_index]
        subset_test = test_X[:,subset_index]
        d_ = len(subset_index)
        
        tsvm.fit(subset_label, label_y, subset_unlabel)
        acc1 = accuracy(test_y,tsvm.predict(subset_test))

        if acc1>acc or (acc1==acc and d_<d):
            t = 0
            d = d_
            acc = acc1
            best_index = subset_index
            best_tsvm = copy.deepcopy(tsvm)
        else:
            t += 1

    unlabel_y_pred = best_tsvm.predict(unlabel_X[:,best_index])
    unlabel_y_score = best_tsvm.decision_function(unlabel_X[:,best_index])
    test_y_pred = best_tsvm.predict(test_X[:,best_index])
    test_y_score = best_tsvm.decision_function(test_X[:,best_index])
    
    print("==================use LVW===================")
    print("acc of unlabel_X:",accuracy(unlabel_y,unlabel_y_pred))
    print("acc of test_X:",accuracy(test_y,test_y_pred))
    print("best subset index of features:",best_index)
    print("============================================")
    
    plot_roc(unlabel_y,unlabel_y_score,"roc curve of unlabel_y")
    plot_roc(test_y,test_y_score,"roc curve of test_y")
    
if __name__=="__main__":
    main_loop()
    LVW(40)