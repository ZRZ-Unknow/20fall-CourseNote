import numpy as np

def Tanh(x):
    return np.nan_to_num((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))

def TanhD(x):
    tmp = Tanh(x)
    return np.nan_to_num(1-np.multiply(tmp,tmp))

def Sigmoid(x):
    return np.nan_to_num(1/(1+np.exp(-x)))

def SigmoidD(x):
    return np.nan_to_num(np.multiply(x,1-x))

def Relu(x):
    return np.nan_to_num(np.maximum(0,x))

def ReluD(x):
    tmp = np.ones_like(x)
    tmp[x<=0] = 0
    return np.nan_to_num(tmp)

def Line(x):
    return x

def LineD(x):
    return np.ones_like(x)


class Linear(object):
    def __init__(self, input_dim, output_dim):
        ''' x: 1*input_dim, x.dot(w)+b: 1*output_dim'''
        self.w = np.random.randn(input_dim, output_dim)*0.1
        self.b = np.random.randn(1, output_dim)*0.1
        self.inputs = None
        self.O = None
        self.A = None
    
    def __call__(self, x):
        self.inputs = x
        res = np.dot(x, self.w) + self.b
        self.O = res
        return res
    
    def activate(self, x, activation):
        res = activation(x)
        self.A = res
        return res
        
    def update(self, delta_w, delta_b, lr):
        self.w -= lr*delta_w
        self.b -= lr*delta_b

class MSELoss(object):
    def __init__(self):
        self.loss = None
        self.lossD = None

    def __call__(self, x, targets):
        targets = targets.reshape(x.shape)
        assert(x.shape==targets.shape)
        self.loss = ((x-targets)**2/2).mean(axis=0,keepdims=True)
        self.lossD = (x-targets).mean(axis=0,keepdims=True)
        return self.loss, self.lossD


class NN(object):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, hidden_layer_num, lr):
        '''activation: sigmoid, tanh, or relu'''
        self.nnSequence = []
        self.hidden_layer_num = hidden_layer_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        if activation=='sigmoid':
            self.activation = Sigmoid
            self.activationD = SigmoidD
        elif activation == 'relu':
            self.activation = Relu
            self.activationD = ReluD
        elif activation == 'tanh':
            self.activation = Tanh
            self.activationD = TanhD
        self.last_activation = Line
        self.last_activationD = LineD
        last_dim = input_dim
        for i in range(hidden_layer_num):
            self.nnSequence.append(Linear(last_dim, hidden_dim))
            last_dim = hidden_dim
        self.nnSequence.append(Linear(last_dim,output_dim))

        assert(len(self.nnSequence)==hidden_layer_num+1)

    def forward(self, inputs):
        outputs = inputs 
        for i in range(self.hidden_layer_num+1):
            neuron = self.nnSequence[i]
            outputs = neuron(outputs)
            outputs = neuron.activate(outputs, self.activation if i<self.hidden_layer_num else self.last_activation)
        return outputs
    
    def backward(self, lossD):
        last_value = lossD.copy()
        for i in range(self.hidden_layer_num, -1, -1):
            neuron = self.nnSequence[i]
            tmp = np.multiply(last_value, self.activationD(neuron.A) if i<self.hidden_layer_num else self.last_activationD(neuron.A))
            delta_w = np.dot(neuron.inputs.T,tmp)
            delta_b = tmp
            assert(delta_w.shape==neuron.w.shape and delta_b.shape==neuron.b.shape)
            self.nnSequence[i].update(delta_w, delta_b, self.lr)
            last_value = tmp.dot(neuron.w.T)

    def fit(self, inputs, targets, criterion):
        outputs = self.forward(inputs) 
        loss, lossD = criterion(outputs, targets)
        self.backward(lossD)
        return loss

    def predict(self, x):
        pred = self.forward(x)
        return pred

    def train(self, X, Y, num_iters, criterion):
        for i_iter in range(num_iters):
            losses = 0
            for i in range(X.shape[0]):
                x, y = X[i].reshape(1,-1), Y[i].reshape(1,-1)
                loss = self.fit(x,y,criterion)
                losses += loss[0][0]
            print('iter {}, loss {}'.format(i_iter,losses/X.shape[0]))

    def train_batch(self,X, Y, num_iters, criterion):
        for i_iter in range(num_iters):
            loss = self.fit(X,Y,criterion)
            print('iter {}, loss {}'.format(i_iter,loss[0][0]/X.shape[0]))
        return losses/X.shape[0]
