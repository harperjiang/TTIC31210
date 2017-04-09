import numpy as np

dt = np.float64


def diff(x, y):
    xs = np.array(x.shape)
    ys = np.array(y.shape)
    pad = len(xs) - len(ys)
    if pad > 0:
        ys = np.pad(ys, [[pad, 0]], 'constant')
    elif pad < 0:
        xs = np.pad(xs, [[-pad, 0]], 'constant')
    os = np.maximum(xs, ys)
    xred = tuple([idx for idx in np.where(xs < os)][0])
    yred = tuple([idx for idx in np.where(ys < os)][0])
    return xred, yred


class Node(object):
    def __init__(self, inputs):
        if len(inputs) > 0:
            context = inputs[0].context
            context.attach_node(self)
            self.context = context

    def forward(self):
        self.value = self.compute()
        self.grad = dt(0)

    def backward(self):
        self.updateGrad()


class Input(Node):
    def __init__(self, context, x=None):
        if x is not None:
            Node.__init__(self, [x])
        else:
            Node.__init__(self, [])
        self.x = x
        self.context = context
        context.attach_node(self)

    def compute(self):
        if self.x is not None:
            return self.x.value
        else:
            return self.value

    def updateGrad(self):
        if self.x is not None:
            self.x.grad += self.grad


class Param(Input):
    def __init__(self, context):
        Input.__init__(self, context, None)
        self.env = {}


class Add(Node):
    def __init__(self, l, r):
        super(Add, self).__init__([l, r])
        self.left = l
        self.right = r

    def compute(self):
        return self.left.value + self.right.value

    def updateGrad(self):
        xdiff, ydiff = diff(self.left.value, self.right.value)

        self.left.grad += np.reshape(np.sum(self.grad, axis=xdiff, keepdims=True),
                                     self.left.value.shape)

        self.right.grad += np.reshape(np.sum(self.grad, axis=ydiff, keepdims=True),
                                      self.right.value.shape)


class Mul(Node):
    def __init__(self, l, r):
        super(Mul, self).__init__([l, r])
        self.left = l
        self.right = r

    def compute(self):
        return self.left.value * self.right.value

    def updateGrad(self):
        xdiff, ydiff = diff(self.left.value, self.right.value)
        self.left.grad += np.reshape(np.sum(self.grad * self.right.value, axis=xdiff, keepdims=True),
                                     self.left.value.shape)
        self.right.grad += np.reshape(np.sum(self.grad * self.left.value, axis=ydiff, keepdims=True),
                                      self.right.value.shape)


class Dot(Node):  # Matrix multiply (fully-connected layer)
    def __init__(self, x, y):
        super(Dot, self).__init__([x, y])
        self.x = x
        self.y = y

    def compute(self):
        return np.matmul(self.x.value, self.y.value)

    def updateGrad(self):
        self.x.grad += np.matmul(self.y.value, self.grad.T).T
        self.y.grad += np.matmul(self.x.value.T, self.grad)


class Sigmoid(Node):
    def __init__(self, x):
        super(Sigmoid, self).__init__([x])
        self.x = x

    def compute(self):
        return 1. / (1. + np.exp(-self.x.value))

    def updateGrad(self):
        self.x.grad += self.grad * self.value * (1. - self.value)


class Tanh(Node):
    def __init__(self, x):
        super(Tanh, self).__init__([x])
        self.input = x

    def compute(self):
        x_exp = np.exp(self.input.value)
        x_neg_exp = np.exp(-self.input.value)

        return (x_exp - x_neg_exp) / (x_exp + x_neg_exp)

    def updateGrad(self):
        self.input.grad += self.grad * (1 - self.value * self.value)


class ReLU(Node):
    def __init__(self, x):
        super(ReLU, self).__init__([x])
        self.x = x

    def compute(self):
        return np.maximum(self.x.value, 0)

    def updateGrad(self):
        self.x.grad += self.grad * (self.value > 0)


class LeakyReLU(Node):
    def __init__(self, x):
        super(LeakyReLU, self).__init__([x])
        self.x = x

    def compute(self):
        return np.maximum(self.x.value, 0.01 * self.x.value)

    def updateGrad(self):
        self.x.grad += self.grad * np.maximum(0.01, self.value > 0)


class SoftMax(Node):
    def __init__(self, x):
        super(SoftMax, self).__init__([x])
        self.x = x

    def compute(self):
        lmax = np.max(self.x.value, axis=-1, keepdims=True)
        ex = np.exp(self.x.value - lmax)
        return ex / np.sum(ex, axis=-1, keepdims=True)

    def updateGrad(self):
        gvdot = np.matmul(self.grad[..., np.newaxis, :], self.value[..., np.newaxis]).squeeze(-1)
        self.x.grad += self.value * (self.grad - gvdot)


class Concat(Node):
    def __init__(self, x, y):
        super(Concat, self).__init__([x, y])
        self.x = x
        self.y = y

    def compute(self):
        return np.concatenate((self.x.value, self.y.value), axis=1)

    def updateGrad(self):
        dim_x = self.x.value.shape[1]
        dim_y = self.y.value.shape[1]

        self.x.grad += self.grad[:, 0:dim_x]
        self.y.grad += self.grad[:, dim_x:dim_x + dim_y]


class Collect(Node):
    def __init__(self, nodes):
        super(Collect, self).__init__(nodes)
        self.nodes = nodes

    def compute(self):
        withNewAxis = [n.value[np.newaxis, :] for n in self.nodes]
        return np.concatenate(withNewAxis, 0)

    def updateGrad(self):
        idx = 0
        for n in self.nodes:
            n.grad += self.grad[idx, :]
            idx += 1

class Average(Node):
    def __init__(self, x):
        super(Average, self).__init__([x])
        self.x = x
    
    def compute(self):
        if type(self.x.value) is np.ndarray:
            return self.x.value.mean(axis=0)
        else:
            list_mean = [item.mean(axis=0) for item in self.x.value]
            return np.array(list_mean)
    
    def updateGrad(self):
        if type(self.x.value) is np.ndarray:
            self.x.grad += np.repeat(self.grad.reshape([1, -1]),
                                     self.x.shape[0], axis=0)
        else:
            list_size = self.grad.shape[0]
            grad_list = [self.grad[i, :].reshape(1, -1) for i in range(list_size)]
            self.x.grad = []
            for i, grad in enumerate(grad_list):
                size = self.x.value[i].shape[0]
                self.x.grad.append(np.repeat(grad.reshape(1, -1), size, axis=0)) 
            
class Embed(Node):
    def __init__(self, idx, w2v):
        super(Embed, self).__init__([idx, w2v])
        self.idx = idx
        self.w2v = w2v

    def compute(self):
        hidden_dim = self.w2v.value.shape[1]
        
        if type(self.idx.value) is np.ndarray:
            return self.w2v.value[np.int32(self.idx.value), :].reshape(-1, hidden_dim)
        else:
            return [self.w2v.value[np.int32(i), :].reshape(-1, hidden_dim) for i in self.idx.value]
        
    def updateGrad(self):
        if type(self.idx.value) is np.ndarray:
            grad = np.zeros_like(self.w2v.value)
            grad[np.int32(self.idx.value), :] += self.grad
            self.w2v.grad += grad
        else:
            grad = np.zeros_like(self.w2v.value)
            for i, item in enumerate(self.idx.value):
                grad[np.int32(item),:]+= self.grad[i]
            self.w2v.grad += grad

class ArgMax(Node):
    def __init__(self, x):
        super(ArgMax, self).__init__([x])
        self.x = x

    def compute(self):
        return np.argmax(self.x.value)

    def backward(self):
        pass
