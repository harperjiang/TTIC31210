import numpy as np
from ndnn.loss import Loss
from ndnn.loss import clip

class LogLoss(Loss):
    def __init__(self):
        super().__init__()

    '''
    Actual is of shape [B, L, M]
    Expect is of shape [B, L]
    Should return an gradient of shape [B, L, M]    
    '''

    def loss(self, actual, expect, fortest):
        # The average loss is averaged to each slice
        all_batch_size = np.product(expect.shape)

        xflat = actual.reshape(-1)
        iflat = expect.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        idx = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        fetch = xflat[idx].reshape(expect.shape)
        clipval = np.maximum(fetch, clip)

        if not fortest:
            # Compute Gradient
            slgrad = -np.ones_like(expect) / (clipval * all_batch_size)
            self.grad = np.zeros_like(actual)
            self.grad.reshape(-1)[idx] = slgrad

        # Accuracy for classification is the number of corrected predicted items
        predict = np.argmax(actual, axis=-1)
        self.acc = np.equal(predict, expect).sum()

        return -np.log(clipval).mean()
