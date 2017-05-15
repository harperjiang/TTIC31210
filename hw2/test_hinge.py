import unittest

import numpy as np

from ndnn.node import Node


class HingeLoss(Node):
    def __init__(self, actual, expect, negSamples):
        super().__init__([actual])
        self.actual = actual
        self.expect = expect
        self.negSamples = negSamples

    def compute(self):
        ytht = np.einsum('ij,ij->i', self.actual.value, self.expect.value)
        ypht = np.matmul(self.actual.value, self.negSamples.value.T)
        value = np.maximum(1 - ytht[:, np.newaxis] + ypht, 0)
        self.mask = value > 0
        return value.sum(axis=1)

    def updateGrad(self):
        gradscalar = self.grad[:, np.newaxis]
        multiplier = self.mask.sum(axis=1, keepdims=True)
        self.actual.grad += gradscalar * (
            np.matmul(self.mask, self.negSamples.value) - multiplier * self.expect.value)
        self.expect.grad += gradscalar * multiplier * (-self.actual.value)
        self.negSamples.grad += np.einsum('br,bh->rh', self.mask, gradscalar * self.actual.value)


class HingePredict(Node):
    def __init__(self, actual, allEmbed, negSamples):
        super().__init__([actual])
        self.actual = actual
        self.allEmbed = allEmbed
        self.negSamples = negSamples

    def compute(self):
        # Find the one with smallest loss as prediction

        # All_embed has size D,H
        # Actual has size B,H
        # Neg Samples has size R,H

        d = self.allEmbed.value.shape[0]
        r = self.negSamples.value.shape[0]

        bd = np.einsum('bh,dh->bd', self.actual.value, self.allEmbed.value)
        br = np.einsum('bh,rh->br', self.actual.value, self.negSamples.value)

        bdr1 = np.repeat(bd[:, :, np.newaxis], r, axis=2)
        bdr2 = np.repeat(br[:, np.newaxis, :], d, axis=1)

        predict = np.maximum(1 - bdr1 + bdr2, 0).sum(axis=2).argmin(axis=1)
        return predict

    def updateGrad(self):
        raise Exception("Operation not supported")


class DummyContext:
    def attach_node(self, node):
        pass


class DummyNode:
    def __init__(self, value):
        self.value = value
        self.grad = np.float64(0)
        self.context = DummyContext()


class HingeLossTest(unittest.TestCase):
    def testHingeLossCompute(self):
        actual = DummyNode(np.array([[1, 1, 1], [2, 2, 3]]))
        expect = DummyNode(np.array([[2, 1, 2], [0, 1, -1]]))
        negSamples = DummyNode(np.array([[1, 0, 1], [2, 1, 2], [1, 3, 1]]))

        hloss = HingeLoss(actual, expect, negSamples)

        result = hloss.compute()

        self.assertTrue(np.array_equal(result, np.array([2, 34])))
        self.assertTrue(np.array_equal(hloss.mask, np.array([[0, 1, 1], [1, 1, 1]])))

    def testHingeLossGradient(self):
        actual = DummyNode(np.array([[1, 1, 1], [2, 2, 3]]))
        expect = DummyNode(np.array([[2, 1, 2], [0, 1, -1]]))
        negSamples = DummyNode(np.array([[1, 0, 1], [2, 1, 2], [1, 3, 1]]))

        hloss = HingeLoss(actual, expect, negSamples)
        hloss.compute()

        hloss.grad = np.array([0.1, 0.2])

        hloss.updateGrad()

        self.assertTrue(np.isclose(actual.grad, np.array([[-0.1, 0.2, -0.1], [0.8, 0.2, 1.4]]), 0.0001).all())
        self.assertTrue(np.isclose(expect.grad, np.array([[-0.2, -0.2, -0.2], [-1.2, -1.2, -1.8]]), 0.0001).all())
        self.assertTrue(np.isclose(negSamples.grad,
                                   np.array([[0.4, 0.4, 0.6], [0.5, 0.5, 0.7], [0.5, 0.5, 0.7]]), 0.0001).all())


class HingePredictTest(unittest.TestCase):
    def testHingePredictCompute(self):
        actual = DummyNode(np.array([[1, 3, 2, 5], [2, 2, 0, 1]]))
        allEmbed = DummyNode(np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1]]))
        negSample = DummyNode(np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1]]))

        hpredict = HingePredict(actual, allEmbed, negSample)

        result = hpredict.compute()

        self.assertTrue(np.array_equal(result, np.array([3, 3])))
