import unittest

import numpy as np

from ndnn.graph import Graph
from ndnn.node import Embed, MDEmbed


class EmbedTest(unittest.TestCase):
    def testEmbedGrad(self):
        graph = Graph(None, None)
        idx = graph.input()
        embed = graph.input()

        idx.value = np.array([1, 0, 3, 2, 0])
        embed.value = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        ebd = Embed(idx, embed)

        graph.compute()

        ebd.grad = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1]])

        ebd.updateGrad()

        result = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 1, 0]])
        self.assertTrue(np.array_equal(result, embed.grad))


class MDEmbedTest(unittest.TestCase):
    def testMDEmbedCompute(self):
        graph = Graph(None, None)
        idx = graph.input()
        embed = graph.input()

        idx.value = np.array([[1, 0, 3, 2, 0], [2, 1, 0, 2, 3]])
        embed.value = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        ebd = MDEmbed(idx, embed)

        graph.compute()

        result = np.array([[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]],
                           [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]])

        self.assertTrue(np.array_equal(result, ebd.value))

    def testMDEmbedUpdateGrad(self):
        graph = Graph(None, None)
        idx = graph.input()
        embed = graph.input()

        idx.value = np.array([[1, 0, 3, 2, 0], [2, 1, 0, 2, 3]])
        embed.value = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        ebd = MDEmbed(idx, embed)

        graph.compute()

        ebd.grad = np.array([[[1, 1, 2, 0], [1, 0, 3, 0], [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]],
                             [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]])

        ebd.updateGrad()

        result = np.array([[3, 1, 4, 0], [1, 2, 3, 0], [1, 1, 3, 1], [1, 0, 1, 2]])

        self.assertTrue(np.array_equal(result, embed.grad))
