from ndnn.graph import Graph
from ndnn.init import Xavier, Zero
from ndnn.node import Concat, Sigmoid, Add, Dot, Tanh, Mul


class LSTMGraph(Graph):
    def __init__(self, loss, update, dict_size, hidden_dim):
        super().__init__(loss, update)

        self.h0 = self.input()
        self.c0 = self.input()

        self.embed = self.param_of([dict_size, hidden_dim], Xavier())
        self.wf = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.bf = self.param_of([hidden_dim], Zero())
        self.wi = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.bi = self.param_of([hidden_dim], Zero())
        self.wc = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.bc = self.param_of([hidden_dim], Zero())
        self.wo = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.bo = self.param_of([hidden_dim], Zero())
        self.v2c = self.param_of([hidden_dim, dict_size], Xavier())

        self.resetNum = len(self.nodes)

    def lstm_cell(self, x, h, c):
        concat = Concat(h, x)

        # Forget Gate
        f_gate = Sigmoid(Add(Dot(concat, self.wf), self.bf))
        # Input Gate
        i_gate = Sigmoid(Add(Dot(concat, self.wi), self.bi))
        # Temp Vars
        c_temp = Tanh(Add(Dot(concat, self.wc), self.bc))
        o_temp = Sigmoid(Add(Dot(concat, self.wo), self.bo))

        # Output
        c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
        h_next = Mul(o_temp, Tanh(c_next))
        return h_next, c_next
