import numpy as np

from ndnn.graph import Graph
from ndnn.init import Xavier, Zero
from ndnn.node import Concat, Sigmoid, Add, Dot, Tanh, Mul, Collect, Embed, SoftMax, Average


class MultiLSTMEncodeGraph(Graph):
    def __init__(self, loss, update, enc_dict_size, dec_dict_size, hidden_dim, num_line):
        super().__init__(loss, update)

        half_dim = int(hidden_dim / 2)

        self.enc_dict_size = enc_dict_size
        self.dec_dict_size = dec_dict_size
        self.hidden_dim = hidden_dim
        self.half_dim = half_dim
        self.num_line = num_line

        self.feh0 = self.input()
        self.fec0 = self.input()
        self.beh0 = self.input()
        self.bec0 = self.input()

        self.feembed = self.param_of([enc_dict_size, half_dim], Xavier())
        self.fewf = self.param_of([2 * half_dim, half_dim], Xavier())
        self.febf = self.param_of([half_dim], Zero())
        self.fewi = self.param_of([2 * half_dim, half_dim], Xavier())
        self.febi = self.param_of([half_dim], Zero())
        self.fewc = self.param_of([2 * half_dim, half_dim], Xavier())
        self.febc = self.param_of([half_dim], Zero())
        self.fewo = self.param_of([2 * half_dim, half_dim], Xavier())
        self.febo = self.param_of([half_dim], Zero())
        self.fev2c = self.param_of([half_dim, enc_dict_size], Xavier())

        self.beembed = self.param_of([enc_dict_size, half_dim], Xavier())
        self.bewf = self.param_of([2 * half_dim, half_dim], Xavier())
        self.bebf = self.param_of([half_dim], Zero())
        self.bewi = self.param_of([2 * half_dim, half_dim], Xavier())
        self.bebi = self.param_of([half_dim], Zero())
        self.bewc = self.param_of([2 * half_dim, half_dim], Xavier())
        self.bebc = self.param_of([half_dim], Zero())
        self.bewo = self.param_of([2 * half_dim, half_dim], Xavier())
        self.bebo = self.param_of([half_dim], Zero())
        self.bev2c = self.param_of([half_dim, enc_dict_size], Xavier())

        self.dembed = self.param_of([dec_dict_size, hidden_dim], Xavier())
        self.dwf = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbf = self.param_of([hidden_dim], Zero())
        self.dwi = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbi = self.param_of([hidden_dim], Zero())
        self.dwc = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbc = self.param_of([hidden_dim], Zero())
        self.dwo = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbo = self.param_of([hidden_dim], Zero())
        self.dv2c = self.param_of([hidden_dim, dec_dict_size], Xavier())

        self.resetNum = len(self.nodes)

    def fenc_lstm_cell(self, x, h, c):
        concat = Concat(h, x)

        # Forget Gate
        f_gate = Sigmoid(Add(Dot(concat, self.fewf), self.febf))
        # Input Gate
        i_gate = Sigmoid(Add(Dot(concat, self.fewi), self.febi))
        # Temp Vars
        c_temp = Tanh(Add(Dot(concat, self.fewc), self.febc))
        o_temp = Sigmoid(Add(Dot(concat, self.fewo), self.febo))

        # Output
        c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
        h_next = Mul(o_temp, Tanh(c_next))
        return h_next, c_next

    def benc_lstm_cell(self, x, h, c):
        concat = Concat(h, x)

        # Forget Gate
        f_gate = Sigmoid(Add(Dot(concat, self.bewf), self.bebf))
        # Input Gate
        i_gate = Sigmoid(Add(Dot(concat, self.bewi), self.bebi))
        # Temp Vars
        c_temp = Tanh(Add(Dot(concat, self.bewc), self.bebc))
        o_temp = Sigmoid(Add(Dot(concat, self.bewo), self.bebo))

        # Output
        c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
        h_next = Mul(o_temp, Tanh(c_next))
        return h_next, c_next

    def dec_lstm_cell(self, x, h, c):
        concat = Concat(h, x)

        # Forget Gate
        f_gate = Sigmoid(Add(Dot(concat, self.dwf), self.dbf))
        # Input Gate
        i_gate = Sigmoid(Add(Dot(concat, self.dwi), self.dbi))
        # Temp Vars
        c_temp = Tanh(Add(Dot(concat, self.dwc), self.dbc))
        o_temp = Sigmoid(Add(Dot(concat, self.dwo), self.dbo))

        # Output
        c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
        h_next = Mul(o_temp, Tanh(c_next))
        return h_next, c_next

    def build_graph(self, batch):
        enc_data = batch.data
        dec_data = batch.expect
        self.reset()
        bsize = 1
        enc_length = enc_data.shape[1]
        dec_length = dec_data.shape[0]

        outputs = []

        hcollect = []
        ccollect = []

        self.feh0.value = np.zeros([bsize, self.half_dim])
        self.fec0.value = np.zeros([bsize, self.half_dim])

        self.beh0.value = np.zeros([bsize, self.half_dim])
        self.bec0.value = np.zeros([bsize, self.half_dim])

        for line_idx in range(self.num_line):
            # Build Fwd Encode Graph

            fh = self.feh0
            fc = self.fec0
            for idx in range(enc_length):
                in_i = self.input()
                in_i.value = enc_data[line_idx, idx].reshape(1)  # Get value from batch
                x = Embed(in_i, self.feembed)
                fh, fc = self.fenc_lstm_cell(x, fh, fc)

            # Build Bwd Encode Graph
            bh = self.beh0
            bc = self.bec0
            for idx in range(enc_length):
                in_i = self.input()
                in_i.value = enc_data[line_idx, enc_length - 1 - idx].reshape(1)  # Get value from batch
                x = Embed(in_i, self.beembed)
                bh, bc = self.benc_lstm_cell(x, bh, bc)

            h = Concat(fh, bh)
            c = Concat(fc, bc)
            hcollect.append(h)
            ccollect.append(c)

        # Build Decode Graph
        h = Average(Collect(hcollect))
        c = Average(Collect(ccollect))

        self.encoded_h = h
        self.encoded_c = c

        for idx in range(dec_length - 1):
            in_i = self.input()
            in_i.value = dec_data[idx].reshape(1)
            x = Embed(in_i, self.dembed)
            h, c = self.dec_lstm_cell(x, h, c)
            out_i = SoftMax(Dot(h, self.dv2c))
            outputs.append(out_i)

        self.output(Collect(outputs))
        self.expect(dec_data[1:])

    def encode_result(self):
        # return np.concatenate((self.encoded_h.value, self.encoded_c.value), axis=1)
        return self.encoded_c.value
