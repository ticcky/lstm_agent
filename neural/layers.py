import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.extra_ops import repeat
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import itertools
from utils import shared0s, flatten
import activations
import inits
import costs

import numpy as np



class Layer(object):
    name = "unnamed_layer"
    srng = None
    #def connect(self):
    #    pass

    def output(self, dropout_active=False):
        raise NotImplementedError()

    def _name_param(self, param_name):
        return "%s__%s" % (self.name, param_name, )

    def dropout(self, X, p=0.):
        if not self.srng:
            self.srng = RandomStreams(seed=1234)

        if p != 0:
            retain_prob = 1 - p
            X = X * self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        return X


class IdentityInput(object):
    def __init__(self, val, size):
        self.val = val
        self.size = size

    def set_val(self, val):
        self.val = val

    def output(self, dropout_active=False):
        return self.val

    def get_params(self):
        return set()


class Embedding(Layer):
    def __init__(self, name=None, size=128, n_features=256, init=inits.normal,
                 static=False, input=None):
        if name:
            self.name = name
        self.init = init
        self.size = size
        self.n_features = n_features
        self.input = input
        self.wv = self.init((self.n_features, self.size),
                            fan_in=self.n_features,
                            name=self._name_param("emb"))
        if static:
            self.params = set()
        else:
            self.params = {self.wv}
        self.static = static

    def output(self, dropout_active=False):
        return self.wv[self.input]

    def get_params(self):
        return self.params


class ZipLayer(object):
    def __init__(self, concat_axis, layers):
        self.layers = layers
        self.concat_axis = concat_axis
        self.size = sum(layer.size for layer in layers) # - layers[1].size

    def output(self, dropout_active=False):
        outs = [layer.output(dropout_active=dropout_active)
                for layer in self.layers]

        res = T.concatenate(outs, axis=self.concat_axis)
        return T.cast(res, dtype=theano.config.floatX)
        #return T.concatenate([outs[0] * T.repeat(outs[1], self.layers[0].size,
        #                                         axis=2),
        #                      outs[2]],
        #                     axis=self.concat_axis)

    def get_params(self):
        return set(flatten([layer.get_params() for layer in self.layers]))


class LstmRecurrent(Layer):

    def __init__(self, name=None, size=256, init=inits.normal, truncate_gradient=-1,
                 seq_output=False, p_drop=0., init_scale=0.1, out_cells=False,
                 peepholes=False, enable_branch_exp=False, backward=False,
                 learn_init_state=True, output_initial_state=False):
        if name:
            self.name = name
        self.init = init
        self.init_scale = init_scale
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.out_cells = out_cells
        self.p_drop = p_drop
        self.peepholes = peepholes
        self.backward = backward
        self.learn_init_state = learn_init_state
        self.output_initial_state = output_initial_state

        self.gate_act = activations.sigmoid
        self.modul_act = activations.tanh

        self.enable_branch_exp = enable_branch_exp
        self.lagged = []

    def _init_input_connections(self, n_in):
        self.w = self.init((n_in, self.size * 4),
                           fan_in=n_in,
                           name=self._name_param("W"))
        self.b = inits.const((self.size * 4, ),
                             0.1,
                             name=self._name_param("b"))

        #self.br = self.init((self.size * 4, ),
        #                   layer_width=self.size,
        #                   scale=self.init_scale,
        #                   name=self._name_param("br"))

        # Initialize forget gates to large values.
        b = self.b.get_value()
        b[:self.size] = 1.0
        #b[self.size:] = 0.0
        self.b.set_value(b)

    def _init_recurrent_connections(self):
        self.u = self.init((self.size, self.size * 4),
                           fan_in=self.size,
                           name=self._name_param("U"))

    def _init_peephole_connections(self):
        self.p_vec_f = self.init((self.size, ),
                                 fan_in=self.size,
                                 name=self._name_param("peep_f"))
        self.p_vec_i = self.init((self.size, ),
                                 fan_in=self.size,
                                 name=self._name_param("peep_i"))
        self.p_vec_o = self.init((self.size, ),
                                 fan_in=self.size,
                                 name=self._name_param("peep_o"))

    def _init_initial_states(self, init_c=None, init_h=None):
        if self.learn_init_state:
            self.init_c = self.init((self.size, ),
                                    fan_in=self.size,
                                    name=self._name_param("init_c"))
            self.init_h = self.init((self.size, ),
                                    fan_in=self.size,
                                    name=self._name_param("init_h"))
        else:
            self.init_c = init_c
            self.init_h = init_h

    def connect(self, l_in, init_c=None, init_h=None):
        self.l_in = l_in

        self._init_input_connections(l_in.size)
        self._init_recurrent_connections()
        self._init_peephole_connections()  # TODO: Make also conditional.

        self.params = [self.w, self.u, self.b]

        self._init_initial_states(init_c, init_h)
        if self.learn_init_state:
            self.params += [self.init_c, self.init_h]

        if self.peepholes:
            self.params += [self.p_vec_f, self.p_vec_i, self.p_vec_o]

    def connect_lagged(self, l_in):
        self.lagged.append(l_in)

    def _slice(self, x, n):
            return x[:, n * self.size:(n + 1) * self.size]

    def step(self, x_t, h_tm1, c_tm1, u, p_vec_f, p_vec_i, p_vec_o,
             dropout_active):
        h_tm1_dot_u = T.dot(h_tm1, u)
        gates_fiom = x_t + h_tm1_dot_u

        g_f = self._slice(gates_fiom, 0)
        g_i = self._slice(gates_fiom, 1)
        g_m = self._slice(gates_fiom, 3)

        if self.peepholes:
            g_f += c_tm1 * p_vec_f
            g_i += c_tm1 * p_vec_i

        g_f = self.gate_act(g_f)
        g_i = self.gate_act(g_i)
        g_m = self.modul_act(g_m)

        c_t = g_f * c_tm1 + g_i * g_m

        g_o = self._slice(gates_fiom, 2)

        if self.peepholes:
            g_o += c_t * p_vec_o

        g_o = self.modul_act(g_o)

        h_t = g_o * T.tanh(c_t)

        return h_t, c_t

    def _compute_x_dot_w(self, dropout_active):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = self.dropout(X, self.p_drop)
            dropout_corr = 1.0
        else:
            dropout_corr = 1.0 - self.p_drop
        x_dot_w = T.dot(X, self.w * dropout_corr) + self.b
        return x_dot_w

    def _reverse_if_backward(self, cells, out):
        if self.backward:
            out = out[::-1, ]
            cells = cells[::-1, ]
        return cells, out

    def _prepare_result(self, cells, out, outputs_info):
        res_init = None
        if self.seq_output:
            if self.out_cells:
                res = cells
                res_init = outputs_info[0]
            else:
                res = out
                res_init = outputs_info[1]
        else:
            if self.out_cells:
                res = cells[-1]
            else:
                res = out[-1]

        if res_init and self.output_initial_state:
            res = T.concatenate([res_init.dimshuffle('x', 0, 1), res], axis=0)

        return res

    def _prepare_outputs_info(self, x_dot_w):
        if self.learn_init_state:
            outputs_info = [
                T.repeat(self.init_c.dimshuffle('x', 0), x_dot_w.shape[1], axis=0),
                T.repeat(self.init_h.dimshuffle('x', 0), x_dot_w.shape[1], axis=0),
            ]
        else:
            outputs_info = [
                self.init_c,
                self.init_h
            ]
        return outputs_info

    def _process_scan_output(self, res):
        (out, cells), _ = res

        return out, cells

    def _compute_seq(self, x_dot_w, dropout_active, outputs_info):
        res = theano.scan(self.step,
                                      sequences=[x_dot_w],
                                      outputs_info=outputs_info,
                                      non_sequences=[self.u, self.p_vec_f,
                                                     self.p_vec_i,
                                                     self.p_vec_o,
                                                     1 if dropout_active else
                                                     0],
                                      truncate_gradient=self.truncate_gradient,
                                      go_backwards=self.backward
        )

        out, cells = self._process_scan_output(res)
        return cells, out

    def output(self, dropout_active=False):
        x_dot_w = self._compute_x_dot_w(dropout_active)
        outputs_info = self._prepare_outputs_info(x_dot_w)

        cells, out = self._compute_seq(x_dot_w, dropout_active, outputs_info)
        cells, out = self._reverse_if_backward(cells, out)

        cells = ifelse(x_dot_w.shape[0] > 0, cells, T.unbroadcast(outputs_info[0].dimshuffle('x', 0, 1), 0, 1))
        out = ifelse(x_dot_w.shape[0] > 0, out, T.unbroadcast(outputs_info[1].dimshuffle('x', 0, 1), 0, 1))

        self.outputs = [cells, out]

        return self._prepare_result(cells, out, outputs_info)

    def get_params(self):
        return set(self.l_in.get_params()).union(self.params)



class Dense(Layer):
    def __init__(self, name=None, size=256, activation='rectify', init=inits.normal,
                 p_drop=0.):
        if name:
            self.name = name
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.init = init
        self.size = size
        self.p_drop = p_drop

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size

        self.w = self.init(
            (self.n_in, self.size),
            fan_in=self.n_in,
            name=self._name_param("w")
        )
        self.b = inits.const(
            (self.size, ),
            val=0.1,
            name=self._name_param("b")
        )
        self.params = [self.w, self.b]

    def output(self, pre_act=False, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = self.dropout(X, self.p_drop)
            dropout_corr = 1.
        else:
            dropout_corr = 1.0 - self.p_drop

        is_tensor3_softmax = X.ndim > 2 and self.activation_str == 'softmax'

        shape = X.shape
        if is_tensor3_softmax: #reshape for tensor3 softmax
            X = X.reshape((shape[0]*shape[1], self.n_in))

        out =  self.activation(T.dot(X, self.w * dropout_corr) + self.b)

        if is_tensor3_softmax: #reshape for tensor3 softmax
            out = out.reshape((shape[0], shape[1], self.size))

        return out

    def get_params(self):
        return set(self.params).union(set(self.l_in.get_params()))


class MLP(Layer):
    def __init__(self, sizes, activations, p_drop=itertools.repeat(0.0),
                 name=None, init=inits.normal):
        self.layers = layers = []
        for layer_id, (size, activation, l_p_drop) in enumerate(zip(sizes,
                                                           activations, p_drop)):
            layer = Dense(size=size, activation=activation, name="%s_%d" % (
                name, layer_id, ), p_drop=l_p_drop, init=init)
            layers.append(layer)

        self.stack = Stack(layers, name=name)

    def connect(self, l_in):
        self.l_in = l_in
        self.stack.connect(l_in)

        if len(self.layers) != 0:
            self.size = self.layers[-1].size
        else:
            self.size = self.l_in.size

    def output(self, dropout_active=False):
        return self.stack.output(dropout_active=dropout_active)

    def get_params(self):
        return list(set(self.stack.get_params()))


class Stack(Layer):
    def __init__(self, layers, name=None):
        if name:
            self.name = name
        self.layers = layers

    def connect(self, l_in):
        self.l_in = l_in
        if len(self.layers) > 0:
            self.layers[0].connect(l_in)
            for i in range(1, len(self.layers)):
                self.layers[i].connect(self.layers[i-1])

            self.size = self.layers[-1].size
        else:
            self.size = l_in.size
            self.layers = [l_in]

    def output(self, dropout_active=False):
        return self.layers[-1].output(dropout_active=dropout_active)

    def get_params(self):
        return set(flatten([layer.get_params() for layer in self.layers]))
