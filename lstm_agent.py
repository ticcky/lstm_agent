import cPickle
import numpy as np
import random
import theano
import theano.tensor as tt

from neural.layers import LstmRecurrent, Embedding, MLP, IdentityInput
from neural.updates import clip_norms, max_norm
import rl


class Episode(object):
    def __init__(self, observs, actions, returns):
        self.observs = observs
        self.actions = actions
        self.returns = returns

    def __eq__(self, other):
        return (self.observs == other.observs and
                self.actions == other.actions and
                self.returns == other.returns)


class LSTMAgent(rl.Agent):
    def __init__(self, alpha, gamma, n_actions, n_states, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.n_actions = n_actions
        self.n_states = n_states

        #self._build_model_ttt(100, 50, 100, 2, self.n_actions, 'rectify')
        self._build_model_o(10, self.n_states, 400, 400, 1, self.n_actions, 'tanh')
        self.debug = False

        self.episodes = []
        self.longest_episode = 0



    def _build_model_o(self, emb_size, n_observations, lstm_n_cells, oclf_n_hidden, oclf_n_layers, n_actions, oclf_activation):
        o = tt.imatrix(name='o')  # Dimensions: (time, seq_id)
        a = tt.imatrix(name='a')
        r = tt.matrix(name='r')
        lr = tt.scalar(name='lr')

        b = self._build_baseline(emb_size, n_observations, 10, o, r, lr)

        l_input = Embedding(name="emb",
                            size=emb_size,
                            n_features=n_observations,
                            input=o)
        prev_layer = l_input


        # l_lstm = LstmRecurrent(name="lstm",
        #                        size=lstm_n_cells,
        #                        seq_output=True,
        #                        out_cells=False,
        #                        peepholes=False,
        #                        output_initial_state=False,
        #                        p_drop=0.0)
        # l_lstm.connect(prev_layer)
        # prev_layer = l_lstm

        l_action = MLP([oclf_n_hidden  ] * oclf_n_layers + [n_actions],
                       [oclf_activation] * oclf_n_layers + ['softmax'],
                       [0.0            ] * oclf_n_layers + [0.0      ],
                       name="mlp")
        l_action.connect(prev_layer)
        prev_layer = l_action

        pi = prev_layer.output()

        # Flatten the actions so that they are stacked in a matrix. Timestep, by timestep.
        orig_shape = pi.shape
        pi = tt.reshape(pi, (pi.shape[0] * pi.shape[1], pi.shape[2]))

        col_actions = tt.reshape(a, (pi.shape[0], ))
        col_rewards = tt.reshape(r, (pi.shape[0], ))
        col_b = tt.reshape(b, (pi.shape[0], ))

        params = [x for x in l_action.get_params()] # if not x.name.startswith('mlp_0')]
        print params

        lin_actions_p = pi[tt.arange(pi.shape[0]), (col_actions)] #+ 1e-7
        objective = tt.sum(tt.log(lin_actions_p) * (col_rewards - col_b)) # * 1.0 / orig_shape[1]

        pi = tt.reshape(pi, orig_shape)

        d_objective = theano.grad(objective, params)
        d_objective = clip_norms(d_objective, 5.0)

        upd = []
        for p, dp in zip(params, d_objective):
            upd.append((p, p + lr * dp))

        self.learn = theano.function([o, a, r, lr], [pi, objective, b] + d_objective, updates=upd)
        self.pi = theano.function([o], pi)

        self.params = params = [x for x in l_action.get_params()]

        self.orig_values = []
        for param in params:
            val = np.copy(param.get_value())
            self.orig_values.append(val)

    def _build_baseline(self, emb_size, n_observations, lstm_n_cells, o, r, lr):
        b_input = Embedding(name="bemb",
                            size=emb_size,
                            n_features=n_observations,
                            input=o)

        b_lstm = LstmRecurrent(name="blstm",
                               size=lstm_n_cells,
                               seq_output=True,
                               out_cells=False,
                               peepholes=False,
                               output_initial_state=False,
                               p_drop=0.0)
        b_lstm.connect(b_input)

        b_out = MLP([1],
                       ['linear'],
                       [0.0      ],
                       name="mlpb")
        b_out.connect(b_lstm)

        b = b_out.output()

        b = b.reshape((b.shape[0], b.shape[1], ))

        params = b_out.get_params()
        loss = ((b - r)**2).sum()

        d_loss = theano.grad(loss, params)

        upd = []
        for p, dp in zip(params, d_loss):
            upd.append((p, p - lr * dp))

        self.blearn = theano.function([o, r, lr], loss, updates=upd)

        return b





    def save(self, f_out):
        model_params = {}
        for param in self.params:
            model_params[param.name] = param.get_value().copy()

        obj = {}
        obj['model_params'] = model_params


        cPickle.dump(obj, f_out, -1)

    def load(self, f_in):
        obj = cPickle.load(f_in)

        model_params = obj['model_params']

        for param in sorted(self.params, key=lambda x: x.name):
            param_val = model_params.get(param.name)
            if param_val != None:
                print 'Loading param: %s' % param.name
                assert param_val.shape == param.get_value().shape
                param.set_value(param_val)
            else:
                print 'Skipping param: %s' % param.name

    def restore_values(self):
        for param, orig_value in zip(self.params, self.orig_values):
            param.set_value(orig_value)



    def get_name(self):
        return "lstm"

    def run(self):
        self._train()


        observs = []
        actions = []
        returns = []
        self.episodes.append(Episode(observs, actions, returns))
        a = None
        while True:
            env_feedback = yield a
            r, s = env_feedback.reward, env_feedback.state_id

            self.debug_print('getting feedback', r, s)
            observs.append(s)
            returns.append(r)

            for i, rr in enumerate(returns[:-1]):
                returns[i] = returns[i] + r * self.gamma ** (len(returns) - i - 1)

            a = self.sample_action(np.array([observs], dtype='int32').swapaxes(0, 1))
            actions.append(a)

            self.longest_episode = max(len(actions), self.longest_episode)

            self.debug_print('executing', a)

            self.debug_print('r', returns)
            self.debug_print('o', observs)
            self.debug_print('a', actions)

    def _train(self):
        if len(self.episodes) == 0: return

        lo = []
        la = []
        lr = []

        for e in self.episodes[-1:]:
            n_padding = self.longest_episode - len(e.actions)
            lo.append(e.observs[:-1] + [0] * n_padding)
            la.append(e.actions[:-1] + [0] * n_padding)
            lr.append(e.returns[1:] + [0.0] * n_padding)

        lo = np.array(lo, dtype='int32').swapaxes(0, 1)
        la = np.array(la, dtype='int32').swapaxes(0, 1)
        lr = np.array(lr, dtype='float32').swapaxes(0, 1)

        res = self.learn(lo, la, lr, self.alpha)
        pi, obj, b = res[0], res[1], res[2]

        #print b
        #print lr

        self.blearn(lo, lr, self.alpha)

        #import ipdb; ipdb.set_trace()

    def debug_print(self, *args):
        if self.debug:
            for a in args:
                print a,
            print

    def sample_action(self, s):
        #p_vec = self.pi(s)
        #p_vec = p_vec[-1][0]
        #a = np.random.choice(self.n_actions, p=p_vec)
        #return a

        if np.random.random() < self.epsilon:
            #print 'random action'
            return np.random.choice(range(self.n_actions))
        else:
            #print 'policy action', s
            p_vec = self.pi(s)
            p_vec = p_vec[-1][0]
            #a = np.random.choice(self.n_actions, p=p_vec)
            a = np.argmax(p_vec)

            #for i, p in enumerate(p_vec):
            #    print '    %d(%.2f)' % (i, p, )

            return a
