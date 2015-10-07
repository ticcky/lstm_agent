import theano
import theano.tensor as T

from utils import floatX


def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g


def clip_norms(gs, c):
    norm = T.sqrt(sum([T.sum(g**2) for g in gs]))
    return [clip_norm(g, c, norm) for g in gs]


def max_norm(p, maxnorm):
    if maxnorm > 0:
        norms = T.sqrt(T.sum(T.sqr(p)))
        desired = T.clip(norms, 0, maxnorm)
        p = p * (desired/ (1e-7 + norms))
    return p


class Regularizer(object):

    def __init__(self, l1=0., l2=0., maxnorm=0.):
        self.l1 = l1
        self.l2 = l2
        self.maxnorm = maxnorm

    def gradient_regularize(self, p, g):
        if self.l1 > 0 or self.l2 > 0:
            g += p * self.l2
            g += T.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        p = max_norm(p, self.maxnorm)
        return p


class Update(object):

    def __init__(self, regularizer=Regularizer(), clipnorm=0.):
        self.regularizer = regularizer
        self.clipnorm = clipnorm

    def get_updates(self, grads, params, cost):
        raise NotImplementedError

    def get_update_ratio(self, params, updates):
        res = 0.0
        for p, np in updates:
            if p in params:
                res += (np - p).norm(2) / p.norm(2)
        res = res / len(updates)
        return res

    def get_grad_norm(self):
        res = 0.0
        for grad in self.grads:
            res += (grad**2).sum()

        return T.sqrt(res)

    def get_grad_vector(self):
        return T.concatenate([T.flatten(x) for x in self.grads])


class Adam(Update):

    #def __init__(self, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, clip=0.0, *args, **kwargs):
    def __init__(self, lr=0.0002, b1=0.9, b2=0.999, e=1e-8, clip=0.0, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.clip = clip
        self.clipnorm = clip

    def get_updates(self, grads, params, cost):
        updates = []
        grads = clip_norms(grads, self.clipnorm)
        #grads = [T.clip(g, -self.clip, self.clip) for g in grads]
        i = theano.shared(floatX(0.))
        i_t = i + 1.
        fix1 = 1. - self.b1**(i_t)
        fix2 = 1. - self.b2**(i_t)
        lr_t = self.lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (self.b1 * g) + ((1. - self.b1) * m)
            v_t = (self.b2 * T.sqr(g)) + ((1. - self.b2) * v)
            g_t = m_t / (T.sqrt(v_t) + self.e)
            g_t = self.regularizer.gradient_regularize(p, g_t)
            p_t = p - (lr_t * g_t)
            #p_t = self.regularizer.weight_regularize(p_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates