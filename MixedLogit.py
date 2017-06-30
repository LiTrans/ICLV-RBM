import numpy as np
import theano
import theano.tensor as T
from theano import shared

floatX = theano.config.floatX

class MixedLogit(object):
	""" Mixed Logit by simulation
		This module provides a simple softmax function on both generic
		and non-generic inputs

		conditional logit: U = B_i * x_im + c_i
		multinomial logit: U = B_im * x_m + c_i
	"""
	def __init__(self, n_out, av, input=[None, None], n_in=[None, None], draws=None):
		"""
		Parameters
		----------
		:RandomStream draws: RandomStream of normal draws
		:int n_out: size of output vectors (number of alternatives)
		:tensor av: symbolic tensor referencing to the availability of the
					alternatives
		:list[tensor] input: non-generic and generic input variables
		:list[int] n_in: size of parameters corresponding to non-generic and
						 generic inputs.
		"""
		self.draws = draws
		self.params = []
		self.c = shared(np.zeros((n_out,), dtype=floatX), name='c', borrow=True)
		self.params.extend([self.c])

		if n_in[0] is not None:
			self.B = shared(np.zeros(np.prod(n_in[0]), dtype=floatX), name='B', borrow=True)
			self.params.extend([self.B])
			self.B_s = shared(np.zeros(np.prod(n_in[0]), dtype=floatX), name='B_s', borrow=True)
			self.params.extend([self.B_s])
		if n_in[1] is not None:
			self.D = shared(np.zeros(np.prod(n_in[1]), dtype=floatX), name='D', borrow=True)
			self.D_mat = self.D.reshape(n_in[1])
			self.params.extend([self.D])

		# utility equation
		self.B_RND = self.B + (self.B_s * draws)
		V = T.batched_tensordot(input[0], self.B_RND, axes=[[2],[2]]).dimshuffle(2,0,1) + T.dot(input[1], self.D_mat) + self.c

		# estimate a logit model given availability conditions
		self.p_y_given_x = T.mean(T.clip(self.softmax(V, av), 1e-8, 1.0), axis=0)

		# prediction given choices
		self.y_pred = T.argmax(self.p_y_given_x, axis=-1)

	def softmax(self, x, av):
		# logitistic probability
		e_x = av*T.exp(x - x.max(axis=-1, keepdims=True))
		return e_x / e_x.sum(axis=-1, keepdims=True)

	def loglikelihood(self, y):
        # loglikelihood sum
		return T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
        # returns the number of errors as a percentage of total number of examples
		return T.mean(T.neq(self.y_pred, y))
