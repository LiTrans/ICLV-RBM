import numpy as np
import theano
import theano.tensor as T
from theano import shared

floatX = theano.config.floatX

class CRBM(object):
	"""Conditional Restricted Boltzmann Machine (CRBM)"""
	def __init__(self, n_out, av, input=[None, None], n_in=[None, None], n_hid=[None]):
		name = 'CRBM'

class ICLV(object):
	""" An intergrated choice and latent variable model
		generic observed variables are used as inputs for the
		latent variables.

		The choice model consists of alternative specific
		variables and generic latent variables.

		Indicator outputs are binary valued [0,1], and a non-
		linear transform of latent variables is performed using
		the sigmoid() function

		The loss function is comprised of the loglikelihood given the
		alternative specific variables and latent variables with
		the cross entropy of the indicators given the latent variables

		Latent variable: x_star_h = G_hm * x_m + g_h
		Indicators: I_p = A_ph * x_star_h + j_p
					P(I|x_star) = sigmoid(I_p)
		Choice utility: U = B_i * x_im + D_ih * x_star_h + c_i
						P(y|x,x_star) = softmax(U, av)
		loss fn: sum(log(P(y|x,x_star))) + sum(log(P(I|x_star))) +
					sum(cross_entropy(z, P(I|x_star)))

		cross_entropy(z,p): sum(z*log(p) + (1-z)*log(1-p), axis=-1)
	"""
	def __init__(self, n_out, av, n_in=[(2,), (2,1)], n_hid=[1, (1,12), (1,6)], n_ind=[(1,)], input=[None, None]):
		"""
		Parameters
		----------
		:int n_out: size of output vectors (number of alternatives)
		:tensor av: symbolic tensor referencing to the availability of the
					alternatives
		:list[tensor] input: non-generic and generic input variables
		:list[int] n_in: size of parameters corresponding to non-generic and
						 generic inputs.
		"""
		self.params = []
		self.c = shared(np.zeros((n_out,), dtype=floatX), name='c', borrow=True)
		self.params.extend([self.c])

		self.B = shared(np.zeros(np.prod(n_in[0]), dtype=floatX), name='B', borrow=True)
		self.params.extend([self.B])

		self.G = shared(np.zeros(np.prod(n_in[1]), dtype=floatX), name='G', borrow=True)
		self.G_mat = self.G.reshape(n_in[1])
		self.params.extend([self.G]) # (obs_var, latent_var)

		self.g = shared(np.zeros((n_hid[0],), dtype=floatX), name='g', borrow=True)
		self.params.extend([self.g]) # (latent_var,)

		self.A = shared(np.zeros(np.prod(n_hid[1]), dtype=floatX), name='A', borrow=True)
		self.A_mat = self.A.reshape(n_hid[1]) # (latent_var, indicators)

		self.j = shared(np.zeros((n_ind[1][-1],), dtype=floatX), name='j', borrow=True)
		self.params.extend([self.j]) # (indicators,)

		self.D = shared(np.zeros(np.prod(n_hid[2]), dtype=floatX), name='D', borrow=True)
		self.D_mat = self.D.reshape(n_hid[2])
		self.params.extend([self.D]) # (latent_var, choices)

		# latent variable equation
		x_s = T.dot(input[1], self.G_mat) + self.g

		# Indicator equation
		I = T.dot(x_star, self.A_mat) + self.j

		# utility equation
		V = T.dot(input[0], self.B) + T.dot(x_star, self.D_mat) + self.c

		# estimate the indicator measurement model
		self.p_I_given_x_s = T.clip(T.nnet.sigmoid(I), 1e-8, 1.0-1e-8)

		# estimate a logit model given availability conditions
		self.p_y_given_x = T.clip(self.softmax(V, av), 1e-8, 1.0)

		# prediction given choices
		self.y_pred = T.argmax(self.p_y_given_x, axis=-1)

	def softmax(self, x, av):
		# logitistic probability
		e_x = av*T.exp(x - x.max(axis=-1, keepdims=True))
		return e_x / e_x.sum(axis=-1, keepdims=True)

	def loglikelihood(self, y):
        # loglikelihood sum
		return T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def cross_entropy(self, y):
		# cross entropy loss cross_entropy(z,p): sum(z*log(p) + (1-z)*log(1-p), axis=-1)
		return T.sum(y*T.log(self.p_I_given_x_star) + (1-y)*T.log(1-self.p_I_given_x_star), axis=-1)

	def errors(self, y):
        # returns the number of errors as a percentage of total number of examples
		return T.mean(T.neq(self.y_pred, y))


class Logistic(object):
	""" Simple discrete choice model
		This module provides a simple softmax function on both generic
		and non-generic inputs

		conditional logit: U = B_i * x_im + c_i
		multinomial logit: U = B_im * x_m + c_i
	"""
	def __init__(self, n_out, av, n_in=[None, None], input=[None, None]):
		"""
		Parameters
		----------
		:int n_out: size of output vectors (number of alternatives)
		:tensor av: symbolic tensor referencing to the availability of the
					alternatives
		:list[tensor] input: non-generic and generic input variables
		:list[int] n_in: size of parameters corresponding to non-generic and
						 generic inputs.
		"""
		self.params = []
		self.masks = []

		self.c = shared(np.zeros((n_out,), dtype=floatX), name='c', borrow=True)
		self.params.extend([self.c])

		self.c_mask = np.ones((n_out,), dtype=np.bool)
		self.c_mask[-1] = 0
		self.masks.extend([shared(self.c_mask)])

		if n_in[0] is not None:
			self.B = shared(np.zeros(np.prod(n_in[0]), dtype=floatX), name='B', borrow=True)
			self.params.extend([self.B])

			self.B_mask = np.ones(np.prod(n_in[0]), dtype=np.bool)
			self.masks.extend([shared(self.B_mask)])

		if n_in[1] is not None:
			self.D = shared(np.zeros(np.prod(n_in[1]), dtype=floatX), name='D', borrow=True)
			self.D_mat = self.D.reshape(n_in[1])
			self.params.extend([self.D])

			self.D_mask = np.ones(n_in[1], dtype=np.bool)
			self.D_mask[:, -1] = 0
			self.masks.extend([shared(self.D_mask.flatten())])

		# utility equation
		V = T.dot(input[0], self.B) + T.dot(input[1], self.D_mat) + self.c

		# estimate a logit model given availability conditions
		self.p_y_given_x = T.clip(self.softmax(V, av), 1e-8, 1.0)

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

if __name__ == '__main__':
	main()
#
