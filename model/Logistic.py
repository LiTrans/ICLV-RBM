import numpy as np
import theano
import theano.tensor as T
from theano import shared

floatX = theano.config.floatX


class Logistic(object):
	""" Simple discrete choice model
		This module provides a simple softmax function on both generic
		and non-generic inputs

		conditional logit: U = B_i * x_im + c_i
		multinomial logit: U = B_im * x_m + c_i

	"""

	def __init__(
		self, n_out, av,
		n_in=[None, None], input=[None, None]):
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

		self.c = shared(
			np.zeros((n_out,), dtype=floatX),
			name='c', borrow=True)

		self.params.extend([self.c])

		self.c_mask = np.ones((n_out,), dtype=np.bool)
		self.c_mask[-1] = 0

		self.masks.extend([shared(self.c_mask)])

		if n_in[0] is not None:
			self.B = shared(
				np.zeros(np.prod(n_in[0]), dtype=floatX),
				name='B', borrow=True)

			self.params.extend([self.B])

			self.B_mask = np.ones(np.prod(n_in[0]), dtype=np.bool)

			self.masks.extend([shared(self.B_mask)])

		if n_in[1] is not None:
			self.D = shared(
				np.zeros(np.prod(n_in[1]), dtype=floatX),
				name='D', borrow=True)
			self.D_mat = self.D.reshape(n_in[1])

			self.params.extend([self.D])

			self.D_mask = np.ones(n_in[1], dtype=np.bool)
			self.D_mask[:, -1] = 0

			self.masks.extend([shared(self.D_mask.flatten())])

		# utility equation
		v = (
			T.dot(input[0], self.B)
			+ T.dot(input[1], self.D_mat)
			+ self.c)

		# estimate a logit model given availability conditions
		self.p_y_given_x = T.clip(self.softmax(v, av), 1e-8, 1.0)

		# prediction given choices
		self.y_pred = T.argmax(self.p_y_given_x, axis=-1)

	def softmax(self, x, av):
		# logitistic probability
		e_x = av * T.exp(x - x.max(axis=-1, keepdims=True))
		return e_x / e_x.sum(axis=-1, keepdims=True)

	def loglikelihood(self, y):
		# loglikelihood sum
		return T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		# returns the number of errors as a percentage of total number of examples
		return T.mean(T.neq(self.y_pred, y))


if __name__ == '__main__':
	main()
