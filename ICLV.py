import numpy as np
import theano
import theano.tensor as T
from theano import shared

floatX = theano.config.floatX


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

	def __init__(
		self, n_out, av,
		n_in=[None, None], n_hid=[None, None, None], n_ind=[None],
		input=[None, None]):
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
		self.c_i = shared(
			np.zeros((n_out,), dtype=floatX), name='c_i', borrow=True)

		self.params.extend([self.c_i])  # Alternative specific constants

		self.B = shared(
			np.zeros(np.prod(n_in[0]), dtype=floatX), name='B', borrow=True)

		self.params.extend([self.B])  # Generic parameters (cost, time, etc.)

		self.G = shared(
			np.zeros(np.prod(n_in[1]), dtype=floatX), name='G', borrow=True)
		self.G_mat = self.G.reshape(n_in[1])

		self.params.extend([self.G])  # Latent variable parameters

		self.c_h = shared(
			np.zeros((n_hid[0],), dtype=floatX), name='c_h', borrow=True)

		self.params.extend([self.c_h])  # Latent variable constants

		self.A = shared(
			np.zeros(np.prod(n_hid[1]), dtype=floatX), name='A', borrow=True)

		self.A_mat = self.A.reshape(n_hid[1])  # Indicator variables

		self.c_z = shared(
			np.zeros((n_ind[1][-1],), dtype=floatX), name='c_z', borrow=True)

		self.params.extend([self.c_z])  # Indicator constants

		self.D = shared(
			np.zeros(np.prod(n_hid[2]), dtype=floatX), name='D', borrow=True)
		self.D_mat = self.D.reshape(n_hid[2])

		self.params.extend([self.D])  # Alternative specific params (Latent)

		# latent variable equation
		x_h = T.dot(input[1], self.G_mat) + self.c_h

		# Indicator equation
		z = T.dot(x_h, self.A_mat) + self.c_z

		# utility equation
		v = (
			T.dot(input[0], self.B)
			+ T.dot(x_h, self.D_mat)
			+ self.c_i)

		# estimate the indicator measurement model
		self.p_z_given_x_h = T.clip(T.nnet.sigmoid(z), 1e-8, 1.0 - 1e-8)

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

	def cross_entropy(self, y):
		# cross entropy loss cross_entropy(z,p):
		# sum(z*log(p) + (1-z)*log(1-p), axis=-1)
		return T.sum(
			y * T.log(self.p_z_given_x_h)
			+ (1 - y) * T.log(1 - self.p_z_given_x_h), axis=-1)

	def errors(self, y):
		# returns the number of errors as a percentage of total number of examples
		return T.mean(T.neq(self.y_pred, y))


if __name__ == '__main__':
	main()
