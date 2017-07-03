import numpy as np
import theano
import theano.tensor as T
from theano import shared

floatX = theano.config.floatX
np.random.seed(42)


class Logistic(object):
	""" Simple discrete choice model
		This module provides a simple softmax function on both generic
		and non-generic inputs

		conditional logit: U = B_i * x_im + c_i
		multinomial logit: U = B_im * x_m + c_i

	"""

	def __init__(self, n_out, av,
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


class MixedLogit(object):
	""" Mixed Logit by simulation
		This module provides a simple softmax function on both generic
		and non-generic inputs

		conditional logit: U = B_i * x_im + c_i
		multinomial logit: U = B_im * x_m + c_i

	"""

	def __init__(self, n_out, av,
			     n_in=[None, None], input=[None, None],
				 draws=None):
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

			self.B_s = shared(
				np.zeros(np.prod(n_in[0]), dtype=floatX),
				name='B_s', borrow=True)

			self.params.extend([self.B_s])

			self.B_s_mask = np.ones(np.prod(n_in[0]), dtype=np.bool)

			self.masks.extend([shared(self.B_s_mask)])

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
		self.B_RND = self.B + (self.B_s * draws)

		v = (
			T.batched_tensordot(
				input[0], self.B_RND, axes=[[2], [2]]).dimshuffle(2, 0, 1)
			+ T.dot(input[1], self.D_mat)
			+ self.c)

		# estimate a logit model given availability conditions
		self.p_y_given_x = T.mean(
			T.clip(self.softmax(v, av), 1e-8, 1.0), axis=0)

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

	def __init__(self, n_out, av,
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
		self.masks = []

		self.c_i = shared(
			np.zeros((n_out,), dtype=floatX), name='c_i', borrow=True)

		self.params.extend([self.c_i])  # Alternative specific constants

		self.c_mask = np.ones((n_out,), dtype=np.bool)
		self.c_mask[-1] = 0

		self.masks.extend([shared(self.c_mask)])

		self.B = shared(
			np.zeros(np.prod(n_in[0]), dtype=floatX), name='B', borrow=True)

		self.params.extend([self.B])  # Generic parameters (cost, time, etc.)

		self.B_mask = np.ones(np.prod(n_in[0]), dtype=np.bool)

		self.masks.extend([shared(self.B_mask)])

		self.G_mask = np.ones(n_in[1], dtype=np.bool)
		self.G_mask[:, -1] = 0

		self.masks.extend([shared(self.G_mask.flatten())])

		self.G = shared(
			np.random.uniform(-1., 1., np.prod(n_in[1]))*self.G_mask.flatten(),
			name='G', borrow=True)
		self.G_mat = self.G.reshape(n_in[1])

		self.params.extend([self.G])  # Latent variable parameters

		self.c_h = shared(
			np.zeros(n_hid[0], dtype=floatX), name='c_h', borrow=True)

		self.params.extend([self.c_h])  # Latent variable constants

		self.c_h_mask = np.ones(n_hid[0], dtype=np.bool)
		self.c_h_mask[-1] = 0

		self.masks.extend([shared(self.c_h_mask)])

		self.A = shared(
			np.zeros(np.prod(n_hid[1]), dtype=floatX), name='A', borrow=True)
		self.A_mat = self.A.reshape(n_hid[1])  # Indicator variables

		self.params.extend([self.A])

		self.A_mask = np.ones(n_hid[1], dtype=np.bool)
		self.A_mask[:, -1] = 0

		self.masks.extend([shared(self.A_mask.flatten())])

		self.c_z = shared(
			np.zeros(n_ind[0], dtype=floatX), name='c_z', borrow=True)

		self.params.extend([self.c_z])  # Indicator constants

		self.c_z_mask = np.ones(n_ind[0], dtype=np.bool)
		self.c_z_mask[-1] = 0

		self.masks.extend([shared(self.c_z_mask)])

		self.D = shared(
			np.zeros(np.prod(n_hid[2]), dtype=floatX), name='D', borrow=True)
		self.D_mat = self.D.reshape(n_hid[2])

		self.params.extend([self.D])  # Alternative specific params (Latent)

		self.D_mask = np.ones(n_hid[2], dtype=np.bool)
		self.D_mask[:, -1] = 0

		self.masks.extend([shared(self.D_mask.flatten())])

		# latent variable equation
		x_h = T.nnet.sigmoid(T.dot(input[1], self.G_mat) + self.c_h)

		# Indicator equation
		ind = T.dot(x_h, self.A_mat) + self.c_z

		# utility equation
		v = (
			T.dot(input[0], self.B)
			+ T.dot(x_h, self.D_mat)
			+ self.c_i)

		# estimate the indicator measurement model
		self.p_z_given_x_h = T.clip(T.nnet.sigmoid(ind), 1e-8, 1.0 - 1e-8)

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

	def cross_entropy(self, z):
		# cross entropy loss cross_entropy(z,p):
		# sum(z*log(p) + (1-z)*log(1-p), axis=-1)
		return T.sum(
			z * T.log(self.p_z_given_x_h)
			+ (1 - z) * T.log(1 - self.p_z_given_x_h))

	def errors(self, y):
		# returns the number of errors as a percentage of total number of examples
		return T.mean(T.neq(self.y_pred, y))

