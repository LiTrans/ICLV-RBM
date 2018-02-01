import numpy as np
import theano
import theano.tensor as T
from theano import shared, scan
from theano.tensor.shared_randomstreams import RandomStreams

floatX = theano.config.floatX
np.random.seed(42)

def softmax(x, av):
	# logitistic probability
	e_x = av * T.exp(x - x.max(axis=-1, keepdims=True))
	return e_x / e_x.sum(axis=-1, keepdims=True)


class CRBM(object):
	""" Conditional Restricted Boltzmann Machine (CRBM) """
	def __init__(self, n_out, av, n_in=[None, None], n_hid=[None, None],
			     n_ind=[None], input=[None, None], output=None, inds=None):
		"""
		Parameters
		----------
		:int n_out: size of output vectors (number of alternatives)
		:tensor av: symbolic tensor referencing to the availability of the
					alternatives
		:list[tensor] input: non-generic and generic input variables
		:list[int] n_in: size of parameters corresponding to non-generic and
						 generic inputs
		:list[int] n_hid: size of parameters corresponding to latent
		                  variables

		Computational graph
		-------------------
		    x
		    |\G  ~~INDICATORS~~
		    | \ /
		   Bx  x_h - c_h
		    | /D
		    |/
		    Y - c_i

		"""
		self.srng = RandomStreams(1234)

		self.params = []
		self.params2 = []
		self.masks = []
		self.masks2 = []

		self.c_mask = np.ones((n_out,), dtype=np.bool)
		self.c_mask[-1] = 0

		self.masks.extend([shared(self.c_mask)])
		self.masks2.extend([shared(self.c_mask)])

		self.c_i = shared(
			np.zeros((n_out,), dtype=floatX), name='c_i', borrow=True)

		self.params.extend([self.c_i])  # Alternative specific constants
		self.params2.extend([self.c_i])  # Alternative specific constants

		self.B_mask = np.ones(np.prod(n_in[0]), dtype=np.bool)

		self.masks.extend([shared(self.B_mask)])
		self.masks2.extend([shared(self.B_mask)])

		self.B = shared(
			np.zeros(np.prod(n_in[0]), dtype=floatX), name='B', borrow=True)

		self.params.extend([self.B])  # Generic parameters (cost, time, etc.)
		self.params2.extend([self.B])  # Generic parameters (cost, time, etc.)

		self.G_mask = np.ones(np.prod(n_in[1]), dtype=np.bool)
		#self.G_mask[:,-1] = 0
		# self.G_mask[[5, 8, 12, 14, 15, 18, 19, 20, 27, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 44, 47, 48, 50, 51, 53, 56, 59]] = 0

		self.masks.extend([shared(self.G_mask.flatten())])
		self.masks2.extend([shared(self.G_mask.flatten())])

		self.G = shared(
			np.random.uniform(-1., 1., np.prod(n_in[1]))*self.G_mask.flatten(),
			name='G', borrow=True)
		self.G_mat = self.G.reshape(n_in[1])

		self.params.extend([self.G])  # Latent variable parameters
		self.params2.extend([self.G])  # Latent variable parameters

		# self.c_h_mask = np.ones(n_hid[0], dtype=np.bool)
		# self.c_h_mask[-1] = 0
		#
		# self.masks.extend([shared(self.c_h_mask)])
		#
		# self.c_h = shared(
		# 	np.zeros(n_hid[0], dtype=floatX), name='c_h', borrow=True)
		#
		# self.params.extend([self.c_h])  # Latent variable constants

		self.D_mask = np.ones(n_hid[1], dtype=np.bool)
		self.D_mask[:, -1] = 0

		self.masks.extend([shared(self.D_mask.flatten())])
		self.masks2.extend([shared(self.D_mask.flatten())])

		self.D = shared(
			np.random.uniform(-1., 1.,
							  np.prod(n_hid[1]))*self.D_mask.flatten(),
							  name='D', borrow=True)
		self.D_mat = self.D.reshape(n_hid[1]) #(LV, out)

		self.params.extend([self.D])  # Alternative specific params (Latent)
		self.params2.extend([self.D])  # Alternative specific params (Latent)

		self.A = shared(
			np.zeros(np.prod(n_hid[2]), dtype=floatX), name='A', borrow=True)
		self.A_mat = self.A.reshape(n_hid[2])  # Indicator variables

		self.params2.extend([self.A])

		self.A_mask = np.ones(np.prod(n_hid[2]), dtype=np.bool)
		self.A_mask[[0,1,2,3, 4,5,6,7,
					 16,17,18,19, 20,21,22,23,
					 24,25,26,27, 32,33,34,35]] = 0

		self.masks2.extend([shared(self.A_mask.flatten())])

		x_h = T.nnet.sigmoid(T.dot(input[1], self.G_mat))

		# Indicator equation
		ind = T.dot(x_h, self.A_mat)

		v = (T.dot(input[0], self.B)
			 + T.dot(x_h, self.D_mat)
			 + self.c_i)

		# estimate the indicator measurement model
		self.p_z_given_x_h = T.clip(T.nnet.sigmoid(ind), 1e-8, 1.0 - 1e-8)

		self.p_y_given_x = T.clip(softmax(v, av), 1e-8, 1.0)

		self.y_pred = T.argmax(self.p_y_given_x, axis=-1)

		# keep track of input
		self.x_ng = input[0]
		self.x_g = input[1]
		self.av = av

	def cross_entropy(self, z):
		# cross entropy loss cross_entropy(z,p):
		# sum(z*log(p) + (1-z)*log(1-p), axis=-1)
		return T.sum(
			z * T.log(self.p_z_given_x_h)
			+ (1 - z) * T.log(1 - self.p_z_given_x_h))

	def free_energy(self, y, x_ng, x_g):
		""" Function to compute the free energy of a sample conditional
    		on the context
		"""
		dx_b = (T.dot(y, self.D_mat.T) + T.dot(x_g, self.G_mat))
		vbias = T.dot(y, self.c_i)
		visible_term = T.batched_dot(y, T.dot(x_ng, self.B))
		hidden_term = T.sum(T.log(1 + T.exp(dx_b)), axis=-1)

		return - hidden_term - vbias - visible_term

	def propup(self, y, x_g):
		""" This function propagates the visible units activation upwards
			to the hidden units
			Note that we return also the pre-sigmoid activation
		"""
		pre_sigmoid = (T.dot(y, self.D_mat.T)
					   + T.dot(x_g, self.G_mat))

		return [pre_sigmoid, T.nnet.sigmoid(pre_sigmoid)]

	def sample_h_given_v(self, v0_y, v0_x_g):
		""" This function infers state of hidden units given
			visible units
		"""
		pre_sigmoid_h1, sigmoid_h1 = self.propup(v0_y, v0_x_g)
		h1_sample = self.srng.binomial(size=sigmoid_h1.shape, n=1, p=sigmoid_h1,
                                       dtype=floatX)

		return [pre_sigmoid_h1, sigmoid_h1, h1_sample]

	def propdown(self, h1, x_ng, av):
		""" This function propagates the hidden units activation
			downwards to the visible units
			Note that we return also the pre_softmax_activation
		"""
		pre_softmax = (T.dot(x_ng, self.B)
					   + T.dot(h1, self.D_mat)
					   + self.c_i)

		return [pre_softmax, softmax(pre_softmax, av)]

	def sample_v_given_h(self, h0_y, v0_x_ng, av):
		""" This function infers state of visible units given
			hidden units
		"""
		pre_softmax_v1, softmax_v1 = self.propdown(h0_y, v0_x_ng, av)
		v1_sample = softmax_v1
		v1_sample = self.srng.binomial(size=softmax_v1.shape, n=1, p=softmax_v1,
                                       dtype=floatX)

		return [pre_softmax_v1, softmax_v1, v1_sample]

	def gibbs_hvh(self, h0_y, v0_x_ng, v0_x_g, av):
		""" This function implements one step of Gibbs sampling,
			starting from the hidden state
		"""
		pre_softmax_v1, softmax_v1, v1_sample = self.sample_v_given_h(
			h0_y, v0_x_ng, av)
		pre_sigmoid_h1, sigmoid_h1, h1_sample = self.sample_h_given_v(
			v1_sample, v0_x_g)
		return [pre_softmax_v1, softmax_v1, v1_sample,
				pre_sigmoid_h1, sigmoid_h1, h1_sample]

	def gibbs_vhv(self, v0_y, v0_x_ng, v0_x_g, av):
		""" This function implements one step of Gibbs sampling,
			starting from the visible state
		"""
		pre_sigmoid_h1, sigmoid_h1, h1_sample = self.sample_h_given_v(
			v0_y, v0_x_g)
		pre_sigmoid_v1, sigmoid_v1, v1_sample = self.sample_v_given_h(
			h1_sample, v0_x_ng, av)
		return [pre_sigmoid_h1, sigmoid_h1, h1_sample,
				pre_softmax_v1, softmax_v1, v1_sample]

	def gibbs_sampling(self, y, x_ng, x_g, av, alts=6, steps=5):
		""" This function starts the Gibbs sampling chain
			and returns the sample at the end of the chain

		Parameters
		----------

		"""
		y_one_hot = T.extra_ops.to_one_hot(y, alts)

		# compute the positive phase of the network
		pre_sigmoid_pp, sigmoid_pp, pp_sample = self.sample_h_given_v(
			y_one_hot, x_g)

		chain_start = pp_sample

		([pre_softmax_v, softmax_v, v_sample,
		  pre_sigmoid_h, sigmoid_h, h_sample],
		 updates) = scan(
		 	fn=self.gibbs_hvh,
			sequences=None,
			outputs_info=[None, None, None, None, None, chain_start],
			non_sequences=[x_ng, x_g, av],
			n_steps=steps,
			name='gibbs_hvh')

		chain_end = v_sample[-1]

		cost = (T.sum(self.free_energy(y_one_hot, x_ng, x_g))
				- T.sum(self.free_energy(chain_end, x_ng, x_g)))

		error = self.reconstruction_error(pre_softmax_v[-1], y_one_hot, av)

		return cost, error, chain_end, updates

	def reconstruction_error(self, pre_softmax_v, y, av):
		""" Approximation to the reconstruction error

		"""
		p_y_given_xh = T.clip(softmax(pre_softmax_v, av), 1e-8, 1.0)

		return T.sum(
			y * T.log(p_y_given_xh)
			+ (1 - y) * T.log(1 - p_y_given_xh))

	def loglikelihood(self, y):
		# loglikelihood sum
		return T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		# returns the number of errors as a percentage
		return T.mean(T.neq(self.y_pred, y))
