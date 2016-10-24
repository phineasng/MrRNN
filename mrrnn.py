import tensorflow as tf
import numpy as np

class Configuration:
	def __init__(self):
		return

class MRRNN:
	""" Multiresolution Recurrent Neural Network
	
	Implementation following the paper 
	"Multiresolution Recurrent Neural Networks:
			An Application to Dialogue Response Generation"
	by I.V.Serban et al.

	"""
	def __init__(self,config):

		self.word_emb_size = config.word_emb_size
		self.coarse_emb_size = config.coarse_emb_size
		self.word_vocab_size = config.word_vocab_size
		self.coarse_vocab_size = config.coarse_vocab_size
		self.word_context_hid_size = config.word_context_hid_size
		self.coarse_context_hid_size = config.coarse_context_hid_size
		self.word_encoder_hid_size = config.word_encoder_hid_size
		self.coarse_encoder_hid_size = config.coarse_encoder_hid_size
		# the coarse context hidden state need to be multiplied to the hidden states 
		# of the decoder -> they have to be of the same size
		self.coarse_decoder_hid_size = config.coarse_context_hid_size
		# the concatenated states of the word context and the coarse prediction 
		#	encoder have to be multiplied with the hidden states of the word
		# decoder
		self.word_decoder_hid_size =\
		 config.word_context_hid_size + config.prediction_encoder_hid_size 
		self.prediction_encoder_hid_size = config.prediction_encoder_hid_size
		self.learning_rate = config.learning_rate

		# inputs - a single utterance for each sub-model
		self.input_word = tf.placeholder(tf.int32,[None],name="word_embedding")
		self.input_coarse = tf.placeholder(tf.int32,[None],name="coarse_embedding")
		# outputs - next utterance
		self.output_word_length = tf.placeholder(\
				tf.int32,[1],\
				name="out_word_seq_length")
		self.output_coarse_length = tf.placeholder(\
				tf.int32,[1],\
				name="out_coarse_seq_length")
		# target labels
		self.target_labels_coarse = tf.placeholder(\
				tf.int32,[None],\
				name="target_labels_coarse")
		self.target_labels_word = tf.placeholder(\
				tf.int32,[None],\
				name="target_labels_word")
		# placeholders for previous context state
		self.previous_coarse_context = tf.placeholder(\
				tf.float32,[1,self.coarse_context_hid_size],\
				name="previous_coarse_context")
		self.previous_word_context = tf.placeholder(\
				tf.float32,[1,self.word_context_hid_size],\
				name="previous_word_context")


		# create embedding variables
		self.word_embedding =\
			tf.Variable(\
				tf.random_uniform([self.word_vocab_size,self.word_emb_size],-1,1))
		self.coarse_embedding =\
			tf.Variable(\
				tf.random_uniform([self.coarse_vocab_size,self.coarse_emb_size],-1,1))
		self.out_word_embedding =\
			tf.Variable(\
				tf.random_uniform(\
					[self.word_vocab_size,self.word_decoder_hid_size],-1,1))
		self.out_coarse_embedding =\
			tf.Variable(\
				tf.random_uniform(\
					[self.coarse_vocab_size,self.coarse_decoder_hid_size],-1,1))


		# create graph and loss
		self._build_graph()
		self._create_loss_and_optimizer()
		self._create_grad_accumul_op()

		init = tf.initialize_all_variables()

		self.sess = tf.Session()
		self.sess.run(init)

		# visualize graph for debugging
		self.writer = tf.train.SummaryWriter("./log",graph=self.sess.graph)

	def _get_embedded_coarse_input(self):

		selected_embeddings = tf.nn.embedding_lookup(\
					self.coarse_embedding,\
					self.input_coarse)

		return tf.pack([selected_embeddings])

	def _get_embedded_word_input(self):

		selected_embeddings = tf.nn.embedding_lookup(\
					self.word_embedding,\
					self.input_word)

		return tf.pack([selected_embeddings])

	def _create_coarse_encoder(self):

		with tf.variable_scope("coarse_encoder"):
			cell = tf.nn.rnn_cell.GRUCell(self.coarse_encoder_hid_size)
			_, hidden_state = tf.nn.dynamic_rnn(\
						cell,\
						self.coarse_representation,\
						dtype=tf.float32)

		return [tf.pack([hidden_state[-1,:]])]

	def _create_word_encoder(self):

		with tf.variable_scope("word_encoder"):
			cell = tf.nn.rnn_cell.GRUCell(self.word_encoder_hid_size)
			_, hidden_state = tf.nn.dynamic_rnn(\
						cell,\
						self.word_representation,\
						dtype=tf.float32)

		return [tf.pack([hidden_state[-1,:]])]

	def _create_coarse_context(self):

		with tf.variable_scope("coarse_context"):
			cell = tf.nn.rnn_cell.GRUCell(self.coarse_context_hid_size)
			
			self.reset_coarse_state = cell.zero_state(1,dtype=tf.float32)
			
			_, hidden = tf.nn.rnn(\
							cell,\
							self.encoded_coarse,\
							initial_state = self.previous_coarse_context)

		return hidden[-1,:]

	def _create_word_context(self):

		with tf.variable_scope("word_context"):
			cell = tf.nn.rnn_cell.GRUCell(self.word_context_hid_size)
			
			self.reset_word_state = cell.zero_state(1,dtype=tf.float32)
			
			_, hidden = tf.nn.rnn(\
							cell,\
							self.encoded_word,\
							initial_state = self.previous_word_context)

		return hidden[-1,:]

	def decoder_coarse_condition(self,it,outputs,hidden):
		return it[0] < self.output_coarse_length[0] - 1
	
	def decoder_word_condition(self,it,outputs,hidden):
		return it[0] < self.output_word_length[0]	- 1

	def decoder_coarse_func(self,it,outputs,hidden):
		# hidden state is multiplied with the context hidden state 
		out,hidden = self.decoder_coarse_cell(\
				tf.pack([outputs[-1,:]]),\
				tf.pack([tf.mul(hidden,self.context_coarse_current_hidden)]))
		
		if it == 0:
			outputs = tf.identity(out)
		else:
			outputs = tf.concat(0,[outputs,out])
		return it+1,outputs,hidden[0,:]

	def decoder_word_func(self,it,outputs,hidden):
		# hidden state is multiplied with the context hidden state 
		out,hidden = self.decoder_word_cell(\
				tf.pack([outputs[-1,:]]),\
				tf.pack([tf.mul(hidden,self.word_context_concatenation)]))
		
		if it == 0:
			outputs = tf.identity(out)
		else:
			outputs = tf.concat(0,[outputs,out])
		return it+1,outputs,hidden[0,:]

	def _create_coarse_decoder(self):
		with tf.variable_scope("coarse_decoder"):
			self.decoder_coarse_cell =\
					 tf.nn.rnn_cell.GRUCell(self.coarse_decoder_hid_size)
			self.iter_decoder_coarse = tf.constant(0)
			result = tf.while_loop(\
				self.decoder_coarse_condition,\
				self.decoder_coarse_func,\
				[	self.iter_decoder_coarse,\
					tf.zeros([1,self.coarse_decoder_hid_size]),\
					tf.ones([self.coarse_decoder_hid_size])],\
				shape_invariants=[\
								self.iter_decoder_coarse.get_shape(),\
								tf.TensorShape([None,self.coarse_decoder_hid_size]),\
								self.context_coarse_current_hidden.get_shape()])
		return result[1]

	def _create_word_decoder(self):
		with tf.variable_scope("word_decoder"):
			self.decoder_word_cell =\
					 tf.nn.rnn_cell.GRUCell(self.word_decoder_hid_size)
			self.iter_decoder_word = tf.constant(0)
			result = tf.while_loop(\
				self.decoder_word_condition,\
				self.decoder_word_func,\
				[	self.iter_decoder_word,\
					tf.zeros([1,self.word_decoder_hid_size]),\
					tf.ones([self.word_decoder_hid_size])],\
				shape_invariants=[\
								self.iter_decoder_word.get_shape(),\
								tf.TensorShape([None,self.word_decoder_hid_size]),\
								self.word_context_concatenation.get_shape()])
		return result[1]

	def _create_prediction_encoder(self):
		with tf.variable_scope("prediction_encoder"):
			cell = tf.nn.rnn_cell.GRUCell(self.prediction_encoder_hid_size)
			_,hidden_state = tf.nn.dynamic_rnn(\
						cell,\
						tf.pack([self.coarse_prediction]),\
						dtype=tf.float32)
		return hidden_state[-1,:]

	def _build_graph(self):

		self.coarse_representation = self._get_embedded_coarse_input()
		#print self.coarse_representation.get_shape()

		self.encoded_coarse = self._create_coarse_encoder()
		#print self.encoded_coarse[0].get_shape()

		self.context_coarse_current_hidden = self._create_coarse_context()
		#print self.context_coarse_current_hidden.get_shape()

		self.coarse_prediction = self._create_coarse_decoder()
		#print self.coarse_prediction.get_shape()

		self.encoded_prediction = self._create_prediction_encoder()
		#print self.encoded_prediction.get_shape()

		self.word_representation = self._get_embedded_word_input()
		#print self.word_representation.get_shape()

		self.encoded_word = self._create_word_encoder()
		#print self.encoded_word[0].get_shape()

		self.context_word_current_hidden = self._create_word_context()
		#print self.context_word_current_hidden.get_shape()

		self.word_context_concatenation = \
				tf.concat(0,\
					[self.context_word_current_hidden,self.encoded_prediction])

		self.word_prediction = self._create_word_decoder()
		#print self.word_prediction.get_shape()

		self.logits_word,self.logits_coarse = self._compute_logits()
		#print self.out_word_embedding.get_shape()
		#print self.out_coarse_embedding.get_shape()
		#print self.logits_word.get_shape()
		#print self.logits_coarse.get_shape()


	def _compute_logits(self):
		word_logits = tf.matmul(\
				self.out_word_embedding,\
				tf.transpose(self.word_prediction))
		coarse_logits = tf.matmul(\
				self.out_coarse_embedding,\
				tf.transpose(self.coarse_prediction))

		return tf.transpose(word_logits),tf.transpose(coarse_logits) 

	def _create_loss_and_optimizer(self):

		word_target_one_hot = tf.one_hot(\
				self.target_labels_word,\
				self.word_vocab_size)
		coarse_target_one_hot = tf.one_hot(\
				self.target_labels_coarse,\
				self.coarse_vocab_size)
		
		map_word = tf.nn.softmax_cross_entropy_with_logits(\
				self.logits_word,\
				word_target_one_hot)
		map_coarse = tf.nn.softmax_cross_entropy_with_logits(\
				self.logits_coarse,\
				coarse_target_one_hot)

		#print map_word.get_shape()
		#print map_coarse.get_shape()
		self.loss = - tf.reduce_sum(map_word) - tf.reduce_sum(map_coarse)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

	def _create_grad_accumul_op(self):
		tvs = tf.trainable_variables()
		accum_vars = [\
				tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)\
						 for tv in tvs]
		zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
		gvs = self.optimizer.compute_gradients(self.loss, tvs)
		accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
		self.zero_grads = zero_ops
		self.grad_accumul = accum_ops
		self.grad_apply = self.optimizer.apply_gradients(\
			[(accum_vars[i].assign(accum_vars[i]), gv[1]) for i, gv in enumerate(gvs)])


	def partial_fit(self,X_word,X_coarse): #
		""" Train model
			
		X_word is a list of utterances of ONE dialogue
		X_coarse is the corresponding list of coarse utterances

		The length of the two lists must be the same

		TEMPORARY IMPLEMENTATION: the model is updated after each utterances' pair
		"""

		# reset context state
		self.sess.run(self.zero_grads)
		for u_id in xrange(len(X_word)-1):
			if u_id == 0:
				prev_coarse = np.zeros([1,self.coarse_context_hid_size])
				prev_word = np.zeros([1,self.word_context_hid_size])
				total_loss = 0
			feed_dict = {\
				self.input_word: X_word[u_id],\
				self.input_coarse: X_coarse[u_id],\
				self.target_labels_word: X_word[u_id+1],\
				self.target_labels_coarse: X_coarse[u_id+1],\
				self.output_word_length: [len(X_word[u_id+1])],\
				self.output_coarse_length: [len(X_coarse[u_id+1])],\
				self.previous_coarse_context: prev_coarse, \
				self.previous_word_context: prev_word\
			}
			prev_coarse,prev_word,loss,_=self.sess.run(\
					(\
						self.context_coarse_current_hidden,\
						self.context_word_current_hidden,\
						self.loss,\
						self.grad_accumul),\
					feed_dict=feed_dict)
			prev_coarse = [prev_coarse]
			prev_word = [prev_word]
			total_loss += loss
			#print prev_coarse,result
			#print self.context_persistent_coarse
			#self.sess.run(self.reset_counters,feed_dict=feed_dict)
		print total_loss
		self.sess.run(self.grad_apply)


if __name__ == "__main__":
	config = Configuration()

	config.word_emb_size = 100
	config.coarse_emb_size = 101
	config.word_vocab_size = 20
	config.coarse_vocab_size = 21
	config.word_encoder_hid_size = 104
	config.coarse_encoder_hid_size = 105
	config.word_context_hid_size = 106
	config.coarse_context_hid_size = 107
	config.prediction_encoder_hid_size = 108
	config.learning_rate = 0.0001

	# dummy input
	x_w = [ [5,3,1,2,5],[10,2,3,5,2,4,5],[2,4,2,1] ]
	x_z = [ [1,1,4],[9,4,0,3],[3,4,5,2,1] ]
	#x_w = [ [5,3,1,2,5],[5,3,1,2,5],[5,3,1,2,5] ]
	#x_z = [ [1,1,4],[1,1,4],[1,1,4] ]

	model = MRRNN(config)
	for i in xrange(100):
		model.partial_fit(x_w,x_z)