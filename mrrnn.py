import tensorflow as tf
import numpy as np

class Configuration:
	def __init__(self):
		self.word_emb_size = 50
		self.coarse_emb_size = 50
		self.word_vocab_size = 10
		self.coarse_vocab_size = 10
		self.word_encoder_hid_size = 100
		self.coarse_encoder_hid_size = 100
		self.word_context_hid_size = 100
		self.coarse_context_hid_size = 100
		self.coarse_decoder_hid_size = 100
		self.word_decoder_hid_size = 100
		self.prediction_encoder_hid_size = 100
		self.learning_rate = 0.0002
		self.end_of_word_utt = 0
		self.end_of_coarse_utt = 0
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
		self.coarse_decoder_hid_size = config.coarse_decoder_hid_size
		self.word_decoder_hid_size = config.word_decoder_hid_size
		self.prediction_encoder_hid_size = config.prediction_encoder_hid_size
		self.learning_rate = config.learning_rate
		self.end_of_word_utt = config.end_of_word_utt
		self.end_of_coarse_utt = config.end_of_coarse_utt

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

		self.eou_coarse = tf.constant(config.end_of_coarse_utt,dtype=tf.int64)
		self.eou_word = tf.constant(config.end_of_word_utt,dtype=tf.int64)

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

		# saver
		self.saver = tf.train.Saver()

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
		return it[0] < self.output_coarse_length[0]
	
	def decoder_word_condition(self,it,outputs,hidden):
		return it[0] < self.output_word_length[0]

	def decoder_coarse_func(self,it,outputs,hidden):
		# hidden state is multiplied with the context hidden state 
		out,hidden = self.decoder_coarse_cell(\
				tf.pack([tf.concat(0,[outputs[-1,:],self.context_coarse_current_hidden])]),\
				tf.pack([hidden]))
		
		
		outputs = tf.concat(0,[outputs,out])
		return it+1,outputs,hidden[0,:]

	def decoder_word_func(self,it,outputs,hidden):
		# hidden state is multiplied with the context hidden state 
		out,hidden = self.decoder_word_cell(\
				tf.pack([tf.concat(0,[outputs[-1,:],self.word_context_concatenation])]),\
				tf.pack([hidden]))
		
		outputs = tf.concat(0,[outputs,out])
		return it+1,outputs,hidden[0,:]

	def _create_coarse_decoder(self):
		with tf.variable_scope("coarse_decoder"):
			self.decoder_coarse_cell =\
					 tf.nn.rnn_cell.GRUCell(self.coarse_decoder_hid_size)
			self.iter_decoder_coarse = tf.constant(0)
			self.iter_decoder_coarse_test = tf.constant(0)
			train_decoder = tf.while_loop(\
				self.decoder_coarse_condition,\
				self.decoder_coarse_func,\
				[	self.iter_decoder_coarse,\
					tf.zeros([1,self.coarse_decoder_hid_size]),\
					tf.ones([self.coarse_decoder_hid_size])],\
				shape_invariants=[\
								self.iter_decoder_coarse.get_shape(),\
								tf.TensorShape([None,self.coarse_decoder_hid_size]),\
								tf.TensorShape([self.coarse_decoder_hid_size])])		
			return train_decoder[1][1:]

	def _create_word_decoder(self):
		with tf.variable_scope("word_decoder"):
			self.decoder_word_cell =\
					 tf.nn.rnn_cell.GRUCell(self.word_decoder_hid_size)
			self.iter_decoder_word = tf.constant(0)
			self.iter_decoder_word_test = tf.constant(0)
			train_decoder = tf.while_loop(\
				self.decoder_word_condition,\
				self.decoder_word_func,\
				[	self.iter_decoder_word,\
					tf.zeros([1,self.word_decoder_hid_size]),\
					tf.ones([self.word_decoder_hid_size])],\
				shape_invariants=[\
								self.iter_decoder_word.get_shape(),\
								tf.TensorShape([None,self.word_decoder_hid_size]),\
								tf.TensorShape([self.word_decoder_hid_size])])
			
		return train_decoder[1][1:]

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
		self.loss = (tf.reduce_sum(map_word) + tf.reduce_sum(map_coarse))
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
			[(accum_vars[i], gv[1])\
					 for i, gv in enumerate(gvs)])

	def split_utterances(self,X_word,X_coarse):
		n_dialogues = len(X_word)
		dialogues_word = []
		dialogues_coarse = []
		for i in xrange(n_dialogues):
			# split utterances
			end_of_utts_word = np.where( np.array(X_word[i]) == self.end_of_word_utt )[0]
			end_of_utts_coarse = np.where( np.array(X_coarse[i]) == self.end_of_coarse_utt )[0]
			current_dialogue_word = np.split( X_word[i], end_of_utts_word[:-1] + 1 )
			current_dialogue_coarse = np.split( X_coarse[i], end_of_utts_coarse[:-1] + 1 )
			dialogues_word.append(current_dialogue_word)
			dialogues_coarse.append(current_dialogue_coarse)

		return dialogues_word,dialogues_coarse

	def partial_fit(self,X_word,X_coarse): #
		""" Train model
			
		X_word is a batch of dialogues
		X_coarse is the corresponding list of coarse utterances

		The length of the two lists must be the same

		TEMPORARY IMPLEMENTATION: the model is updated after each utterances' pair
		"""

		# reset gradient computation
		self.sess.run(self.zero_grads)
		# split dialogues
		dialogues_word, dialogues_coarse = self.split_utterances(X_word,X_coarse)
		# train
		for dialogue in xrange(len(dialogues_word)):
			for u_id in xrange(len(dialogues_word[dialogue])-1):
				if u_id == 0:
					prev_coarse = np.zeros([1,self.coarse_context_hid_size])
					prev_word = np.zeros([1,self.word_context_hid_size])
				feed_dict = {\
					self.input_word: dialogues_word[dialogue][u_id],\
					self.input_coarse: dialogues_coarse[dialogue][u_id],\
					self.target_labels_word: dialogues_word[dialogue][u_id+1],\
					self.target_labels_coarse: dialogues_coarse[dialogue][u_id+1],\
					self.output_word_length: [len(dialogues_word[dialogue][u_id+1])],\
					self.output_coarse_length: [len(dialogues_coarse[dialogue][u_id+1])],\
					self.previous_coarse_context: prev_coarse, \
					self.previous_word_context: prev_word\
				}
				prev_coarse,prev_word,_=self.sess.run(\
						(\
							self.context_coarse_current_hidden,\
							self.context_word_current_hidden,\
							self.grad_accumul),\
						feed_dict=feed_dict)
				prev_coarse = [prev_coarse]
				prev_word = [prev_word]
		self.sess.run(self.grad_apply)

	def cost(self,X_word,X_coarse):
		dialogues_word, dialogues_coarse = self.split_utterances(X_word,X_coarse)
		for dialogue in xrange(len(dialogues_word)):
			for u_id in xrange(len(dialogues_word[dialogue])-1):
				if u_id == 0:
					prev_coarse = np.zeros([1,self.coarse_context_hid_size])
					prev_word = np.zeros([1,self.word_context_hid_size])
					total_loss = 0
				feed_dict = {\
					self.input_word: dialogues_word[dialogue][u_id],\
					self.input_coarse: dialogues_coarse[dialogue][u_id],\
					self.target_labels_word: dialogues_word[dialogue][u_id+1],\
					self.target_labels_coarse: dialogues_coarse[dialogue][u_id+1],\
					self.output_word_length: [len(dialogues_word[dialogue][u_id+1])],\
					self.output_coarse_length: [len(dialogues_coarse[dialogue][u_id+1])],\
					self.previous_coarse_context: prev_coarse, \
					self.previous_word_context: prev_word\
				}
				prev_coarse,prev_word,loss=self.sess.run(\
						(\
							self.context_coarse_current_hidden,\
							self.context_word_current_hidden,\
							self.loss),\
						feed_dict=feed_dict)
				prev_coarse = [prev_coarse]
				prev_word = [prev_word]
				total_loss += loss
		total_loss /= float(len(X_word))
		return total_loss 

	def generate(self,dialogue_word,dialogue_coarse,\
				max_coarse_generation=10,max_word_generation=10): #

		for u_id in xrange(len(dialogue_word)-1):
			if u_id == 0:
				prev_coarse = np.zeros([1,self.coarse_context_hid_size])
				prev_word = np.zeros([1,self.word_context_hid_size])
			feed_dict = {\
				self.input_word: dialogue_word[u_id],\
				self.input_coarse: dialogue_coarse[u_id],\
				self.target_labels_word: dialogue_word[u_id+1],\
				self.target_labels_coarse: dialogue_coarse[u_id+1],\
				self.output_word_length: [len(dialogue_word[u_id+1])],\
				self.output_coarse_length: [len(dialogue_coarse[u_id+1])],\
				self.previous_coarse_context: prev_coarse, \
				self.previous_word_context: prev_word\
			}
			prev_coarse,prev_word=self.sess.run(\
					(\
						self.context_coarse_current_hidden,\
						self.context_word_current_hidden),\
					feed_dict=feed_dict)
			prev_coarse = [prev_coarse]
			prev_word = [prev_word]
		u_id = len(dialogue_word)-1
		feed_dict = {\
			self.input_word: dialogue_word[u_id],\
			self.input_coarse: dialogue_coarse[u_id],\
			self.previous_coarse_context: prev_coarse, \
			self.previous_word_context: prev_word,\
			self.output_word_length: [max_word_generation],\
			self.output_coarse_length: [max_coarse_generation]\
		}
		w_logits = self.sess.run(self.logits_word,feed_dict=feed_dict)
		prediction = np.zeros([w_logits.shape[0]],dtype=int)
		for i in xrange(len(prediction)):
			probs = np.exp(w_logits[i])
			probs /= probs.sum()
			prediction[i] = np.random.choice(self.word_vocab_size,p=probs)
		return prediction

	def save(self,file_path):
		self.saver.save(self.sess,file_path)

	def restore(self,file_path):
		self.saver.restore(self.sess,file_path)


if __name__ == "__main__":

	config = Configuration()
	config.word_vocab_size = 11
	config.coarse_vocab_size = 11
	config.end_of_word_utt = 10
	config.end_of_coarse_utt = 10

	# dummy input
	x_w = [ [ 3,5,8,3,10,2,3,5,9,9,10,8,7,6,1,0,10] ]
	x_z = [ [ 2,4,1,2,7,8,10,1,6,8,8,10,9,0,0,1,10] ]
	x_w_test = [ [3,5,8,3,10],[2,3,5,9,9,10] ] 
	x_z_test = [ [2,4,1,2,7,8,10],[1,6,8,8,10] ]

	model = MRRNN(config)
	for i in xrange(1000):
		model.partial_fit(x_w,x_z)
		loss = model.cost(x_w,x_z)
		print loss
		prediction = model.generate(x_w_test,x_z_test,5,6)
		# should learn to predict the sequence [8,7,6,1,1,10]
		print prediction