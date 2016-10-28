import pickle
from mrrnn import MRRNN
from mrrnn import Configuration
import random

# TOKEN_ID for end of utterance
LANGUAGE_END = 18575
COARSE_END = 10

if __name__ == "__main__":

	# import dictionaries
	dictionary_path = "./data/Dataset.dict.pkl"
	with open(dictionary_path,"r") as file:
		vocab_word = pickle.load(file)
		vocab_word = sorted( vocab_word, key=lambda tup: tup[1] )
	dictionary_coarse_path = "./data/abstract.dict.pkl"
	with open(dictionary_coarse_path,"r") as file:
		vocab_coarse = pickle.load(file)
		vocab_coarse = sorted( vocab_coarse, key=lambda tup: tup[1] )

	# import train data
	train_word_path = "./data/Training.dialogues.pkl"
	with open(train_word_path,"r") as file:
		train_word_data = pickle.load(file)
	train_coarse_path = "./data/abstract.train.dialogues.pkl"
	with open(train_coarse_path,"r") as file:
		train_coarse_data = pickle.load(file)

	# import test data
	test_word_path = "./data/Test.dialogues.pkl"
	with open(test_word_path,"r") as file:
		test_word_data = pickle.load(file)
	test_coarse_path = "./data/abstract.test.dialogues.pkl"
	with open(test_coarse_path,"r") as file:
		test_coarse_data = pickle.load(file)

	# import valid data
	valid_word_path = "./data/Validation.dialogues.pkl"
	with open(valid_word_path,"r") as file:
		valid_word_data = pickle.load(file)
	valid_coarse_path = "./data/abstract.valid.dialogues.pkl"
	with open(valid_coarse_path,"r") as file:
		valid_coarse_data = pickle.load(file)

	config = Configuration()
	config.word_vocab_size = len(vocab_word)
	config.coarse_vocab_size = len(vocab_coarse)
	config.end_of_word_utt = LANGUAGE_END 
	config.end_of_coarse_utt = COARSE_END
	config.learning_rate = 0.00005

	# create model
	model = MRRNN(config)

	batch_size = 100	
	print_every = 1
	n_train = len(train_word_data)
	n_epochs = 5
	max_train_data_id = n_train - batch_size

	save_dir = "./ckpts/training_5/"
	restore_file = "./ckpts/training_5/trained.ckpt"
	file_name = "./ckpts/training_5/trained.ckpt"

	if restore_file:
		model.restore(restore_file)

	loss = model.cost(valid_word_data[:10],valid_coarse_data[:10])
	print loss
	start_id = 103500

	for ep in xrange(n_epochs):
		it = 0
		while start_id < max_train_data_id:
			end_id = start_id + batch_size
			model.partial_fit(train_word_data[start_id:end_id],train_coarse_data[start_id:end_id])
			if not (it % print_every):
				loss = model.cost(valid_word_data[:10],valid_coarse_data[:10])
				model.save(file_name)
				print "{0} of {1}: {2}".format(start_id,n_train,loss)
			start_id += batch_size
			it += 1
		start_id = 0