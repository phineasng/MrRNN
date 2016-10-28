
import pickle
from mrrnn import MRRNN
from mrrnn import Configuration

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

	# import test data
	test_word_path = "./data/Test.dialogues.pkl"
	with open(test_word_path,"r") as file:
		test_word_data = pickle.load(file)
	test_coarse_path = "./data/abstract.test.dialogues.pkl"
	with open(test_coarse_path,"r") as file:
		test_coarse_data = pickle.load(file)

	config = Configuration()
	config.word_vocab_size = len(vocab_word)
	config.coarse_vocab_size = len(vocab_coarse)
	config.end_of_word_utt = LANGUAGE_END 
	config.end_of_coarse_utt = COARSE_END
	# create model
	model = MRRNN(config)
	N_dialogue = 7

	file_name = "./ckpts/training_5/trained.ckpt"
	model.restore(file_name)
	dial_word,dial_coarse = model.split_utterances(test_word_data[:N_dialogue+1],test_coarse_data[:N_dialogue+1])
	prediction = model.generate(dial_word[N_dialogue],dial_coarse[N_dialogue],5,20)
	# print dial_word[0]
	for curr_utt in xrange(len(dial_word[N_dialogue])):
		curr_str = ""
	 	for k in xrange(len(dial_word[N_dialogue][curr_utt])-1):
	 		curr_str += vocab_word[dial_word[N_dialogue][curr_utt][k]][0] + " "
	 	print curr_str

	curr_str = ""
	for k in xrange(len(prediction)):
		curr_str += vocab_word[prediction[k]][0] + " "
	print curr_str