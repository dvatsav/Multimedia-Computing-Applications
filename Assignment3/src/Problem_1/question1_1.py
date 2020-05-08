import numpy as np 
import nltk
import os
import sys
from nltk.corpus import stopwords, abc as abc_corpus
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
torch.manual_seed(42)
np.random.seed(42)

from tqdm import tqdm
import pickle

import nltk
nltk.download('abc')
nltk.download('punkt')
stem = True

window_size = 3
embedding_size = 200

n_epochs = 100
lr = 0.005
batch_size = 2000
momentum = 0.9

snapshot_dir = "snapshots"
viz_words_per_epoch = 20
viz_sim_words_per_query = 20
viz_interval = 5
snapshot_interval = 10
training_resume = False
training_resume_model = "" # Just the file name without snapshot dir path

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print ("Using device {}".format(device))

if not os.path.exists(snapshot_dir):
	os.makedirs(snapshot_dir, exist_ok=True)

class word2vec:
	def __init__(self, corpus):
		self.words = []
		self.window_size = window_size
		self._get_words(corpus)
		self.vocab_size = len(self.words)
		self.word_idx = {word: i for i, word in enumerate(self.words)}
		self.rev_word_idx = {i: word for i, word in enumerate(self.words)}

	def _get_words(self, corpus):
		for sentence in corpus:
			self.words += sentence
		self.words = list(set(self.words))

	def word_to_onehot(self, word):
		vector = np.zeros(self.vocab_size)
		vector[self.word_idx[word]] = 1
		return vector

	def onehot_to_word(self, vector):
		return self.words[np.where(vector==1)[0]]


class cbow(nn.Module):
	def __init__(self, w2v):
		super(cbow, self).__init__()
		self.w2v = w2v
		self.embedding = nn.Embedding(self.w2v.vocab_size, embedding_size)
		self.linear = nn.Linear(embedding_size, self.w2v.vocab_size)
		
	def forward(self, x):
		x = self.embedding(x)
		x = torch.sum(x, dim=1)
		x = self.linear(x)
		return x

	def get_embeddings(self, inp):
		return self.embedding(inp)


class skip_gram(nn.Module):
	def __init__(self, w2v):
		super(skip_gram, self).__init__()
		self.w2v = w2v
		self.embedding = nn.Embedding(self.w2v.vocab_size, embedding_size)
		self.linear1 =  nn.Linear(embedding_size, embedding_size)
		self.linear2 = nn.Linear(embedding_size, self.w2v.vocab_size)
		

	def forward(self, x):
		x = self.embedding(x)
		x = self.linear1(x)
		x = self.linear2(x)
		return x

	def get_embeddings(self, inp):
		return self.embedding(inp)

def get_similar_words(embeddings, w2v, word_idx, n):
	embeddings = embeddings.cpu().detach().numpy()
	pairwise_cosine_sim = cosine_similarity(embeddings)
	cosine_sims = pairwise_cosine_sim[word_idx]
	top_n = np.argsort(cosine_sims)[::-1][:n+1]
	return top_n, [w2v.rev_word_idx[idx] for idx in top_n], embeddings[top_n]


def clean_corpus(corpus):
	nltk.download('stopwords')
	stop_words = set(stopwords.words('english'))
	
	porter_stemmer = PorterStemmer() if stem else None
	
	for idx, sentence in enumerate(corpus):
		sentence = [word.lower() for word in sentence if word.isalpha()]
		sentence = [word for word in sentence if word not in stop_words]
		if stem:
			sentence = [porter_stemmer.stem(word) for word in sentence]
		corpus[idx] = sentence

	return corpus

def generate_skp_train_data(corpus, w2v):
	train_x = []
	train_y = []
	for sentence in corpus:
		sentence_len = len(sentence)
		for idx, word in enumerate(sentence):
			
			target_word = w2v.word_idx[word]

			for window_idx in range(max(0, idx - w2v.window_size), min(sentence_len, idx+w2v.window_size+1)):
				if window_idx != idx:
					context_word = w2v.word_idx[sentence[window_idx]]
					train_x.append(target_word)
					train_y.append(context_word)
	
	train_x = np.array(train_x)
	train_y = np.array(train_y)
	return train_x, train_y

def generate_cbow_train_data(corpus, w2v):
	train_x = []
	train_y = []
	for sentence in corpus:
		sentence_len = len(sentence)
		for idx, word in enumerate(sentence):
			
			target_word = w2v.word_idx[word]
			cur_train = []
			for window_idx in range(max(0, idx - w2v.window_size), min(sentence_len, idx+w2v.window_size+1)):
				if window_idx != idx:
					context_word = w2v.word_idx[sentence[window_idx]]
					cur_train.append(context_word)
			if len(cur_train) == 2 * window_size:
				train_y.append(target_word)
				train_x.append(cur_train)
	
	train_x = np.array(train_x)
	train_y = np.array(train_y)
	return train_x, train_y

def train(model, generate_data, corpus, w2v):
	if training_resume:
		model.load_state_dict(torch.load(os.path.join(snapshot_dir, training_resume_model)))
	model.train()
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss() # combines LogSoftmax and NLLloss
	criterion.to(device)
	train_x, train_y = generate_data(corpus, w2v)

	train_size = train_x.shape[0]
	
	best_model = deepcopy(model)
	min_loss = np.inf
	for epoch in range(1, n_epochs+1):
		
		cur_loss = 0
		for idx in tqdm(range(0, train_size, batch_size)):
			inp = torch.from_numpy(train_x[idx:idx+batch_size]).long().to(device)
			
			out = torch.from_numpy(train_y[idx:idx+batch_size]).long().to(device)
			
			optimizer.zero_grad()
			pred = model(inp)
			loss = criterion(pred, out)
			
			loss.backward()
			optimizer.step()

			cur_loss += loss.data

		if cur_loss < min_loss:
			min_loss = cur_loss
			best_model = deepcopy(model)

		if epoch % snapshot_interval == 0:
			file_name = "snapshot_{}_epoch{}.pth".format(mode, epoch)
			snapshot_path = os.path.join(snapshot_dir, file_name)
			torch.save(model.state_dict(), snapshot_path)

		print ("epoch = {} Loss = {} Min Loss = {}".format(epoch, cur_loss, min_loss))


		writer.add_scalar('loss', cur_loss, epoch)

		if epoch % viz_interval == 0:

			random_viz_idxs = np.random.choice(np.arange(w2v.vocab_size), viz_words_per_epoch)
			embeddings = model.get_embeddings(torch.from_numpy(np.arange(w2v.vocab_size)).long().to(device))
			viz_embeddings = np.empty(((viz_sim_words_per_query+1)*viz_words_per_epoch, embedding_size))
			viz_metadata = []
			for viz_idx in range(viz_words_per_epoch):        	
				sim_idxs, sim_words, sim_embeddings = get_similar_words(embeddings.clone(), w2v, random_viz_idxs[viz_idx], viz_sim_words_per_query)
				viz_embeddings[viz_idx*(viz_sim_words_per_query+1):(viz_idx+1)*(viz_sim_words_per_query+1)] = sim_embeddings
				viz_metadata += [viz_idx] * (viz_sim_words_per_query+1)

			writer.add_embedding(viz_embeddings, global_step=epoch, metadata=viz_metadata)
	
	print ("Saving best model")
	torch.save(best_model.state_dict(), os.path.join(snapshot_dir, "best_model_{}.pth".format(mode)))
	
	hparam_dict = {
			'Window Size': window_size,
			'Context Size': window_size*2,
			'Embedding Size': embedding_size,
			'Number of Epochs': n_epochs,
			'Learning Rate': lr,
			'Batch Size': batch_size,
			'Vocabulary Size': w2v.vocab_size,
			'Mode': mode,
			'Stem words': stem,
			'Snapshot directory': snapshot_dir
		}

	metric_dict = {
		'Minimum Loss': min_loss
	}

	writer.add_hparams(
		hparam_dict, metric_dict	
	)

	with open(os.path.join(snapshot_dir, "w2v_model_{}.pkl".format(mode)), 'wb') as output:
		pickle.dump(w2v, output, pickle.HIGHEST_PROTOCOL)



def main():
	corpus = [sentence for sentence in abc_corpus.sents()]
	corpus = clean_corpus(corpus)
	corpus_size = len(corpus)

	w2v = word2vec(corpus)

	print ("Corpus Size - {}".format(corpus_size))
	print ("Vocab size - {}".format(w2v.vocab_size))
	
	if mode == "cbow":
		model = cbow(w2v)
		train(model, generate_cbow_train_data, corpus, w2v)
	elif mode == "skp":
		model = skip_gram(w2v)
		train(model, generate_skp_train_data, corpus, w2v)

if __name__ == '__main__':
	cargs = len(sys.argv)
	if cargs != 2:
		print ("Usage: python question1_1.py <skp/cbow>")
		sys.exit()
	mode = sys.argv[1]
	if mode not in ['skp', 'cbow']:
		print ("Usage: python question1_1.py <skp/cbow>")
		sys.exit()


	writer = SummaryWriter("{}_logs".format(mode))
	main()