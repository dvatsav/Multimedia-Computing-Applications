import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
torch.manual_seed(42)
np.random.seed(42)

from tqdm import tqdm
import pickle

w2v_skp_path = "snapshots/w2v_model_skp.pkl"
w2v_cbow_path = "snapshots/w2v_model_skp.pkl"
cbow_model_path = "snapshots/best_model_cbow.pth"
skp_model_path = "snapshots/best_model_skp.pth"

window_size = 3
embedding_size = 200
num_words_to_test = 10

device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
	return [w2v.rev_word_idx[idx] for idx in top_n]


def main():
	with open(w2v_skp_path, "rb") as pkl_file:
		w2v_skp = pickle.load(pkl_file)
	with open(w2v_cbow_path, "rb") as pkl_file:
		w2v_cbow = pickle.load(pkl_file)
	
	skp_model = skip_gram(w2v_skp)
	cbow_model = cbow(w2v_cbow)

	skp_model.load_state_dict(torch.load(skp_model_path, map_location=torch.device('cpu')))
	cbow_model.load_state_dict(torch.load(cbow_model_path, map_location=torch.device('cpu')))
	skp_model.to(device)
	cbow_model.to(device)
	skp_model.eval()
	cbow_model.eval()

	for (model, w2v, name) in [(skp_model, w2v_skp, "skp"), (cbow_model, w2v_cbow, "cbow")]:
		print ("Method - {}".format(name))

		random_idxs = np.random.choice(np.arange(w2v.vocab_size), num_words_to_test)
		embeddings = model.get_embeddings(torch.from_numpy(np.arange(w2v.vocab_size)).long().to(device))
		for test_idx in range(num_words_to_test):
			test_word = w2v.rev_word_idx[random_idxs[test_idx]]
			sim_words = get_similar_words(embeddings.clone(), w2v, random_idxs[test_idx], num_words_to_test)
			print ("Test word - {}, Related words - {}".format(test_word, str(sim_words)))
		print ("------------------------------------------------")
		
if __name__ == '__main__':
	main()