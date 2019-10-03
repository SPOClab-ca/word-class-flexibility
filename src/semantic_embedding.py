import numpy as np
import pandas as pd
import allennlp.commands.elmo
from gensim.models import KeyedVectors
import sklearn.metrics
import tqdm

GLOVE_LOCATION = '/h/bai/moar/snap/data/glove.840B.300d.txt'
#GLOVE_LOCATION = '../data/glove/glove.6B.100d.txt'

class SemanticEmbedding:
  def __init__(self, sentences, generate_elmo=True):
    self.sentences = sentences
    self.glove_vectors = KeyedVectors.load_word2vec_format(GLOVE_LOCATION, limit=100000)
    #self.glove_vectors = KeyedVectors.load_word2vec_format(GLOVE_LOCATION)

    if generate_elmo:
      # Generate ELMo embeddings here, because we must batch to be efficient
      self.elmo = allennlp.commands.elmo.ElmoEmbedder(cuda_device=0)
      data_as_tokens = [[t['word'] for t in sentence] for sentence in sentences]

      BATCH_SIZE = 64
      self.elmo_embeddings = []
      for ix in tqdm.tqdm(range(0, len(data_as_tokens), BATCH_SIZE)):
        batch = data_as_tokens[ix : ix+BATCH_SIZE]
        batch_embeddings = self.elmo.embed_batch(batch)
        # Only take embeddings from last ELMo layer
        batch_embeddings = [x[-1] for x in batch_embeddings]
        self.elmo_embeddings.extend(batch_embeddings)

  def get_elmo_embeddings_for_lemma(self, lemma):
    noun_embeddings = []
    verb_embeddings = []

    for sentence_ix in range(len(self.sentences)):
      token_list = self.sentences[sentence_ix]
      embeddings = self.elmo_embeddings[sentence_ix]
      for i in range(len(token_list)):
        if token_list[i]['lemma'] == lemma:
          if token_list[i]['pos'] == 'NOUN':
            noun_embeddings.append(embeddings[i])
          elif token_list[i]['pos'] == 'VERB':
            verb_embeddings.append(embeddings[i])

    noun_embeddings = np.vstack(noun_embeddings)
    verb_embeddings = np.vstack(verb_embeddings)
    return noun_embeddings, verb_embeddings

  def get_elmo_nv_similarity(self, lemma):
    noun_embeddings, verb_embeddings = self.get_elmo_embeddings_for_lemma(lemma)
    
    avg_noun_embedding = np.mean(noun_embeddings, axis=0)
    avg_verb_embedding = np.mean(verb_embeddings, axis=0)

    return float(
      sklearn.metrics.pairwise.cosine_similarity(
        avg_noun_embedding[np.newaxis,:],
        avg_verb_embedding[np.newaxis,:]
      )
    )

  def get_glove_nv_similarity(self, lemma, context=8):
    noun_embeddings = []
    verb_embeddings = []

    for sentence_ix in range(len(self.sentences)):
      token_list = self.sentences[sentence_ix]

      # Find index of lemma
      lemma_ix = None
      lemma_pos = None
      for ix, token in enumerate(token_list):
        if token['lemma'] == lemma:
          lemma_ix = ix
          lemma_pos = token['pos']
          break
      if lemma_ix is None:
        continue

      context_embeddings = []
      for ix, token in enumerate(token_list):
        if ix != lemma_ix and abs(ix - lemma_ix) <= context:
          try:
            context_embeddings.append(self.glove_vectors[token['word'].lower()])
          except Exception:
            pass
      if len(context_embeddings) > 0:
        context_embeddings = np.vstack(context_embeddings)
        if lemma_pos == 'NOUN':
          noun_embeddings.append(np.mean(context_embeddings, axis=0))
        elif lemma_pos == 'VERB':
          verb_embeddings.append(np.mean(context_embeddings, axis=0))

    noun_embeddings = np.vstack(noun_embeddings)
    verb_embeddings = np.vstack(verb_embeddings)

    avg_noun_embedding = np.mean(noun_embeddings, axis=0)
    avg_verb_embedding = np.mean(verb_embeddings, axis=0)

    return float(
      sklearn.metrics.pairwise.cosine_similarity(
        avg_noun_embedding[np.newaxis,:],
        avg_verb_embedding[np.newaxis,:]
      )
    )
