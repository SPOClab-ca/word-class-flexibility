import numpy as np
import pandas as pd
import allennlp.commands.elmo
import sklearn.metrics
import tqdm

class SemanticEmbedding:
  def __init__(self, sentences):
    self.sentences = sentences
    self.elmo = allennlp.commands.elmo.ElmoEmbedder(cuda_device=0)

    # Generate ELMo embeddings here, because we must batch to be efficient
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

  def get_nv_cosine_similarity(self, lemma):
    noun_embeddings, verb_embeddings = self.get_elmo_embeddings_for_lemma(lemma)
    
    avg_noun_embedding = np.mean(noun_embeddings, axis=0)
    avg_verb_embedding = np.mean(verb_embeddings, axis=0)

    return float(
      sklearn.metrics.pairwise.cosine_similarity(
        avg_noun_embedding[np.newaxis,:],
        avg_verb_embedding[np.newaxis,:]
      )
    )
