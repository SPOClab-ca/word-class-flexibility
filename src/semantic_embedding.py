import numpy as np
import pandas as pd
import allennlp.commands.elmo
from gensim.models import KeyedVectors
import sklearn.metrics
import torch
import transformers
import tqdm

GLOVE_LOCATION = '/h/bai/moar/snap/data/glove.840B.300d.txt'

class SemanticEmbedding:
  def __init__(self, sentences):
    self.sentences = sentences

  def init_glove(self, limit=100000):
    """Load GloVe vectors
    @param limit = max number of words to load
    """
    self.glove_vectors = KeyedVectors.load_word2vec_format(GLOVE_LOCATION, limit=limit)

  def init_elmo(self, layer=2):
    """Init here because batching required for efficiency
    @param layer = one of [0, 1, 2]
    """
    self.elmo = allennlp.commands.elmo.ElmoEmbedder(cuda_device=0)
    data_as_tokens = [[t['word'] for t in sentence] for sentence in self.sentences]

    BATCH_SIZE = 64
    self.elmo_embeddings = []
    for ix in tqdm.tqdm(range(0, len(data_as_tokens), BATCH_SIZE)):
      batch = data_as_tokens[ix : ix+BATCH_SIZE]
      batch_embeddings = self.elmo.embed_batch(batch)
      # Only take embeddings from specified ELMo layer
      batch_embeddings = [x[layer] for x in batch_embeddings]
      self.elmo_embeddings.extend(batch_embeddings)

  def init_bert(self, layer=12):
    """Compute BERT embeddings in batch
    @param layer = integer between 0 and 12
    """
    data_as_sentences = [' '.join([t['word'] for t in sentence]) for sentence in self.sentences]
    self.bert_model = transformers.BertModel.from_pretrained(
      'bert-base-uncased',
      output_hidden_states=True
    ).cuda()
    self.bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # Helper function for padding input for BERT so that we can batch it
    # Truncate to 100 tokens at most to avoid memory problems
    def convert_to_bert_input(sentences):
      def pad_to_length(tokens, desired_len):
        return tokens + (['[PAD]'] * (desired_len - len(tokens)))
      bert_tokens = [self.bert_tokenizer.tokenize(sentence)[:100] for sentence in sentences]
      max_len = max([len(tokens) for tokens in bert_tokens])
      padded_tokens = [pad_to_length(tokens, max_len) for tokens in bert_tokens]
      padded_ids = [self.bert_tokenizer.encode(tokens) for tokens in padded_tokens]
      attn_mask = [[1 if token != '[PAD]' else 0 for token in tokens] for tokens in padded_tokens]
      return padded_tokens, padded_ids, attn_mask

    BATCH_SIZE = 16
    self.bert_embeddings = []
    self.bert_tokens = []
    for ix in tqdm.tqdm(range(0, len(data_as_sentences), BATCH_SIZE)):
      batch_sentences = data_as_sentences[ix : ix+BATCH_SIZE]
      padded_tokens, padded_ids, attn_mask = convert_to_bert_input(batch_sentences)
      self.bert_tokens.extend(padded_tokens)
      batch_embeddings = self.bert_model(
        torch.tensor(padded_ids).cuda(),
        attention_mask=torch.tensor(attn_mask).cuda()
      )[2][layer]
      self.bert_embeddings.extend(batch_embeddings.cpu().detach().numpy())

  def get_bert_embeddings_for_lemma(self, lemma):
    noun_embeddings = []
    verb_embeddings = []

    # Need to do a two-step matching process because the BERT embeddings correspond to
    # WordPiece tokens, which don't always match up with our tokens.
    for sentence_ix in range(len(self.sentences)):
      token_list = self.sentences[sentence_ix]
      wordpiece_tokens = self.bert_tokens[sentence_ix]
      embeddings = self.bert_embeddings[sentence_ix]

      assert len(wordpiece_tokens) == len(embeddings)

      # Find word in the sentence that has the lemma
      pos = None
      lemma_form = None
      for i in range(len(token_list)):
        if token_list[i]['lemma'] == lemma:
          if token_list[i]['pos'] in ['NOUN', 'VERB']:
            pos = token_list[i]['pos']
            lemma_form = token_list[i]['word']
            break

      # Get the embedding that matches token
      for i in range(len(wordpiece_tokens)):
        if wordpiece_tokens[i] == lemma_form:
          token_embedding = embeddings[i]
          if pos == 'NOUN':
            noun_embeddings.append(token_embedding)
            break
          elif pos == 'VERB':
            verb_embeddings.append(token_embedding)
            break

    noun_embeddings = np.vstack(noun_embeddings)
    verb_embeddings = np.vstack(verb_embeddings)
    return noun_embeddings, verb_embeddings


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


  def get_contextual_nv_similarity(self, lemma, method):
    """Compute cosine similarity between noun and verb embeddings, for a given lemma.
    Method can be 'elmo' or 'bert'.

    Returns: (n/v similarity, n-variation, v-variation)
    """
    if method == 'elmo':
      noun_embeddings, verb_embeddings = self.get_elmo_embeddings_for_lemma(lemma)
    elif method == 'bert':
      noun_embeddings, verb_embeddings = self.get_bert_embeddings_for_lemma(lemma)
    else:
      assert(False)
    
    avg_noun_embedding = np.mean(noun_embeddings, axis=0)
    avg_verb_embedding = np.mean(verb_embeddings, axis=0)
    nv_similarity = float(
      sklearn.metrics.pairwise.cosine_similarity(
        avg_noun_embedding[np.newaxis,:],
        avg_verb_embedding[np.newaxis,:]
      )
    )

    n_variation = np.mean(np.sum((noun_embeddings - avg_noun_embedding)**2, axis=1))
    v_variation = np.mean(np.sum((verb_embeddings - avg_verb_embedding)**2, axis=1))

    return nv_similarity, n_variation, v_variation

  def get_glove_nv_similarity(self, lemma, context=8, include_self=False):
    """
    @param context = window of context around word to use
    @param include_self = whether to include the lemma itself
    """
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
        # Decide if this token is in context window
        dist_to_lemma = abs(ix - lemma_ix)
        if include_self:
          is_in_context = dist_to_lemma <= context
        else:
          is_in_context = (ix != lemma_ix) and (dist_to_lemma <= context)

        if is_in_context:
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
