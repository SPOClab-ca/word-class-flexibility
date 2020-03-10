"""
Script to process multilingual Wikipedia files and save data into Python Pickle format.

Usage:
  python scripts/process_wikipedia.py \
    --wiki_dir=data/wiki/ \
    --ud_dir=data/ud_all/ud-treebanks-v2.5/ \
    --dest_dir=data/processed/ \
    --lang=nl \
    --tokens 1000
"""

import collections
import argparse
import pickle
import multiprocessing
import random
import glob
import tqdm
import spacy_udpipe
import src.corpus

LANGUAGES = {
  'ar': 'Arabic',
  'da': 'Danish',
  'de': 'German',
  'en': 'English',
  'es': 'Spanish',
  'fr': 'French',
  'he': 'Hebrew',
  'id': 'Indonesian',
  'it': 'Italian',
  'ja': 'Japanese',
  'la': 'Latin',
  'nl': 'Dutch',
  'nn': 'Norwegian',
  'pl': 'Polish',
  'ro': 'Romanian',
  'sl': 'Slovenian',
  'zh': 'Chinese',
}

parser = argparse.ArgumentParser()
parser.add_argument('--wiki_dir', type=str)
parser.add_argument('--ud_dir', type=str)
parser.add_argument('--dest_dir', type=str)
parser.add_argument('--lang', type=str)
parser.add_argument('--tokens', type=int)
args = parser.parse_args()
print(args)

# Download language models
spacy_udpipe.download(args.lang)

# Process UD to get lemma mappings
lang_name_full = LANGUAGES[args.lang]
ud_treebanks = src.corpus.group_treebanks_by_language(args.ud_dir)[lang_name_full]
print('Processing UD:', lang_name_full)
ud_corpus = src.corpus.POSCorpus.create_from_ud(data_file_list=ud_treebanks)

# Process Wikipedia
ALL_LINES = []
for fname in glob.glob(args.wiki_dir + '/' + args.lang + '/**/*'):
  print(fname)
  with open(fname) as f:
    contents = f.read()
  for line in contents.split('\n'):
    if line.strip() == '':
      continue
    if line.startswith('<doc') or line.startswith('</doc'):
      continue
    ALL_LINES.append(line)

print('Lines:', len(ALL_LINES))
random.seed(12345)
random.shuffle(ALL_LINES)

nlp = spacy_udpipe.load(args.lang)
def process_line(line):
  return nlp(line)

pool = multiprocessing.Pool()
token_count = 0
pos_counts = collections.defaultdict(int)

sentences = []
for doc in tqdm.tqdm(pool.imap(process_line, ALL_LINES)):
  if token_count > args.tokens:
    break
  for sent in doc.sents:
    if len(sent) < 5:
      continue
    sentence = []
    for token in sent:
      token_count += 1
      # Try using UD lemma merger, otherwise fallback to udpipe lemma output
      token_lemma = ud_corpus.get_merged_lemma_for_word(token.text.lower())
      if token_lemma is None:
        token_lemma = token.lemma_
      sentence.append({'word': token.text, 'lemma': token_lemma, 'pos': token.pos_})
      pos_counts[token.pos_] += 1
    sentences.append(sentence)

print('Tokens:', token_count)
for tok, tokcount in pos_counts.items():
  print('%s: %d' % (tok, tokcount))

save_file = args.dest_dir + '/' + args.lang + '.pkl'
print('Saving:', save_file)
with open(save_file, 'wb') as f:
  pickle.dump(sentences, f)
