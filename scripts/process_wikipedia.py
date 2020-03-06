"""(Work in progress)
Script to process multilingual Wikipedia files and save data into Python Pickle format.

Usage:
  python scripts/process_wikipedia.py \
    --dir=data/wiki/nl \
    --to_file=data/processed/nl.pkl \
    --sample 1000
"""

import collections
import argparse
import pickle
import multiprocessing
import random
import glob
import tqdm
import spacy_udpipe

# Download language models
#for lang in 'ar da en es fr he id it ja la nl nn pl ro sl zh'.split():
#  spacy_udpipe.download(lang)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--to_file', type=str)
parser.add_argument('--sample', type=int)
args = parser.parse_args()
print(args)

ALL_LINES = []
for fname in glob.glob(args.dir + '/**/*'):
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

if args.sample:
  random.seed(12345)
  ALL_LINES = random.sample(ALL_LINES, k=args.sample)

nlp = spacy_udpipe.load("en")
def process_line(line):
  return nlp(line)

pool = multiprocessing.Pool()
token_count = 0
pos_counts = collections.defaultdict(int)

for doc in tqdm.tqdm(pool.imap(process_line, ALL_LINES), total=len(ALL_LINES)):
  for sent in doc.sents:
    for token in sent:
      token_count += 1
      #print(token.text, token.pos_)
      pos_counts[token.pos_] += 1

print('Tokens:', token_count)
for tok, tokcount in pos_counts.items():
  print('%s: %d' % (tok, tokcount))
