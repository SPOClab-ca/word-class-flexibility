"""
Hacky script to process PDF tables in Barli Bram's thesis (2011)
To run: python3 bram_thesis_clean.py > attestation_dates.csv
"""

# Extracted from PDF (Appendix 1)
with open('bram-thesis-data/noun_first.txt') as f:
  NOUN_FIRST = f.read()

# Extracted from PDF (Appendix 2)
with open('bram-thesis-data/same_time.txt') as f:
  SAME_TIME = f.read()

# Extracted from PDF (Appendix 3)
with open('bram-thesis-data/verb_first.txt') as f:
  VERB_FIRST = f.read()

print('word,noun_date,verb_date')

# Either (d1=noun, d2=verb, d3=none) or (d1=noun, d2=intrans_verb, d3=trans_verb)
for line in NOUN_FIRST.split('\n') + SAME_TIME.split('\n'):
  line = line.split()
  if line == []:
    continue
  w = line[1]
  d1 = int(line[2])
  d2 = int(line[3])
  try:
    d3 = int(line[4])
  except:
    d3 = 99999
  if d1 < 30:
    d1 *= 100
  if d2 < 30:
    d2 *= 100
  if d3 < 30:
    d3 *= 100
  d2 = min(d2, d3)
  assert d1 <= d2
  assert d1 != 99999
  assert d2 != 99999
  print('%s,%d,%d' % (w, d1, d2))

# Either (d1=verb, d2=noun, d3=none) or (d1=intrans_verb, d2=trans_verb, d3=noun)
for line in VERB_FIRST.split('\n'):
  line = line.split()
  if line == []:
    continue
  w = line[1]
  d1 = int(line[2])
  d2 = int(line[3])
  try:
    d3 = int(line[4])
  except:
    d3 = 99999
  if d1 < 30:
    d1 *= 100
  if d2 < 30:
    d2 *= 100
  if d3 < 30:
    d3 *= 100
  # Swap so that d1 is noun, d2 is min(trans, intrans)
  if d3 == 99999:
    d1, d2 = d2, d1
  else:
    d1, d2 = d3, min(d1, d2)
  assert d1 >= d2
  assert d1 != 99999
  assert d2 != 99999
  print('%s,%d,%d' % (w, d1, d2))

# Flexible but missing from Bram's list. The following data I manually scraped from OED.
print('count,1338,1380')
print('estimate,1630,1609')
print('use,1225,1300')
print('record,1399,1340')
print('increase,1374,1380')
print('start,1569,1570')
print('stress,1440,1859')
print('match,1549,1393')
print('miss,1175,1000')
print('process,1325,1878')
print('measure,1375,1382')
print('tie,733,1000')
print('break,1325,1000')
print('account,1300,1426')
print('back,885,1548')
print('face,1300,1570')
print('matter,1200,1553')
print('press,1100,1330')
print('roll,1200,1325')
print('sort,1380,1358')
print('wave,1526,1380')
print('deal,1000,1000')
print('present,1200,1300')
print('set,1561,800')
print('suit,1420,1577')
print('wind,897,1374')
print('address,1539,1325')
print('bear,1000,1000')
print('head,900,1390')
print('park,1222,1846')
print('ring,900,1000')
print('state,1200,1641')
print('store,1471,1264')
print('train,1606,1531')
