# python3 convert-d2-items.py d2-items-input.txt d2-drop-words.txt d2-drop-items.txt

import sys

assert len(sys.argv) == 4
words = []
items = []
with open(sys.argv[1]) as fin:
  lines = fin.readlines()
  for l in lines:
    if l.startswith('*') or len(l) <= 1:
      continue
    #l = l.rstrip()
    if ',' in l:
      l = l.split(',')[1]
    if '(' in l:
      l = l.split('(')[0]
    l = l.strip()
    if len(l) == 0:
      continue
    items.append(l)
    l = l.replace('-', ' ')
    for w in l.split(' '):
      if w not in words:
        words.append(w)
with open(sys.argv[2], 'w') as fout:
  for w in words:
    #print(w)
    fout.write(w + '\n')
with open(sys.argv[3], 'w') as fout:
  for w in items:
    #print(w)
    fout.write(w + '\n')
