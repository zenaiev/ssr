# python3 convert-d2-items.py d2-items-input.txt d2-drop-words.txt d2-drop-items.txt d2-drop-words-for-dict.txt

import sys

assert len(sys.argv) == 5
words = []
items = []
letters = {}
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
        for l in w.lower():
          if l in letters:
            letters[l] = letters[l] + 1
          else:
            letters[l] = 1
letters = {k: v for k, v in sorted(letters.items(), key=lambda item: item[1])}
print(letters) # rarest are q, z -> use them to replace ' ' and '-'
with open(sys.argv[2], 'w') as fout:
  for w in words:
    #print(w)
    fout.write(w + '\n')
with open(sys.argv[3], 'w') as fout:
  with open(sys.argv[4], 'w') as fout_single:
    for w in items:
      #print(w)
      fout.write(w + '\n')
      w_single = w.lower()
      w_single = w_single.replace(' ', 'q').replace('-', 'z')
      fout_single.write(w_single + '\n')
