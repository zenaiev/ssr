from os import write
import pytesseract
import numpy as np
import imutils
import cv2
#from cv2 import cv2
import random
from imutils.object_detection import non_max_suppression
import time
import math
import pytesseract
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import time
import atexit
import traceback
import operator
import aspell

# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
def timeit(method):
  global _do_time
  if _do_time:
    def timed(*args, **kw):
      global _g_time
      #print args
      #print kw
      ts = time.time()
      result = method(*args, **kw)
      te = time.time()
      args_no_img = ['img' if type(a) is np.ndarray else a for a in args]
      kw_no_img = ['img' if type(a) is np.ndarray else a for a in kw]
      #name = '{}{:.2f}s {}{}{}'.format(' ' * len(traceback.extract_stack()), te - ts, method.__name__, args, kw)
      name = '{}{:.2f}s {}{}{}'.format(' ' * len(traceback.extract_stack()), te - ts, method.__name__, args_no_img, kw_no_img)
      #name = '{}{:.2f}s {}'.format(' ' * len(traceback.extract_stack()), te - ts, method.__name__)
      _g_time.append(name)
      #if 'log_time' in kw:
      #  name = kw.get('log_name', method.__name__.upper())
      #  kw['log_time'][name] = int((te - ts) * 1000)
      #else:
      #  print '%r  %2.2f ms' % \
      #    (method.__name__, (te - ts) * 1000)
      return result
    return timed
  else:
    return method

def print_all_time():
  global _do_time, _g_time, g_skip_atexit
  if g_skip_atexit:
    return
  if not _do_time:
    return
  if not _g_time:
    return
  print("Timing all:")
  sum_per_method = {}
  for t in _g_time:
    print(t)
    val,method = t.split('(')[0].lstrip(' ').split(' ')[0:2]
    #print(val,method)
    val = float(val.rstrip('s'))
    if method not in sum_per_method:
      sum_per_method.update({method: val})
    else:
      sum_per_method[method] += val
  #print(sum_per_method)
  sum_per_method = sorted(sum_per_method.items(), key=operator.itemgetter(1),  reverse=True)
  #print(sum_per_method)
  print("Timing sorted per function:")
  for tup in sum_per_method:
    print(' {:.2f}s {}'.format(tup[1], tup[0]))
    #print(' {:.2f}s'.format(tup[1]))

g_skip_atexit = False
_g_time = []
_do_time = 1

class Drop:
    def __init__(self):
      self.box_height = 16
      self.basedir = '/home/zenaiev/games/Diablo2/ssr/'
      #self.config_tesseract = "-c tessedit_char_whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -'"
      self.config_tesseract = "-c tessedit_char_whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZ -'" + "-c user_words_file=/home/zenaiev/games/Diablo2/ssr/d2-drop-words-upper.txt"
      with open(self.basedir + 'd2-drop-items.txt') as f:
        self.items = [l[:-1] for l in f.readlines()]
      with open(self.basedir + 'd2-drop-words-for-dict.txt') as f:
        # same order as self.items
        self.items_single = [l[:-1] for l in f.readlines()]
      self.current_drop = []
      self.flag_train = False
      self.flag_mycheck = False
      self.flag_textrec = True
      self.flag_selectedbox = False
      self.flag_allboxes = True
      self.flag_skipwp = False
      self.matched = 0
      self.notmatched = 0
      self.sig = []
      self.bkg = []
      self.sig_x1 = []
      self.bkg_x1 = []
      self.cuts = []
      self.cuts_x1 = []
      self.vars_title = []
      self.vars_title_x1 = []
      #self.cuts = {}
      self._config_spellchecker()

    def _config_spellchecker(self):
      #self.spellchecker = aspell.Speller('lang', 'en')
      self.spellchecker = aspell.Speller(('master', self.basedir + 'd2-words.rws'), ('home-dir', self.basedir))
      return
      #self.spellchecker = aspell.Speller()
      #ll = self.spellchecker.getMainwordlist()
      #print(len(ll))
      #a
      #with open(self.basedir + 'd2-drop-words.txt') as f:
      #  for l in f.readlines():
      #    w = l.rstrip()
      #    print('adding {}'.format(w))
      #    self.spellchecker.addtoSession(w)

    def _suggest_item(self, item):
      # https://github.com/WojciechMula/aspell-python
      if item.endswith(' PS TION'):
        item = item[:-8] + ' POTION'
      item = item.lower().replace(' ', 'q').replace('-', 'z')
      item_new_w = []
      # in principle item should be one word now
      for w in item.split():
        if w in self.spellchecker:
          print('{} in spellchecker'.format(w))
          item_new_w.append(w)
        else:
          w_sug = self.spellchecker.suggest(w)
          print('{} not in spellchecker, suggested {}'.format(w, w_sug))
          if len(w_sug) > 0:
            item_new_w.append(w_sug[0])
      item_new = ' '.join(item_new_w)
      if item_new in self.items_single:
        item_new = self.items[self.items_single.index(item_new)]
      print('  --> {}'.format(item_new))
      #cv2.waitKey()
      return item_new

    def stats(self):
      print('MATCHED/NOT {}/{}'.format(self.matched, self.notmatched))
      if self.flag_train:
        cuts_xy, cuts_x1 = [], []
        print('{:20s}{:>18s}  {:>18s}  {:>5s} [{:>6s}]{:>8s}{:>8s}'.format('Variable', 'sig', 'bkg', 'cut', 'q', 'sig_egg', 'bkg_eff'))
        nchannels = 4
        for sig,bkg,title,cuts in zip([self.sig, self.sig_x1], [self.bkg, self.bkg_x1], [self.vars_title, self.vars_title_x1], [cuts_xy, cuts_x1]):
          icomb_last = 0
          for i in range(len(sig[0])):
            delta_i = i-icomb_last
            if len(title[i].split('[')) > 1:
              if delta_i%nchannels == 0:
                fig, axes = plt.subplots(nchannels, 1, sharex=True, squeeze=True)
                fig.suptitle(title[i].split('[')[0])
                fig.canvas.set_window_title(title[i].split('[')[0])
                plt.subplots_adjust(hspace = 0.0, left = 0.01, right = 0.99)
                bins = None
              ax = axes[delta_i%nchannels]
              if delta_i%nchannels == nchannels-1:
                store = title[i].split('[')[0]
                icomb_last = i+1
              else:
                store = None
            else:
              bins = None
              fig = plt.figure(title[i])
              ax = None
              store = title[i]
              icomb_last = i+1
            bkg_sample = random.sample(bkg, min(10000, len(bkg)))
            cut, bins = self._plot(title[i], [v[i] for v in sig], [v[i] for v in bkg_sample], bins, [1.0, 0.90], ax, col=['b','g','r', 'magenta'][delta_i%nchannels], store=store)
            cuts.append(cut)
        if len(self.cuts) == len(cuts_xy):
          if any(a<b for a,b in zip(cuts_xy, self.cuts)):
            print('cuts_xy to be updated: {}'.format([a<b for a,b in zip(cuts_xy, self.cuts)]))
            cuts_xy = [min(a,b) for a,b in zip(cuts_xy, self.cuts)]
            print('dropper.cuts = {}'.format([math.floor(c*1000)/1000.0 for c in cuts_xy]))
          else:
            print('cuts_xy are the same')
        else:
          print('cuts_xy are new')
          print('dropper.cuts = {}'.format([math.floor(c*1000)/1000.0 for c in cuts_xy]))
        if len(self.cuts_x1) == len(cuts_x1):
          if any(a<b for a,b in zip(cuts_x1, self.cuts_x1)):
            print('cuts_x1 to be updated: {}'.format([a<b for a,b in zip(cuts_x1, self.cuts_x1)]))
            cuts_x1 = [min(a,b) for a,b in zip(cuts_x1, self.cuts_x1)]
            print('dropper.cuts_x1 = {}'.format([math.floor(c*1000)/1000.0 for c in cuts_x1]))
          else:
            print('cuts_x1 are the same')
        else:
          print('cuts_x1 are new')
          print('dropper.cuts_x1 = {}'.format([math.floor(c*1000)/1000.0 for c in cuts_x1]))
        plt.show(block=False)
        cv2.waitKey()
        #print(self.cuts)
        #assert all(abs(v1-v2)<1e-5 for v1,v2 in zip(self.cuts, cuts_xy))
        #print(self.cuts_x1)
        #assert all(abs(v1-v2)<1e-5 for v1,v2 in zip(self.cuts_x1, cuts_x1))

    def _plot(self, label, sig, bkg, bins=None, qs=[0.90], ax=None, col=None, store=False):
      #print(len(sig), len(bkg))
      sig_mean,sig_min,sig_max = sum(sig)/len(sig), min(sig), max(sig)
      sig_str = '{:5.0f} [{:5.0f}{:5.0f}]'.format(sig_mean,sig_min,sig_max)
      bkg_mean,bkg_min,bkg_max = sum(bkg)/len(bkg), min(bkg), max(bkg)
      bkg_str = '{:5.0f} [{:5.0f}{:5.0f}]'.format(bkg_mean,bkg_min,bkg_max)
      print('{:20s}{}  {}'.format(label, sig_str, bkg_str), end='')
      q_str = []
      for q in qs:
        c, sig_eff, bkg_eff= self._calc_sig_bkg_eff(sig, bkg, q)
        q_str.append('  {:5.0f} [{:6.1f}]{:8.4f}{:8.4f}'.format(c, q, sig_eff, bkg_eff))
        print(q_str[-1], end='')
      print()
      if ax is None:
        #ax = plt
        ax = plt.gca()
      if bins is None:
        bins = np.linspace(min(sig_min, bkg_min), max(sig_max, bkg_max), 100)
      ax.hist(bkg, bins, alpha=0.5, density=True, label='bkg({}) {:.0f}[{:.0f},{:.0f}]'.format(len(bkg), bkg_mean,bkg_min,bkg_max), color='k')
      ax.hist(sig, bins, alpha=0.5, density=True, label='sig({}) {:.0f}[{:.0f},{:.0f}]'.format(len(sig), sig_mean,sig_min,sig_max), color=col)
      ax.legend(loc=2)
      ax.text(0.0,0.35,'\n'.join(q_str),horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
      ax.axes.get_yaxis().set_visible(False)
      if store is not None:
        if not os.path.isdir('plots'):
          os.mkdir('plots')
        plt.savefig('plots/{}.pdf'.format(store))
        plt.savefig('plots/{}.png'.format(store))
      return sig_min, bins

    def _calc_sig_bkg_eff(self, sig, bkg, q):
      if q == 1.0:
        c = min(sig)
      else:
        c = np.quantile(sig, 1-q)
      sig_eff = sum(1 for v in sig if v >= c) / len(sig)
      bkg_eff = sum(1 for v in bkg if v >= c) / len(bkg)
      #print('sig_eff {} bkg_eff {} cut {}'.format(sig_eff, bkg_eff, c))
      return (c, sig_eff, bkg_eff)

    def reset_drop(self):
        self.current_drop = []
    
    @timeit
    def process_drop(self, img, signal=[]):
      #self._waypoint(self.img_orig, 'orig')
      #self._waypoint(img, 'input')
      if self.flag_selectedbox:
        ys, ls, rs, mask = self._selected_box(img, signal)
      elif self.flag_allboxes:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_gray = np.int16(img_gray)
        img_gray = np.float32(img_gray)
        img_float = np.float32(img)
        ys, ls, rs = self._boxes(img_float, signal=signal)
        mask = None
      ys = [y+1 for y in ys]
      ls = [y+1 for y in ls]
      rs = [x for x in rs]
      print(ys, ls, rs, sep='\n')
      #ys = [249, 249, 249, 264, 264, 279, 294, 309, 324, 339, 357, 376]
      #ls = [328, 421, 578, 268, 409, 307, 312, 289, 326, 244, 386, 296]
      #rs = [419, 576, 783, 407, 574, 476, 400, 458, 491, 434, 432, 451]
      #ys,ls,rs = [284], [303], [433]
      drop = []
      if not self.flag_train and self.flag_textrec:
        i = -1
        for y,l,r in zip(ys, ls, rs):
          #if i != 10: continue
          i += 1
          #if i != 6: continue
          #for scale in range(1, 30):
          #  print('scale = {}'.format(scale))
          #  drop.append(self._process_box(img_gray[y:y+self.box_height, l:r+1], self.img_orig[y:y+self.box_height, l:r+1], i, scale))
          #  print('box {:2d}: y = {} x = {} x1 = {} color = {} conf = {} item = {}'.format(i, y, l, r, drop[-1][2], drop[-1][1], drop[-1][0]))
          drop.append(self._process_box(self.img_orig[y:y+self.box_height, l:r+1], i, mask=mask))
          #if len(drop[-1][0]) > 0:
          drop[-1][0] = self._suggest_item(drop[-1][0])
          drop[-1] += [y, l, r]
          print('box {:2d}: y = {} x = {} x1 = {} color = {} conf = {} item = {}'.format(i, y, l, r, drop[-1][2], drop[-1][1], drop[-1][0]))
          #cv2.waitKey()
          #if i == 1: break
        print('drop {}: {}'.format(len(drop), drop))
        self._print_coloured(drop)
        if len(signal) > 0:
          #print(len(drop), len(signal))
          match = len(drop) == len(signal)
          if match:
            for i in range(len(signal)):
              print('i = {}'.format(i))
              print(signal[i])
              print(drop[i])
              if signal[i][0] != drop[i][3] or signal[i][1] != drop[i][4] or signal[i][2] != drop[i][5] or signal[i][4] != drop[i][2] or signal[i][3] != drop[i][0]:
                match = False
                break
          print('MATCH: {}'.format(match))
          if match:
            self.matched += 1
          else:
            self.notmatched += 1
      for y,l,r in zip(ys, ls, rs):
        cv2.rectangle(img, (l-1, y-1), (r+1, y+self.box_height), (0, 255, 0), 1)
      self._waypoint(img, 'output', 1)
      #if len(drop) > 0:
      #  cv2.waitKey()
      return drop

    def _selected_box(self, img_bgr, signal=[]):
      box_x = 30
      box_y = 16
      hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
      '''fig, ax = plt.subplots(3)
      y,x,x1 = signal[0:3]
      self._waypoint(self.img_orig[y:y+box_y,x:x1], 'tmp', 0)
      print(hsv[y+5:y+6,x:x1])
      hist_hue,_,_ = ax[0].hist(hsv[y:y+box_y,x:x1,0].ravel(), np.linspace(0.0, 180., 180), alpha=0.5)
      hist_sat,_,_ = ax[1].hist(hsv[y:y+box_y,x:x1,1].ravel(), np.linspace(0.0, 255., 255), alpha=0.5)
      hist_val,_,_ = ax[2].hist(hsv[y:y+box_y,x:x1,2].ravel(), np.linspace(0.0, 255., 255), alpha=0.5)
      #print(hist_hue)
      print('hist_hue !=0: {}'.format(np.where(hist_hue != 0)))
      print('hist_sat !=0: {}'.format(np.where(hist_sat != 0)))
      print('hist_val !=0: {}'.format(np.where(hist_val != 0)))
      plt.show()
      a'''
      hsv_filt = cv2.inRange(hsv, (108,190,30), (115,255,120))
      #self._waypoint(hsv_filt, 'filt', 1)
      #print(np.nonzero(hsv_filt))
      pos_x,pos_y = np.nonzero(hsv_filt)
      x1 = None
      for y,x in zip(pos_x,pos_y):
        #print(x,y,sum(hsv_filt[y,x+i] for i in range(box_x)), sum(hsv_filt[y+i,x] for i in range(box_y)))
        if sum(hsv_filt[y,x+i] for i in range(box_x)) > 255.*box_x/2. and sum(hsv_filt[y+i,x] for i in range(box_y)) > 255.*box_y/2.:
          #print(np.nonzero(hsv_filt[y]))
          x1 = max(np.nonzero(hsv_filt[y])[0])
          break
      if x1 is None:
        return [], [], [], None
      print('y,x,x1: {} {} {}'.format(y,x,x1))
      #hor = cv2.filter2D(hsv_filt, -1, np.matrix('1 '*box_x), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)
      #ver = cv2.filter2D(hsv_filt, -1, np.matrix('1 '*box_x), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)
      return [y-1], [x-1], [x1], cv2.bitwise_not(hsv_filt[y-1:y-1+box_y,x-1:x1])

    def _get_coloured_item(self, item, col):
      cols = {'u': '\033[91m', 'b': '\033[94m', 'o': '\033[95m', 'g': '\033[92m', 'y': '\033[93m', 'w': '\033[39m', 's': '\033[90m'}
      #cols = {'u': '\033[91m', 'b': '\033[1;31m', 'o': '\033[95m', 'g': '\033[92m', 'y': '\033[93m', 'w': '\033[39m'}
      #class bcolors:
      #  LU = '\033[91m'
      #  LB = '\033[94m'
      #  LO = '\033[95m'
      #  LG = '\033[92m'
      #  LY = '\033[93m'
      #  ENDC = '\033[0m'
      return cols[col] + item + '\033[0m'

    def _print_coloured(self, drop):
      # {'u': 23.5, 'b': 119, 'o': 20, 'g': 120, 'y': 29}
      cols = {'u': '\033[91m', 'b': '\033[94m', 'o': '\033[95m', 'g': '\033[92m', 'y': '\033[93m', 'w': '\033[39m'}
      #class bcolors:
      #  LU = '\033[91m'
      #  LB = '\033[94m'
      #  LO = '\033[95m'
      #  LG = '\033[92m'
      #  LY = '\033[93m'
      #  ENDC = '\033[0m'
      print('_'*30)
      for d in drop:
        print(self._get_coloured_item(d[0], d[2]))
      print(' \u0305'*30)

    def _set_to_zero(self, img, y0=None, y1=None, x0=None, x1=None):
      if y0 is None: y0 = 0
      if y1 is None: y1 = img.shape[0]
      if x0 is None: x0 = 0
      if x1 is None: x1 = img.shape[1]
      assert y0 <= y1
      assert x0 <= x1
      assert y1 <= img.shape[0]
      assert x1 <= img.shape[1]
      img[y0:y1, x0:x1] = np.zeros(img[y0:y1, x0:x1].shape)

    def _max_img_from_list(self, imgs):
      img_max = imgs[0].copy()
      for i in range(1, len(imgs)):
        img_max = cv2.max(img_max, imgs[i])
      return img_max

    def _calc_rat(self, img, nom, denom):
      denom[np.where(denom == 0)] = 1
      rat = nom/denom*100
      rat[np.where(np.isinf(img))] = -255
      return rat

    @timeit
    def _boxes(self, img_bgr, signal=[]):
      img = cv2.merge((*cv2.split(img_bgr), np.sum(img_bgr, axis=2)/3))
      channels = 'bgrv'
      #print(img[0:2,0:2])
      assert img.shape[2] == 4 # bgr+val
      box_x = 31
      n_box_x = 7
      if self.flag_mycheck:
        n_box_x = 1
      box_y = self.box_height
      #box_y = 6
      mar_l, mar_r, mar_t, mar_b = 4, 5, 2, 5
      #mar_l, mar_r, mar_t, mar_b = 2, 2, 2, 2
      min_sumrat = -500
      mean_bgr = np.mean(img, axis=(0,1))
      mean_img = np.mean(mean_bgr)
      #print('mean: {} {}'.format(mean_bgr, mean_img))
      t = cv2.filter2D(img, -1, np.matrix('1; -2'), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)
      b = cv2.filter2D(img, -1, np.matrix('-2; 1'), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)
      l = cv2.filter2D(img, -1, np.matrix('1 -2'), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)
      r = cv2.filter2D(img, -1, np.matrix('-2 1'), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)
      #diff_min = 10
      #r_3 = cv2.inRange(t, (-diff_min,-diff_min,-diff_min, -np.inf), (diff_min,diff_min,diff_min,+np.inf))
      #self._waypoint(r_3, 'r3', 1)
      _,r0 = cv2.threshold(np.sum(np.absolute(r), axis=2),20,1,cv2.THRESH_BINARY_INV)
      rnl = 2*cv2.filter2D(img, -1, np.matrix('-1 1 -1'), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)
      space = -1*cv2.filter2D(img, -1, np.matrix(';'.join(['1 '*10]*box_y)), anchor=(4,0), borderType=cv2.BORDER_ISOLATED)/10/box_y
      ta = np.absolute(t)*-1
      ba = np.absolute(b)*-1
      la = np.absolute(l)*-1
      ra = np.absolute(r)*-1
      rnla = np.absolute(rnl)*-1
      tsum = [cv2.filter2D(t, -1, np.matrix('1 '*box_x*i), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_x*i) for i in range(1,n_box_x+1)]
      tsum_max = self._max_img_from_list(tsum)
      tsumrat_max = self._max_img_from_list([self._calc_rat(img, tsum[i-1], cv2.filter2D(img, -1, np.matrix('1 '*box_x*i), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_x*i)) for i in range(1,n_box_x+1)])
      tsumrat_max = np.clip(tsumrat_max, min_sumrat, np.inf)
      tasum = [cv2.filter2D(ta, -1, np.matrix('1 '*box_x*i), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_x*i) for i in range(1,n_box_x+1)]
      tasum_max = self._max_img_from_list(tasum)
      tasumrat_max = self._max_img_from_list([self._calc_rat(img, tasum[i-1], cv2.filter2D(img, -1, np.matrix('1 '*box_x*i), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_x*i)) for i in range(1,n_box_x+1)])
      tasumrat_max = np.clip(tasumrat_max, min_sumrat, np.inf)
      bsum = [cv2.filter2D(b, -1, np.matrix('1 '*box_x*i), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_x*i) for i in range(1,n_box_x+1)]
      bsum_max = self._max_img_from_list(bsum)
      bsumrat_max = self._max_img_from_list([self._calc_rat(img, bsum[i-1][:-1], cv2.filter2D(img[1:, :], -1, np.matrix('1 '*box_x*i), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_x*i)) for i in range(1,n_box_x+1)])
      bsumrat_max = np.clip(bsumrat_max, min_sumrat, np.inf)
      basum = [cv2.filter2D(ba, -1, np.matrix('1 '*box_x*i), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_x*i) for i in range(1,n_box_x+1)]
      basum_max = self._max_img_from_list(basum)
      basumrat_max = self._max_img_from_list([self._calc_rat(img, basum[i-1][:-1], cv2.filter2D(img[1:, :], -1, np.matrix('1 '*box_x*i), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_x*i)) for i in range(1,n_box_x+1)])
      basumrat_max = np.clip(basumrat_max, min_sumrat, np.inf)
      lsum = cv2.filter2D(l, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_y)
      lbmsum = cv2.filter2D(l, -1, np.matrix(';'.join('1'*mar_b)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(mar_b)
      lsumrat = self._calc_rat(img, lsum[:, :-1], cv2.filter2D(img[:, :-1], -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/box_y)
      lsumrat = np.clip(lsumrat, min_sumrat, np.inf)
      lbmsumrat = self._calc_rat(img, lbmsum[:, :-1], cv2.filter2D(img[:, :-1], -1, np.matrix(';'.join('1'*mar_b)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/mar_b)
      lbmsumrat = np.clip(lbmsumrat, min_sumrat, np.inf)
      lasum = cv2.filter2D(la, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_y)
      lbmasum = cv2.filter2D(la, -1, np.matrix(';'.join('1'*mar_b)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(mar_b)
      lasumrat = self._calc_rat(img, lasum[:, :-1], cv2.filter2D(img[:, :-1], -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/box_y)
      lasumrat = np.clip(lasumrat, min_sumrat, np.inf)
      lbmasumrat = self._calc_rat(img, lbmasum[:, :-1], cv2.filter2D(img[:, :-1], -1, np.matrix(';'.join('1'*mar_b)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/mar_b)
      lbmasumrat = np.clip(lbmasumrat, min_sumrat, np.inf)
      rsum = cv2.filter2D(r, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_y)
      rsumrat = self._calc_rat(img, rsum[:, :-1], cv2.filter2D(img[:, 1:], -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/box_y)
      rsumrat = np.clip(rsumrat, min_sumrat, np.inf)
      rbmsum = cv2.filter2D(r, -1, np.matrix(';'.join('1'*mar_b)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(mar_b)
      rbmsumrat = self._calc_rat(img, rbmsum[:, :-1], cv2.filter2D(img[:, 1:], -1, np.matrix(';'.join('1'*mar_b)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/mar_b)
      rbmsumrat = np.clip(rbmsumrat, min_sumrat, np.inf)
      r0sum = cv2.filter2D(r0, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)
      rnsum = cv2.filter2D(img, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_y)
      rasum = cv2.filter2D(ra, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_y)
      rasumrat = self._calc_rat(img, rasum[:, :-1], cv2.filter2D(img[:, 1:], -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/box_y)
      rasumrat = np.clip(rasumrat, min_sumrat, np.inf)
      rbmasum = cv2.filter2D(ra, -1, np.matrix(';'.join('1'*mar_b)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(mar_b)
      rbmasumrat = self._calc_rat(img, rbmasum[:, :-1], cv2.filter2D(img[:, 1:], -1, np.matrix(';'.join('1'*mar_b)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/mar_b)
      rbmasumrat = np.clip(rbmasumrat, min_sumrat, np.inf)
      rnlsum = cv2.filter2D(rnl, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_y)
      rnlasum = cv2.filter2D(rnla, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(box_y)
      dxa = np.absolute(cv2.filter2D(img, -1, np.matrix('1 -1'), anchor=(0,0), borderType=cv2.BORDER_ISOLATED))
      dya = np.absolute(cv2.filter2D(img, -1, np.matrix('1; -1'), anchor=(0,0), borderType=cv2.BORDER_ISOLATED))
      dxya = dxa + dya
      assert box_y > (mar_b+mar_t+1)
      assert box_x > (mar_l+mar_r+1)
      boxsum = [cv2.filter2D(dxya, -1, np.matrix(';'.join(['1 '*(box_x-mar_l-1)*i]*(box_y-mar_t-mar_b-1))), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(2*(box_x-mar_l-1)*i*(box_y-mar_b-mar_t-1)) for i in range(1,n_box_x+1)]
      #maxrgb = lambda img: np.maximum(np.maximum(img[:, :, :1], img[:, :, 1:2]), img[:, :, 2:])
      #maxrgbv = lambda img: np.maximum(np.maximum(np.maximum(img[:, :, :1], img[:, :, 1:2]), img[:, :, 2:3]), img[:, :, 3:])
      maxrgbv = lambda img: np.amax(img, axis=2)
      boxsum = [maxrgbv(i) for i in boxsum]
      boxsum_max = self._max_img_from_list(boxsum)
      marlsum = maxrgbv(cv2.filter2D(dxya, -1, np.matrix(';'.join(['1 '*(mar_l-1)]*(box_y-1))), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(2*(mar_l-1)*(box_y-1)))
      marrsum = cv2.filter2D(dxya, -1, np.matrix(';'.join(['1 '*(mar_r-1)]*(box_y-1))), anchor=(0,0), borderType=cv2.BORDER_ISOLATED)/(2*(mar_r-1)*(box_y-1))
      num_rows, num_cols = marrsum.shape[:2]
      translation_matrix = np.float32([ [1,0,mar_r-1], [0,1,0] ])
      marrsum = cv2.warpAffine(marrsum, translation_matrix, (num_cols,num_rows))
      marrsum = maxrgbv(marrsum)
      if len(self.vars_title) == 0:
        self.vars_title += ['tsum_max'+'['+c+']' for c in channels]
        self.vars_title += ['tasum_max'+'['+c+']' for c in channels]
        self.vars_title += ['tsumrat_max'+'['+c+']' for c in channels]
        self.vars_title += ['tasumrat_max'+'['+c+']' for c in channels]
        self.vars_title += ['bsum_max'+'['+c+']' for c in channels]
        self.vars_title += ['basum_max'+'['+c+']' for c in channels]
        self.vars_title += ['bsumrat_max'+'['+c+']' for c in channels]
        self.vars_title += ['basumrat_max'+'['+c+']' for c in channels]
        self.vars_title += ['lsum'+'['+c+']' for c in channels]
        self.vars_title += ['lasum'+'['+c+']' for c in channels]
        self.vars_title += ['lsumrat'+'['+c+']' for c in channels]
        self.vars_title += ['lasumrat'+'['+c+']' for c in channels]
        self.vars_title += ['lbmsum'+'['+c+']' for c in channels]
        self.vars_title += ['lbmasum'+'['+c+']' for c in channels]
        self.vars_title += ['lbmsumrat'+'['+c+']' for c in channels]
        self.vars_title += ['lbmasumrat'+'['+c+']' for c in channels]
        self.vars_title += ['boxsum_max']
        self.vars_title += ['marlsum']
      #print(tsum_max[:-box_y-1, 1:-box_x, 0].shape, tsumrat_max[:-box_y-1, 1:-box_x, 0].shape, bsum_max[box_y:-1, 1:-box_x, 0].shape, bsumrat_max[box_y:-1, 1:-box_x, 0].shape)
      img_vars = cv2.merge(
        () +
        tuple(tsum_max[:-box_y-1, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(tasum_max[:-box_y-1, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(tsumrat_max[:-box_y-1, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(tasumrat_max[:-box_y-1, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(bsum_max[box_y:-1, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(basum_max[box_y:-1, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(bsumrat_max[box_y:, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(basumrat_max[box_y:, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(lsum[1:-box_y, :-box_x-1, c] for c in range(img.shape[2])) +
        tuple(lasum[1:-box_y, :-box_x-1, c] for c in range(img.shape[2])) +
        tuple(lsumrat[1:-box_y, :-box_x, c] for c in range(img.shape[2])) +
        tuple(lasumrat[1:-box_y, :-box_x, c] for c in range(img.shape[2])) +
        tuple(lbmsum[1+box_y-mar_b:-mar_b, :-box_x-1, c] for c in range(img.shape[2])) +
        tuple(lbmasum[1+box_y-mar_b:-mar_b, :-box_x-1, c] for c in range(img.shape[2])) +
        tuple(lbmsumrat[1+box_y-mar_b:-mar_b, :-box_x, c] for c in range(img.shape[2])) +
        tuple(lbmasumrat[1+box_y-mar_b:-mar_b, :-box_x, c] for c in range(img.shape[2])) +
        tuple((boxsum_max[mar_t+1:-box_y+mar_t, mar_l+1:-box_x+mar_l],)) +
        tuple((marlsum[1:-box_y, 1:-box_x]*-1,)) +
      ())
      #print(img[0:3, 0:3])
      if len(self.vars_title_x1) == 0:
        self.vars_title_x1 += ['rsum'+'['+c+']' for c in channels]
        self.vars_title_x1 += ['rasum'+'['+c+']' for c in channels]
        self.vars_title_x1 += ['rsumrat'+'['+c+']' for c in channels]
        self.vars_title_x1 += ['rasumrat'+'['+c+']' for c in channels]
        self.vars_title_x1 += ['rbmsum'+'['+c+']' for c in channels]
        self.vars_title_x1 += ['rbmasum'+'['+c+']' for c in channels]
        self.vars_title_x1 += ['rbmsumrat'+'['+c+']' for c in channels]
        self.vars_title_x1 += ['rbmasumrat'+'['+c+']' for c in channels]
        self.vars_title_x1 += ['marrsum']
        #
        #self.vars_title_x1 += ['r0sum']
        #self.vars_title_x1 += ['r0sum'+'['+c+']' for c in 'bgr']
        #self.vars_title_x1 += ['rnsum'+'['+c+']' for c in 'bgr']
        #self.vars_title_x1 += ['rnlsum'+'['+c+']' for c in 'bgr']
        #self.vars_title_x1 += ['ntsum'+'['+c+']' for c in 'bgr']
        #self.vars_title_x1 += ['rnlmntsum'+'['+c+']' for c in 'bgr']
        #self.vars_title_x1 += ['rnlmntasum'+'['+c+']' for c in 'bgr']
        #self.vars_title_x1 += ['space'+'['+c+']' for c in 'bgr']
      img_vars_x1 = cv2.merge(
        () +
        tuple(rsum[1:-box_y, :-1, c] for c in range(img.shape[2])) +
        tuple(rasum[1:-box_y, :-1, c] for c in range(img.shape[2])) +
        tuple(rsumrat[1:-box_y, :, c] for c in range(img.shape[2])) +
        tuple(rasumrat[1:-box_y, :, c] for c in range(img.shape[2])) +
        tuple(rbmsum[1+box_y-mar_b:-mar_b, :-1, c] for c in range(img.shape[2])) +
        tuple(rbmasum[1+box_y-mar_b:-mar_b, :-1, c] for c in range(img.shape[2])) +
        tuple(rbmsumrat[1+box_y-mar_b:-mar_b, :, c] for c in range(img.shape[2])) +
        tuple(rbmasumrat[1+box_y-mar_b:-mar_b, :, c] for c in range(img.shape[2])) +
        tuple((marrsum[1:-box_y, 0:-1]*-1,)) +
        #
        #tuple((r0sum[1:-box_y, :-1],)) +
        #tuple(r0sum[1:-box_y, :-1, c] for c in range(img.shape[2])) +
        #tuple(rnsum[1:-box_y, 1:, c] for c in range(img.shape[2])) +
        #tuple(rnlsum[1:-box_y, :-1, c] for c in range(img.shape[2])) +
        #tuple(tsum_max[:-box_y-1, 1:, c] for c in range(img.shape[2])) +
        #tuple(rnlsum[1:-box_y, :-1, c]-tsum_max[:-box_y-1, 1:, c] for c in range(img.shape[2])) +
        #tuple(rnlasum[1:-box_y, :-1, c]-tasum_max[:-box_y-1, 1:, c] for c in range(img.shape[2])) +
        #tuple(space[1:-box_y, 0:-1, c] for c in range(img.shape[2])) +
      ())
      assert img_vars.shape[0]+box_y+1 == img.shape[0] and img_vars.shape[1]+box_x+1 == img.shape[1]
      #assert img_vars.shape[2] == 18
      ys, xs, x1s = [], [], []
      sig = np.zeros(img_vars.shape[0:2])
      #sig = np.zeros(img.shape[0:2])
      if self.flag_train:
        print_str = '{:4.0f}{:4.0f}{:4.0f}  {:25s}{:3s}' + '{:4.0f}'*len(self.vars_title) + '{:4.0f}'*len(self.vars_title_x1)
        print('Variables: {}'.format({iv: v for iv,v in enumerate(self.vars_title + self.vars_title_x1)}))
        print(print_str.replace('.0f}', 's}').replace('{:', '{:>').format('y','x0','x1','Drop','c',*('v'+str(i) for i in range(len(self.vars_title) + len(self.vars_title_x1)))))
        for s in signal:
          #print(s)
          assert s[0] > 0 and s[1] > 0
          y, x, x1 = s[0]-1, s[1]-1, s[2]
          if self.flag_mycheck:
            my_tsum = sum(img[y, xx]-2*img[y+1, xx] for xx in range(x+1, x+1+box_x))/box_x
            my_tasum = -1*sum(abs(img[y, xx]-2*img[y+1, xx]) for xx in range(x+1, x+1+box_x))/box_x
            tsum_denom = sum(img[y, xx] for xx in range(x+1, x+1+box_x))
            my_tsumrat = [100*sum(img[y, xx, c]-2*img[y+1, xx, c] for xx in range(x+1, x+1+box_x))/tsum_denom[c] if tsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
            my_tsumrat = np.clip(my_tsumrat, min_sumrat, np.inf)
            my_tasumrat = [100*-1*sum(abs(img[y, xx, c]-2*img[y+1, xx, c]) for xx in range(x+1, x+1+box_x))/tsum_denom[c] if tsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
            my_tasumrat = np.clip(my_tasumrat, min_sumrat, np.inf)
            my_bsum = sum(-2*img[y+box_y, xx]+img[y+box_y+1, xx] for xx in range(x+1, x+1+box_x))/box_x
            my_basum = -1*sum(abs(2*img[y+box_y, xx]-img[y+box_y+1, xx]) for xx in range(x+1, x+1+box_x))/box_x
            bsum_denom = sum(img[y+box_y+1, xx] for xx in range(x+1, x+1+box_x))
            my_bsumrat = [100*sum(-2*img[y+box_y, xx, c]+img[y+1+box_y, xx, c] for xx in range(x+1, x+1+box_x))/bsum_denom[c] if bsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
            my_bsumrat = np.clip(my_bsumrat, min_sumrat, np.inf)
            my_basumrat = [100*-1*sum(abs(-2*img[y+box_y, xx, c]+img[y+1+box_y, xx, c]) for xx in range(x+1, x+1+box_x))/bsum_denom[c] if bsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
            my_basumrat = np.clip(my_basumrat, min_sumrat, np.inf)
            my_lsum = sum(img[yy, x]-2*img[yy, x+1] for yy in range(y+1, y+1+box_y))/box_y
            my_lbmsum = sum(img[yy, x]-2*img[yy, x+1] for yy in range(y+1+box_y-mar_b, y+1+box_y))/mar_b
            my_lasum = -1*sum(abs(img[yy, x]-2*img[yy, x+1]) for yy in range(y+1, y+1+box_y))/box_y
            my_lbmasum = -1*sum(abs(img[yy, x]-2*img[yy, x+1]) for yy in range(y+1+box_y-mar_b, y+1+box_y))/mar_b
            lsum_denom = sum(img[yy,x] for yy in range(y+1, y+1+box_y))
            lbmsum_denom = sum(img[yy,x] for yy in range(y+1+box_y-mar_b, y+1+box_y))
            my_lsumrat = [100*sum(img[yy, x, c]-2*img[yy, x+1, c] for yy in range(y+1, y+1+box_y))/lsum_denom[c] if lsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
            my_lsumrat = np.clip(my_lsumrat, min_sumrat, np.inf)
            my_lbmsumrat = [100*sum(img[yy, x, c]-2*img[yy, x+1, c] for yy in range(y+1+box_y-mar_b, y+1+box_y))/lbmsum_denom[c] if lbmsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
            my_lbmsumrat = np.clip(my_lbmsumrat, min_sumrat, np.inf)
            my_lasumrat = [100*-1*sum(abs(img[yy, x, c]-2*img[yy, x+1, c]) for yy in range(y+1, y+1+box_y))/lsum_denom[c] if lsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
            my_lasumrat = np.clip(my_lasumrat, min_sumrat, np.inf)
            my_lbmasumrat = [100*-1*sum(abs(img[yy, x, c]-2*img[yy, x+1, c]) for yy in range(y+1+box_y-mar_b, y+1+box_y))/lbmsum_denom[c] if lbmsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
            my_lbmasumrat = np.clip(my_lbmasumrat, min_sumrat, np.inf)
            my_boxsum = max(self._calc_adxdysum(img, x+1+mar_l, x+box_x, y+1+mar_t, y+box_y-mar_b))
            my_marlsum = max(self._calc_adxdysum(img, x+1, x+mar_l, y+1, y+box_y))*-1
            my_vars = sum((list(img) for img in (my_tsum, my_tasum, my_tsumrat, my_tasumrat, my_bsum, my_basum, my_bsumrat, my_basumrat, my_lsum, my_lasum, my_lsumrat, my_lasumrat, my_lbmsum, my_lbmasum, my_lbmsumrat, my_lbmasumrat, [my_boxsum], [my_marlsum])), [])
            my_diff = [a-b for a,b in zip(my_vars, img_vars[y, x])]
            if not all(abs(d) < 3e-5 for d in my_diff):
              print('img {}'.format(img_vars[y, x]))
              print('my {}'.format(my_vars))
              print('diff (len diff {}-{}={}): {}'.format(len(my_vars), len(img_vars[y, x]), len(my_vars)-len(img_vars[y, x]), my_diff))
              print('diffs are: {}'.format([(a,b) for t,a,b in zip([abs(d) < 1e-5 for d in my_diff], my_vars, img_vars[y, x]) if not t]))
              assert 0
          print(print_str.format(*s, *img_vars[y, x], *img_vars_x1[y, x1]))
          self.sig.append(img_vars[y, x])
          sig[y, x]
          ys.append(y)
          xs.append(x)
          x1s.append(x1)
          for x1_scan in range(x+box_x, img_vars_x1.shape[1]):
            if x1_scan == x1:
              if self.flag_mycheck:
                my_rsum = sum(-2*img[yy, x1]+img[yy, x1+1] for yy in range(y+1, y+1+box_y))/box_y
                rsum_denom = sum(img[yy,x1+1] for yy in range(y+1, y+1+box_y))
                my_rbmsum = sum(-2*img[yy, x1]+img[yy, x1+1] for yy in range(y+1+box_y-mar_b, y+1+box_y))/mar_b
                rbmsum_denom = sum(img[yy,x1+1] for yy in range(y+1+box_y-mar_b, y+1+box_y))
                my_rsumrat = [100*sum(-2*img[yy, x1, c]+img[yy, x1+1, c] for yy in range(y+1, y+1+box_y))/rsum_denom[c] if rsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
                my_rsumrat = np.clip(my_rsumrat, min_sumrat, np.inf)
                my_rbmsumrat = [100*sum(-2*img[yy, x1, c]+img[yy, x1+1, c] for yy in range(y+1+box_y-mar_b, y+1+box_y))/rbmsum_denom[c] if rbmsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
                my_rbmsumrat = np.clip(my_rbmsumrat, min_sumrat, np.inf)
                my_rasumrat = [100*-1*sum(abs(-2*img[yy, x1, c]+img[yy, x1+1, c]) for yy in range(y+1, y+1+box_y))/rsum_denom[c] if rsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
                my_rasumrat = np.clip(my_rasumrat, min_sumrat, np.inf)
                my_rbmasumrat = [100*-1*sum(abs(-2*img[yy, x1, c]+img[yy, x1+1, c]) for yy in range(y+1+box_y-mar_b, y+1+box_y))/rbmsum_denom[c] if rbmsum_denom[c] != 0 else -255 for c in range(img.shape[2])]
                my_rbmasumrat = np.clip(my_rbmasumrat, min_sumrat, np.inf)
                #my_r0sum = sum(sum(abs(-2*img[yy, x1, c]+img[yy, x1+1, c]) for c in range(3))<=20 for yy in range(y+1, y+1+box_y))
                my_r0sum = sum(sum(abs(-2*img[yy, x1, c]+img[yy, x1+1, c]) for c in range(3))<=20 for yy in range(y+1, y+1+box_y))
                my_rasum = -1*sum(abs(-2*img[yy, x1]+img[yy, x1+1]) for yy in range(y+1, y+1+box_y))/box_y
                my_rbmasum = -1*sum(abs(-2*img[yy, x1]+img[yy, x1+1]) for yy in range(y+1+box_y-mar_b, y+1+box_y))/mar_b
                my_marrsum = max(self._calc_adxdysum(img, x1-mar_r+1, x1, y+1, y+box_y))*-1
                my_rnsum = sum(img[yy, x1+1] for yy in range(y+1, y+1+box_y))/box_y
                my_rnlsum = 2*sum(-img[yy, x1]+img[yy, x1+1]-img[yy, x1+2] for yy in range(y+1, y+1+box_y))/box_y
                my_rnlasum = -1*sum(abs(-img[yy, x1]+img[yy, x1+1]-img[yy, x1+2]) for yy in range(y+1, y+1+box_y))/box_y
                my_ntsum = sum(img[y, xx]-2*img[y+1, xx] for xx in range(x1+1, min(x1+1+box_x, img.shape[1])))/box_x
                #if y == 278 and x == 306 and x1_scan == 476:
                #  l = [img[y, xx]-2*img[y+1, xx] for xx in range(x1+1, min(x1+1+box_x, img.shape[1]))]
                #  print(len(l), sum(l), sum(l)/box_x, l)
                #  a
                my_ntasum = -1*sum(abs(img[y, xx]-2*img[y+1, xx]) for xx in range(x1+1, min(x1+1+box_x, img.shape[1])))/box_x
                my_space = -1*sum(img[yy,xx] for xx in range(max(0,x1-4), min(x1+6, img.shape[1])) for yy in range(y+1, y+1+box_y))/10/box_y
                #my_vars = sum((list(img) for img in (my_rsum, my_rasum, [my_marrsum])), [])
                my_vars = sum((list(img) for img in (my_rsum, my_rasum, my_rsumrat, my_rasumrat, my_rbmsum, my_rbmasum, my_rbmsumrat, my_rbmasumrat, [my_marrsum])), [])
                #my_vars = sum((list(img) for img in (my_rsum, my_rasum, [my_marrsum], [my_r0sum])), [])
                #my_vars = sum((list(img) for img in (my_rsum, my_rasum, [my_marrsum], my_rnsum)), [])
                #my_vars = sum((list(img) for img in (my_rsum, my_rasum, [my_marrsum], my_rnlsum, my_ntsum, my_rnlsum-my_ntsum)), [])
                #my_vars = sum((list(img) for img in (my_rsum, my_rasum, [my_marrsum], my_rnlsum, my_ntsum)), [])
                #my_vars = sum((list(img) for img in (my_rsum, my_rasum, [my_marrsum], my_rnlsum-my_ntsum)), [])
                #my_vars = sum((list(img) for img in (my_rsum, my_rasum, [my_marrsum], my_rnlsum-my_ntsum, my_rnlasum-my_ntasum)), [])
                #my_vars = sum((list(img) for img in (my_rsum, my_rasum, [my_marrsum], my_rnlsum-my_ntsum)), [])
                #my_vars = sum((list(img) for img in (my_rsum, my_rasum, [my_marrsum], my_space)), [])
                my_diff = [a-b for a,b in zip(my_vars, img_vars_x1[y, x1])]
                #print(my_vars)
                #print(img_vars_x1[y, x1])
                if not all(abs(d) < 3e-5 for d in my_diff):
                  print('img {}'.format(img_vars_x1[y, x1]))
                  print('my {}'.format(my_vars))
                  print('diff (len diff {}-{}={}): {}'.format(len(my_vars), len(img_vars_x1[y, x1]), len(my_vars)-len(img_vars_x1[y, x1]), my_diff))
                  print('diffs are: {}'.format([a,b] for t,a,b in zip((abs(d) < 1e-5 for d in my_diff), my_vars, img_vars_x1[y, x]) if t))
                  assert 0
              self.sig_x1.append(img_vars_x1[y, x1_scan])
            else:
              self.bkg_x1.append(img_vars_x1[y, x1_scan])
        for x in range(sig.shape[1]):
          for y in range(sig.shape[0]):
            if sig[y, x] == 0:
              self.bkg.append(img_vars[y, x])
      else:
        for y in range(sig.shape[0]-box_y):
          x1_last = -1
          for x in range(sig.shape[1]-box_x):
            ret = self._check_box_at_xy(y, x, x1_last, box_x, sig, img_vars, img_vars_x1)
            if len(ret) > 0:
              #print(ret)
              for r in ret:
                ys.append(r[0])
                xs.append(r[1])
                x1s.append(r[2])
              x1_last = x1s[-1]
              #return (ys, xs, x1s)
      return (ys, xs, x1s)
    
    def _check_box_at_xy(self, y, x, x1_last, box_x, sig, img_vars, img_vars_x1):
      #print(y,x,x1_last,img_vars.shape[1],x+box_x, sig.shape[1]+box_x)
      #if y > 176: a
      ret = []
      if x < x1_last:
        return ret
      if x >= img_vars.shape[1]:
        return ret
      if all(v >= (c-1e-5) for v,c in zip(img_vars[y, x], self.cuts)):
        #print('checking x,y {} {}'.format(x, y))
        for x1 in range(x+box_x, sig.shape[1]+box_x):
          #print(x1)
          #if y == 176 and x == 278 and x1 == 389:
          #  print(img_vars[y, x])
          #  print(all(v >= (c-1e-5) for v,c in zip(img_vars[y, x], self.cuts)))
          #  print([v >= (c-1e-5) for v,c in zip(img_vars[y, x], self.cuts)])
          #  print(img_vars_x1[y, x1])
          #  print(all(v >= (c-1e-5) for v,c in zip(img_vars_x1[y, x1], self.cuts_x1)))
          #  print([v >= (c-1e-5) for v,c in zip(img_vars_x1[y, x1], self.cuts_x1)])
          #  a
          #if y == 248 and x == 577:
          #  print(' checking x1 {}: {}'.format(x1, img_vars_x1[y, x1]))
          if all(v >= (c-1e-5) for v,c in zip(img_vars_x1[y, x1], self.cuts_x1)):
            #print('  possible right edge x1 = {}'.format(x1))
            # check for top and bottom edges
            x_next = x1 + 1
            if x_next < img_vars.shape[1] and all(v >= (c-1e-5) for v,c in zip(img_vars[y, x_next], self.cuts[0:12])):
              #print('   possible top bottom edges continue')
              # check for next box after one pixel
              x1_last_next = x1
              #print('checking next x,y {} {}'.format(y, x_next))
              ret_next = self._check_box_at_xy(y, x_next, x1_last_next, box_x, sig, img_vars, img_vars_x1)
              #print('checking next x,y {} {}: {}'.format(y, x_next, ret_next))
              if len(ret_next) > 0:
                ret = ret_next
                #a
              else:
                # no box right edge, continue scan
                #b
                continue
                #pass
            #print(x1, img_vars_x1[y, x1])
            #x1s.append(x1)
            #print('signal found y,x = {} {}'.format(y,x))
            sig[y, x] = 1
            #ys.append(y)
            #xs.append(x)
            x1_last = x1
            #print(y, x, x1)
            #cv2.waitKey()
            ret.insert(0, [y, x, x1])
            return ret
      return ret

    def _calc_adxdysum(self, img, l, r, t, b):
      #print(l, r, t, b, r-l, b-t)
      assert r>=l and b>=t
      ret = []
      for c in range(img.shape[2]):
        #print([(abs(img[y,x,c]-img[y,x+1,c]),abs(img[y,x,c]-img[y+1,x,c])) for x in range(l,r) for y in range(t,b)])
        ret.append(sum(abs(img[y,x,c]-img[y,x+1,c])+abs(img[y,x,c]-img[y+1,x,c]) for x in range(l,r) for y in range(t,b))/(2*(r-l)*(b-t)))
      return ret

    @timeit
    def _waypoint(self, img, label=None, wait=False):
      if self.flag_skipwp:
        return
      cv2.imshow(label, img)
      if label is not None:
        fname = 'img/{}_{}.png'.format(self.timestamp, label)
        cv2.imwrite(fname, img)
        print('stored {}'.format(fname))
      if wait:
        cv2.waitKey()

    def _thresh(self, img):
      ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
      #print(th)
      m_sd_0 = cv2.meanStdDev(img)
      m_sd_1 = cv2.meanStdDev(img, mask=th)
      m_sd_2 = cv2.meanStdDev(img, mask=cv2.bitwise_not(th))
      #print('thresholding gray[{}]: {} {} {}'.format(ret, m_sd_2, m_sd_1, m_sd_0))
      return th, m_sd_2[0]-m_sd_1[0]

    def _thresh_my(self, img):
      img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      print(img_hsv[4,4])
      img_hsv = cv2.inRange(img_hsv, (110,140,159), (130,170,255))
      #img_hsv = cv2.inRange(img_hsv, (119,140,150), (121,170,255)) # good
      #img_hsv = cv2.inRange(img_hsv, (118,0,0), (120,255,255))
      ret,th = cv2.threshold(img_hsv,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
      #print(th)
      m_sd_0 = cv2.meanStdDev(img)
      m_sd_1 = cv2.meanStdDev(img, mask=th)
      m_sd_2 = cv2.meanStdDev(img, mask=cv2.bitwise_not(th))
      #print('thresholding gray[{}]: {} {} {}'.format(ret, m_sd_2, m_sd_1, m_sd_0))
      return th, m_sd_2[0]-m_sd_1[0]

    @timeit
    def _process_box(self, img_col, i=0, scale=15, mask=None):
      print('in _process_box')
      drop_color = None
      '''img_gray = np.uint8(img_gray_orig)
      #print(img_gray)
      self._waypoint(img_gray, 'gray_{:02d}'.format(i))
      th_gray, diff_gray = self._thresh(img_gray)
      #img_h = cv2.split(cv2.cvtColor(img_col, cv2.COLOR_BGR2HSV))[0]
      ret = cv2.split(cv2.cvtColor(img_col, cv2.COLOR_BGR2HSV))
      #img_h = np.uint8(ret[0]/2+ret[2]/2)
      img_h = ret[0]
      #print(img_h)
      self._waypoint(img_h, 'h_{:02d}'.format(i))
      th_h, diff_h = self._thresh(img_h)
      print('diff_gray, diff_h: {} {}'.format(diff_gray, diff_h))
      if diff_gray > diff_h and 0:
        print('using th_gray')
        th = th_gray
      else:
        print('using th_h')
        th = th_h
        #th = cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(th_gray), cv2.bitwise_not(th_h)))
        img_col = cv2.resize(img_col, (img_col.shape[1]*scale, img_col.shape[0]*scale))
        scale = 1
        th,d = self._thresh_my(img_col)
        #print(th)
        self._waypoint(th, 'thmy_{:02d}'.format(i), 0)
      #print(th)
      #m_sd_0 = cv2.meanStdDev(img_gray)
      #m_sd_1 = cv2.meanStdDev(img_gray, mask=th)
      #m_sd_2 = cv2.meanStdDev(img_gray, mask=cv2.bitwise_not(th))
      #print('thresholding gray[{}]: {} {} {}'.format(ret, m_sd_2, m_sd_1, m_sd_0))
      if th is None:
        return '', -1, ''
      #print('thresholding: ret = {}, th.shape = {}'.format(ret, th.shape))
      self._waypoint(th, 'thresh_{:02d}'.format(i))
      img_col_fg = cv2.bitwise_and(img_col, img_col, mask=cv2.bitwise_not(th))
      self._waypoint(img_col_fg, 'thresh_fg_{:02d}'.format(i))
      mean_col = cv2.mean(img_col, mask=cv2.bitwise_not(th))[0:3]
      img_1 = np.zeros((1, 1, 3), dtype='uint8')
      img_1[0, 0] = mean_col
      hsv = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
      print('foreground rgb: {} hsv: {}'.format(mean_col, hsv))
      hue = hsv[0, 0, 0] * 2
      if all(c > 150 for c in mean_col):
        drop_color = 'w'
      elif hue < 42: # 40
        drop_color = 'o'
      elif hue < 53: # 45
        drop_color = 'u'
      elif hue < 70: # 60
        drop_color = 'y'
      elif hue < 150: # 120
        drop_color = 'g'
      else: # 240
        drop_color = 'b'
      #th_tes = th[1:-5, 4:-5]
      th_tes = th[:-4, 3:-4]
      th_tes[:1] = 255
      th_tes[-1:] = 255
      th_tes[:, :1] = 255
      th_tes[:, -1:] = 255
      #print(th_tes)
      th_tes = cv2.resize(th_tes, (th_tes.shape[1]*scale, th_tes.shape[0]*scale))
      #th_tes = th
      bsize = 5
      th_tes = cv2.copyMakeBorder(th_tes, top=bsize, bottom=bsize, left=bsize, right=bsize, borderType=cv2.BORDER_CONSTANT, value=255)'''
      th_tes = img_col
      if 0:
        th_tes[:1] = 0
        th_tes[-4:] = 0
        th_tes[:, :5] = 0
        th_tes[:, -4:] = 0
      #bsize = 5
      #th_tes = cv2.copyMakeBorder(th_tes, top=bsize, bottom=bsize, left=bsize, right=bsize, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
      #th_tes = th_tes[:-4, 3:-4]
      #th_tes = cv2.resize(th_tes, (th_tes.shape[1]*scale, th_tes.shape[0]*scale))
      if mask is not None:
        th_tes = cv2.bitwise_and(th_tes, th_tes, mask=mask)
      th_tes = cv2.resize(th_tes, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
      #th_tes = cv2.resize(th_tes, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
      #th_tes = cv2.medianBlur(th_tes, 27)
      #th_tes = cv2.bilateralFilter(th_tes,9,75,75)
      #self._waypoint(th_tes, 'tes_{:02d}'.format(i))
      #item, conf = self._run_tesseract(th_tes)
      th_tes_mask,drop_color = self._text_thresh(th_tes, i)
      #self._waypoint(th_tes_mask, 'mask1_{:02d}'.format(i), 1)
      th_tes_mask = cv2.bitwise_not(th_tes_mask)
      #self._waypoint(th_tes_mask, 'mask2_{:02d}'.format(i), 1)
      self._waypoint(th_tes_mask, 'tesin_{:02d}'.format(i), 0)
      item, conf = self._run_tesseract(th_tes_mask)
      img_tesout = cv2.imread('tessinput.tif')
      self._waypoint(img_tesout, 'tesout_{:02d}'.format(i), 0)
      img_tesfg = cv2.bitwise_and(th_tes, th_tes, mask=cv2.bitwise_not(cv2.cvtColor(img_tesout, cv2.COLOR_BGR2GRAY)))
      #img1 = th_tes = self._text_thresh([th_tes,img_tesfg])
      #img1 = th_tes = self._text_thresh([th_tes])
      self._waypoint(img_tesfg, 'tesfg_{:02d}'.format(i), 0)
      #item, conf = self._run_tesseract(img1)
      #drop_color = self._detect_color(img_tesfg)
      return [item, conf, drop_color]

    def _text_thresh_hist(self, img):
      max_nb = 2
      hsv = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in img]
      fig, ax = plt.subplots(3)
      n_hue,_,_ = ax[0].hist(hsv[0][:,:,0].ravel(), np.linspace(0.001, 180., 180), alpha=0.5, density=True, label='all')
      #for i in range(len(n_hue)):
      #  print('{}: {}'.format(i, n_hue[i]))
      #n_hue_sum = [sum(n_hue[i:i+1+2*max_nb]) for i in range(len(n_hue)-1+2*max_nb)]
      #peak_hue_sum = max(n_hue_sum)
      #peak_hue = n_hue_sum.index(peak_hue_sum) + max_nb
      #print('peak_hue = {} peak_hue_sum = {}'.format(peak_hue, peak_hue_sum))
      '''img1 = cv2.inRange(hsv[0], (59,0,0), (60,255,255))
      #img1 = cv2.bitwise_and(img[0], img[0], mask=img1)
      #self._waypoint(img1, wait=True)
      img[0][np.where(img1>0)] = [0,0,255]
      print(np.where(img1>0))
      self._waypoint(img[0], wait=True)
      cv2.waitKey()'''
      ax[1].hist(hsv[0][:,:,1].ravel(), np.linspace(0.001, 255., 255), alpha=0.5, density=True, label='all')
      ax[2].hist(hsv[0][:,:,2].ravel(), np.linspace(0.001, 255., 255), alpha=0.5, density=True, label='all')
      ax[0].hist(hsv[1][:,:,0].ravel(), np.linspace(0.001, 180., 180), alpha=0.5, density=True, label='fg')
      ax[1].hist(hsv[1][:,:,1].ravel(), np.linspace(0.001, 255., 255), alpha=0.5, density=True, label='fg')
      ax[2].hist(hsv[1][:,:,2].ravel(), np.linspace(0.001, 255., 255), alpha=0.5, density=True, label='fg')
      plt.legend()
      plt.show()
      #cv2.waitKey()
      #img1 = cv2.inRange(hsv[0], (peak_hue-max_nb,100,100), (peak_hue+max_nb,255,255))
      #img1 = cv2.bitwise_and(img[0], img[0], mask=img1)
      #self._waypoint(img1, label='text_thresh', wait=True)
    
    def _find_peak(self, l, norm, nb_min=1, frac_min=0.1):
      bins,_,_ = plt.hist(l[l!=0], np.linspace(0., 180., 180))
      #print(bins)
      nb = nb_min
      l_sum = [sum(bins[i-nb:i+nb+1]) for i in range(nb, len(bins)-nb-1)]
      l_sum_max = max(l_sum)
      l_sum_pos = l_sum.index(l_sum_max)
      frac = 1.*l_sum_max / norm
      '''while frac < frac_min:
        print('find peak (while) l_sum_pos,frac,nb = {} {} {}'.format(l_sum_pos,frac,nb))
        nb = nb + 1
        if nb >= 20:
          break
        l_sum_max = l_sum_max + bins[l_sum_pos-nb] + bins[l_sum_pos+nb]
        frac = 1.*l_sum_max / norm
      print('find peak (return) l_sum_pos,frac,nb = {} {} {}'.format(l_sum_pos,frac,nb))'''
      return l_sum_pos,frac,nb
    
    def _text_thresh(self, img, i):
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      #self._waypoint(hsv, wait=1)
      fg = cv2.inRange(hsv, (0,0,150), (255,25,255))
      frac = 1.*cv2.countNonZero(fg) / (img.shape[0] * img.shape[1])
      #print('frac white = {}'.format(frac))
      #self._waypoint(fg, wait=1)
      if frac > 0.05:
        col = 'w'
      else:
        hsv_filt = cv2.bitwise_and(hsv, hsv, mask=cv2.inRange(hsv, (0, 125, 125), (255,250,255)))
        hue_pos,_,hue_nb = self._find_peak(hsv_filt[:,:,0].ravel(), (img.shape[0] * img.shape[1]))
        sat_pos,_,sat_nb = self._find_peak(hsv_filt[:,:,1].ravel(), (img.shape[0] * img.shape[1]))
        nb = 1
        fg = cv2.inRange(hsv, (hue_pos-nb,sat_pos-2*nb,125), (hue_pos+nb+0.001,sat_pos+2*nb,255))
        frac = cv2.countNonZero(fg) / (img.shape[0] * img.shape[1])
        while frac < 0.16:
          nb = nb + 1
          if nb > 70:
            break
          fg = cv2.inRange(hsv, (hue_pos-nb,sat_pos-2*nb,125), (hue_pos+nb+0.001,sat_pos+2*nb,255))
          frac = cv2.countNonZero(fg) / (img.shape[0] * img.shape[1])
          #print('frac,nb: {} {}'.format(frac, nb))
        #print('frac,nb: {} {}'.format(frac, nb))
        #fg = cv2.inRange(hsv, (hue_pos-nb,0,0), (hue_pos+nb+0.001,255,255))
        #fg = cv2.inRange(hsv, (118-5,153-10,125), (118+5+0.001,153+10,255))
        #fg = cv2.inRange(hsv, (118-5,0,0), (118+5+0.001,255,255))
        cols = {'u': 23.5, 'b': 119, 'o': 20, 'g': 120, 'y': 29}
        cols_dif = {k: abs(hue_pos-v) for k,v in cols.items()}
        col = min(cols_dif, key=cols_dif.get)
        print('best col {} = {} [central {}]'.format(hue_pos, col, cols[col]))
      #self._waypoint(fg, 'thr_{:02d}'.format(i), wait=0)
      return fg,col

    def _detect_color(self, img):
      mean_col = np.mean(img, axis=(0,1))
      img_1 = np.zeros((1, 1, 3), dtype='uint8')
      img_1[0, 0] = mean_col
      hsv = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
      #print('foreground rgb: {} hsv: {}'.format(mean_col, hsv))
      hue = hsv[0, 0, 0] * 2
      if all(c > 10 for c in mean_col) and abs(mean_col[0]-mean_col[1])<10 and abs(mean_col[0]-mean_col[2])<10 and abs(mean_col[2]-mean_col[1])<10:
        drop_color = 'w'
      elif hue < 42: # 40
        drop_color = 'o'
      elif hue < 53: # 45
        drop_color = 'u'
      elif hue < 70: # 60
        drop_color = 'y'
      elif hue < 150: # 120
        drop_color = 'g'
      else: # 240
        drop_color = 'b'
      return drop_color
    
    def _get_drop_from_img(self, img):
        return [self.items[random.randrange(len(self.items))] for i in range(random.randrange(1, 5))]
    
    @timeit
    def _run_tesseract(self, image):
      res = pytesseract.image_to_data(image, config=self.config_tesseract+' --psm 7'+' -c tessedit_write_images=True')
      #res = pytesseract.image_to_string(image, config=self.config_tesseract+' --psm 7')
      print(res)
      #print('res: {}'.format(res))
      #conf, text = res.splitlines()[-1].split()[-2:]
      conf, text = [], []
      for l in res.splitlines()[1:]:
        #print(l.split())
        c, t = l.split()[-2:]
        #print(c, t)
        if t != '-1' and float(c) >= 0:
          conf.append(float(c))
          text.append(t)
      #print([l.split()[-2:] for l in res.splitlines()])
      #print(zip([l.split()[-2:] for l in res.splitlines()]))
      #conf, text = zip([l.split()[-2:] for l in res.splitlines()])
      print('tesseract: conf = {} text = {}'.format(conf, text))
      if len(conf) != 0:
        return ' '.join(text), sum(conf)/len(conf)
      else:
        return '', 0


dropper = Drop()
    

#def py_droprec(mode, img, timestamp=0, y0=20, y1=570, x0=None, x1=None, signal=[]):
@timeit
def py_droprec(mode, img, timestamp=0, signal=[], y0=None, y1=None, x0=None, x1=None):
#def py_droprec(mode, img):
#def py_droprec(mode, ):
  #return '42\n65'
  #print('in py_droprec')
  #print(img[0:2,0:2])
  #if img.shape[2] == 4:
  #  print(img[np.where(img[:,:,3] != 255)])
  #print(img)
  if mode == 'test':
    pass
  elif mode == 'newrun' or mode == 'append':
    dropper.flag_allboxes = 0
    dropper.flag_selectedbox = 1
    dropper.flag_skipwp = 1
  else:
    assert 0
  print('SZ droprec timestamp = {}, crop = [{}:{}, {}:{}], signal = {}'.format(timestamp, y0, y1, x0, x1, signal))
  signal_copy = signal[:]
  for s in signal_copy:
    if y0 is not None:
      s[0] = s[0] - y0
    if x0 is not None:
      s[1] = s[1] - x0
  dropper.img_orig = img
  dropper.timestamp = timestamp
  img_cropped = img[y0:y1, x0:x1]
  drop = dropper.process_drop(img_cropped, signal_copy)
  dropper.reset_drop()
  #print(drop)
  #s = '\n'.join(['({}){}[{}]'.format(d[2], d[0], d[1]) for d in drop])
  #s = '\n'.join('({}){}[{},{},{}]'.format(d[2], d[0], d[3], d[4], d[5]) for d in drop)
  s = '\n'.join(dropper._get_coloured_item(d[0], d[2]) for d in drop)
  #print('returning {}'.format(s))
  #cv2.waitKey()
  #print('ce dupa')
  return s


image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath

def list_images(basePath, contains=None):
    print(basePath)
    if not os.path.isdir(basePath):
        #print('dupa')
        return [basePath]
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)

GHP = 'Greater Healing Potion'
GMP = 'Greater Mana Potion'
SHP = 'Super Healing Potion'
SMP = 'Super Mana Potion'
HP = 'Healing Potion'
MP = 'Mana Potion'
FRP = 'Full Rejuvenation Potion'
RP = 'Rejuvenation Potion'
def get_img(img_name, signal=None):
  print('img_name = {}'.format(img_name))
  img = cv2.imread(img_name)
  timestamp = os.path.splitext(os.path.basename(img_name))[0]
  if signal is None:
    sig_file = os.path.splitext(img_name)[0] + '.txt'
    if os.path.exists(sig_file):
      print('reading signal from {}'.format(sig_file))
      signal = []
      with open(sig_file) as f:
        for l in f.readlines():
          y,x,x1,c = l.split(' ')[:4]
          item = ' '.join(l.split(' ')[4:]).rstrip()
          try:
            eval(item)
          except (NameError, SyntaxError):
            pass
          else:
            item = eval(item)
          signal.append([int(y), int(x), int(x1), item, c])
    else:
      signal = []
  return img, timestamp, signal

def store_sig(img_name, rets, overwrite=False):
  sig_file = os.path.splitext(img_name)[0] + '.txt'
  if not overwrite and os.path.exists(sig_file):
    return
  with open(sig_file, 'w') as fout:
    for ret in rets.split('\n'):
      print(ret)
      y,x,x1 = ret.split('[')[1][:-1].split(',')
      item = ret.split(')')[1].split('[')[0]
      col = ret[1]
      fout.write('{} {} {} {} {}\n'.format(y, x, x1, col, item))
  print('signal written to {}'.format(sig_file))

if __name__ == '__main__':
  #py_droprec(mode, cv2.imread('../../502/screens_1/95621693075.png'))
  #a
  _do_time = 0
  atexit.register(print_all_time)
  np.set_printoptions(threshold=np.inf, linewidth=np.inf)
  #dropper.cuts = [-7.16129, -7.0, -10.4838705, -38.83871, -38.451614, -36.322582, -5.096774, -4.1827955, -5.419355, -21.655914, -20.903225, -22.838709, -39.5, -30.375, -22.625, -42.0, -33.625, -35.5, 44.62088, -19.088888]
  #dropper.cuts_x1 = [-9.1875, -12.9375, -12.875, -32.25, -37.25, -44.0625, -16.022223]
  #dropper.cuts = [-7.16129, -7.0, -10.4838705, -38.83871, -38.451614, -36.322582, -7.2645164, -6.0903225, -8.858065, -21.941935, -22.754839, -26.393549, -39.5, -30.375, -22.625, -42.0, -33.625, -35.5, 39.403847, -19.088888]
  #dropper.cuts_x1 = [-15.625, -16.5625, -13.1875, -32.25, -37.25, -44.0625, -16.15]
  #dropper.cuts = [-7.16129, -7.0, -10.4838705, -38.83871, -38.451614, -36.322582, -7.2645164, -6.0903225, -8.858065, -21.941935, -22.754839, -26.393549, -39.5, -30.375, -22.625, -42.0, -33.625, -35.5, 39.403847, -19.088888]
  #dropper.cuts_x1 = [-15.625, -16.5625, -13.1875, -32.25, -37.25, -44.0625, -16.15, -105.72177, -107.47177, -108.33064]
  #dropper.cuts = [-33.548386, -34.322582, -23.290323, -40.870968, -39.967743, -36.322582, -33.741936, -37.032257, -43.35484, -35.35484, -39.225807, -50.064518, -39.5, -30.375, -22.625, -42.0, -33.625, -35.5, 36.33173, -19.088888]
  #dropper.cuts_x1 = [-15.625, -16.5625, -13.1875, -32.25, -37.25, -44.0625, -16.15, -123.3, -124.725, -125.1125]
  #dropper.cuts = [-7.16129, -7.0, -10.4838705, -38.83871, -38.451614, -36.322582, -7.2645164, -6.0903225, -8.858065, -21.941935, -22.754839, -26.393549, -39.5, -30.375, -22.625, -42.0, -33.625, -35.5, 39.403847, -19.088888]
  #dropper.cuts_x1 = [-15.625, -16.5625, -13.1875, -32.25, -37.25, -44.0625, -16.15, -105.72177, -107.47177, -108.33064]
  #dropper.cuts = [-7.16129, -7.0, -10.4838705, -38.83871, -38.451614, -36.322582, -7.2645164, -6.0903225, -8.858065, -21.941935, -22.754839, -26.393549, -39.5, -30.375, -22.625, -42.0, -33.625, -35.5, 39.403847, -19.088888]
  #dropper.cuts_x1 = [-15.625, -16.5625, -13.1875, -32.25, -37.25, -44.0625, -16.15, 17.5, 26.0, 30.6875]
  #dropper.cuts = [-7.16129, -7.0, -10.4838705, -38.83871, -38.451614, -36.322582, -7.2645164, -6.0903225, -8.858065, -21.941935, -22.754839, -26.393549, -39.5, -30.375, -22.625, -42.0, -33.625, -35.5, 39.403847, -19.088888]
  #dropper.cuts_x1 = [-15.625, -16.5625, -13.1875, -32.25, -37.25, -44.0625, -16.15, 1.0]
  #dropper.cuts = [-7.16129, -7.0, -10.4838705, -6.8817225, -38.83871, -38.451614, -36.322582, -36.881725, -7.2645164, -6.0903225, -8.858065, -7.4043007, -21.941935, -22.754839, -26.393549, -21.632257, -39.5, -30.375, -22.625, -30.145834, -42.0, -33.625, -35.5, -32.14583, 39.403847, -19.088888]
  #dropper.cuts_x1 = [-15.625, -16.5625, -13.1875, -15.124999, -32.25, -37.25, -44.0625, -36.812496, -16.15]
  #dropper.cuts = [-7.16129, -7.0, -10.4838705, -6.8817225, -38.83871, -38.451614, -36.322582, -36.881725, -7.2645164, -6.0903225, -8.858065, -7.4043007, -21.941935, -22.754839, -26.393549, -21.632257, -39.5, -30.375, -22.625, -30.145834, -42.0, -33.625, -35.5, -32.14583, 39.403847, -19.088888]
  #dropper.cuts_x1 = [-15.625, -16.5625, -13.1875, -15.124999, -32.25, -37.25, -44.0625, -36.812496, -16.15, -31.428572, -14.091219, -13.324709, -11.878683]
  #dropper.cuts = [-7.16129, -7.0, -10.4838705, -6.8817225, -38.83871, -38.451614, -36.322582, -36.881725, -11.501648, -10.992908, -13.68421, -10.386243, -89.89042, -79.00921, -35.410526, -58.58474, 
  #                -7.2645164, -6.0903225, -8.858065, -7.4043007, -21.941935, -22.754839, -26.393549, -21.632257, -16.058187, -10.803388, -13.728627, -13.370353, -52.581924, -49.029125, -40.905907, -39.062557,
  #                 -39.5, -30.375, -22.625, -30.145834, -42.0, -33.625, -35.5, -32.14583, -50.348026, -54.018696, -57.278477, -54.317898, -74.013916, -77.196266, -84.81013, -79.22403, 39.403847, -19.088888]
  #dropper.cuts_x1 = [-15.625, -16.5625, -13.1875, -15.124999, -32.25, -37.25, -44.0625, -36.812496, -31.428572, -14.091219, -13.324709, -11.878683, -90.87452, -89.43396, -79.430374, -78.7234, -16.15]
  #dropper.cuts = [-98.92, -33.936, -16.0, -50.162, -99.307, -66.388, -84.975, -51.273, -94.412, -88.906, -86.392, -91.833, -94.782, -89.387, -88.978, -92.332, -44.42, -6.091, -8.859, -7.405, -76.033, -57.673, -92.135, -37.248, -27.535, -10.804, -13.729, -13.371, -52.582, -49.359, -76.747, -39.063, -39.5, -30.375, -22.625, -30.146, -47.0, -178.875, -219.75, -148.542, -50.349, -54.019, -57.279, -54.318, -74.014, -77.197, -89.058, -79.225, 39.403, -19.089]
  #dropper.cuts_x1 = [-63.438, -16.563, -13.188, -15.125, -63.438, -80.25, -111.125, -54.98, -44.499, -14.092, -13.325, -11.879, -90.875, -89.434, -79.431, -78.724, -23.076]
  #dropper.cuts = [-98.92, -33.936, -16.0, -50.162, -99.307, -66.388, -84.975, -51.273, -94.412, -88.906, -86.392, -91.833, -94.782, -89.387, -88.978, -92.332, -44.42, -6.091, -8.859, -7.405, -76.033, -57.673, -92.135, -37.248, -27.535, -10.804, -13.729, -13.371, -52.582, -49.359, -76.747, -39.063, -39.5, -30.375, -22.625, -30.146, -47.0, -178.875, -219.75, -148.542, -50.349, -54.019, -57.279, -54.318, -74.014, -77.197, -89.058, -79.225, -37.401, -20.0, -26.0, -19.534, -70.801, -184.601, -226.2, -153.801, -34.954, -28.468, -43.919, -33.25, -70.518, -72.393, -88.706, -60.314, 39.403, -19.089]
  #dropper.cuts_x1 = [-63.438, -16.563, -13.188, -15.125, -63.438, -80.25, -111.125, -54.98, -44.499, -14.092, -13.325, -11.879, -90.875, -89.434, -79.431, -78.724, -49.8, -32.401, -35.6, -28.867, -59.201, -87.6, -122.6, -53.467, -41.74, -39.131, -77.392, -52.754, -128.696, -127.827, -85.218, -111.885, -23.076]
  dropper.cuts = [-98.92, -33.936, -16.0, -50.162, -99.307, -66.388, -84.975, -51.273, -94.412, -88.906, -86.392, -91.833, -94.782, -89.387, -88.978, -92.332, -44.42, -6.091, -8.859, -7.405, -76.033, -57.673, -92.135, -37.248, -27.535, -10.804, -13.729, -13.371, -58.738, -49.515, -76.747, -53.028, -39.5, -30.375, -22.625, -30.146, -52.563, -178.875, -219.75, -148.542, -50.349, -54.019, -57.279, -54.318, -74.014, -77.197, -89.058, -79.225, -37.401, -20.0, -26.0, -19.534, -70.801, -184.601, -226.2, -153.801, -34.954, -28.468, -43.919, -33.25, -71.429, -72.393, -88.706, -60.314, 38.007, -20.712]
  dropper.cuts_x1 = [-63.438, -16.563, -15.438, -15.125, -63.438, -80.25, -111.125, -54.98, -44.499, -29.111, -27.66, -28.188, -100.519, -133.196, -128.071, -78.724, -56.0, -53.6, -52.401, -50.734, -65.6, -87.6, -122.6, -62.734, -479.167, -487.5, -166.667, -212.281, -500.0, -500.0, -366.667, -227.486, -35.617]
  dropper.flag_train = 0
  dropper.flag_mycheck = 0
  dropper.flag_textrec = 1
  dropper.flag_allboxes = 1
  dropper.flag_selectedbox = 0
  dropper.flag_skipwp = 0
  mode = 'test'
  if len(sys.argv) == 1:
    #ret = py_droprec(mode, cv2.imread('../../502/screens_1/95621693075.png'), y0=300, y1=345, x0=275, x1=465, signal=[[4+300, 10+275], [22+300, 62+275], [1+300, 1+275]])
    #ret = py_droprec(mode, cv2.imread('../158458820669.png'), signal=[[145,245,361,'Martel De Fer','y'],[145,363,553,GHP,'w'],[145,555,745,GHP,'w'],[160,221,319,'Tusk Sword','g'],[160,321,526,FRP,'w'],[175,232,408,GMP,'w'],[190,222,351,'Studded Leather','b'],[205,226,381,SMP,'w'],[220,249,425,GMP,'w'],[235,176,366,GHP,'w'],[251,270,403,'Conquest Sword','y'],[268,207,397,GHP,'w']])
    # trained
    #ret = py_droprec(mode, *get_img('../../502/screens_1/94456900671.png', signal=[[249,328,419,'Bone Wand','y'],[249,421,576,SMP,'w'],[249,578,783,FRP,'w'],[264,268,407,'Flawed Amethyst','w'],[264,409,574,RP,'w'],[279,307,476,SHP,'w'],[294,312,400,'Bone Shield','b'],[309,289,458,SHP,'w'],[324,326,491,RP,'w'],[339,244,434,GHP,'w'],[357,386,432,'Jewel','y'],[376,296,451,SMP,'w']]))
    #
    #ret = py_droprec(mode, *get_img('../../502/screens_1/95217013080.png'))
    #ret = py_droprec(mode, *get_img('../../502/screens_1/94456900671.png'))
    #ret = py_droprec(mode, *get_img('../../502/screens_1/94778304482.png'))
    #
    ret = py_droprec(mode, *get_img('../../502/screens_1/95509173077.png'))
    #ret = py_droprec(mode, *get_img('../../502/screens_1/97899599672.png'))
    #ret = py_droprec(mode, *get_img('../../502/screens_1/96566595942.png'))
    #ret = py_droprec(mode, *get_img('../../502/screens_1/95842955271.png'))
    #ret = py_droprec(mode, *get_img('../../502/screens_1/97200755937.png'))
    # selected box
    #ret = py_droprec(mode, *get_img('../../502/screens_1/95637733084.png', signal=[[299,184,374,GHP,'w']]))
    #ret = py_droprec(mode, *get_img('../../502/screens_1/96130115273.png'))
    # blue
    # ../../502/screens_1/94778304482.png
    # ../../502/screens_1/96189675273.png
    # ../../502/screens_1/98011159677.png
    # ../../502/screens_1/98362279672.png
    # ../../502/screens_1/96495515941.png
    # ../../502/screens_1/98492479671.png
    # ../../502/screens_1/97327835937.png
    # ../../502/screens_1/.png
    # ../../502/screens_1/.png
    # ../../502/screens_1/.png
    print(ret)
  else:
    for img_name in list_images(sys.argv[1]):
      print(img_name)
      #img = cv2.imread(img_name)
      timestamp = os.path.splitext(os.path.basename(img_name))[1]
      ret = py_droprec(mode, *get_img(img_name))
      print(ret)
      #store_sig(img_name, ret, overwrite=0)
  dropper.stats()
