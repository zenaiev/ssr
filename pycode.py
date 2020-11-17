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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

class Drop:
    def __init__(self):
      self.box_height = 16
      self.basedir = '/home/zenaiev/games/Diablo2/ssr/'
      self.config_tesseract = "-c tessedit_char_whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -'"
      with open(self.basedir + 'd2-drop-items.txt') as f:
          self.items = [l[:-1] for l in f.readlines()]
      self.current_drop = []
      self.flag_train = False
      self.sig = []
      self.bkg = []
      self.sig_x1 = []
      self.bkg_x1 = []
      self.cuts = []
      self.cuts_x1 = []
      self.vars_title = []
      self.vars_title_x1 = []
    
    def stats(self):
      if self.flag_train:
        cuts_xy, cuts_x1 = [], []
        print('{:20s}{:>18s}  {:>18s}  {:>5s} [{:>6s}]{:>8s}{:>8s}'.format('Variable', 'sig', 'bkg', 'cut', 'q', 'sig_egg', 'bkg_eff'))
        for sig,bkg,title,cuts in zip([self.sig, self.sig_x1], [self.bkg, self.bkg_x1], [self.vars_title, self.vars_title_x1], [cuts_xy, cuts_x1]):
          for i in range(len(sig[0])):
            if len(title[i].split('[')) > 1:
              if i%3 == 0:
                fig, axes = plt.subplots(3, 1, sharex=True, squeeze=True)
                fig.suptitle(title[i].split('[')[0])
                fig.canvas.set_window_title(title[i].split('[')[0])
                plt.subplots_adjust(hspace = 0.0, left = 0.01, right = 0.99)
              ax = axes[i%3]
              if i%3 == 2:
                store = title[i].split('[')[0]
              else:
                store = None
            else:
              fig = plt.figure(title[i])
              ax = None
              store = title[i]
            bkg_sample = random.sample(bkg, min(10000, len(bkg)))
            cuts.append(self._plot(title[i], [v[i] for v in sig], [v[i] for v in bkg_sample], np.linspace(0, 300, 100), [1.0, 0.90], ax, col=['b','g','r'][i%3], store=store))
        print('dropper.cuts = {}'.format(cuts_xy))
        print('dropper.cuts_x1 = {}'.format(cuts_x1))
        plt.show(block=False)
        cv2.waitKey()
        #print(self.cuts)
        assert all(abs(v1-v2)<1e-5 for v1,v2 in zip(self.cuts, cuts_xy))
        #print(self.cuts_x1)
        assert all(abs(v1-v2)<1e-5 for v1,v2 in zip(self.cuts_x1, cuts_x1))

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
      #print(bins)
      if ax is None:
        #ax = plt
        ax = plt.gca()
      bins = np.linspace(min(sig_min, bkg_min), max(sig_max, bkg_max), 100)
      if bins is not None:
        ax.hist(bkg, bins, alpha=0.5, density=True, label='bkg({}) {:.0f}[{:.0f},{:.0f}]'.format(len(bkg), bkg_mean,bkg_min,bkg_max), color='k')
      else:
        _,bins,_ = ax.hist(bkg, alpha=0.5, density=True, label='bkg({}) {:.0f}[{:.0f},{:.0f}]'.format(len(bkg), bkg_mean,bkg_min,bkg_max), color=col)
      ax.hist(sig, bins, alpha=0.5, density=True, label='sig({}) {:.0f}[{:.0f},{:.0f}]'.format(len(sig), sig_mean,sig_min,sig_max), color=col)
      ax.legend(loc=2)
      ax.text(0.0,0.45,'\n'.join(q_str),horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
      ax.axes.get_yaxis().set_visible(False)
      if store is not None:
        plt.savefig('vars/{}.pdf'.format(store))
        plt.savefig('vars/{}.png'.format(store))
      return sig_min

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

    def process_drop(self, img, signal=[]):
        self._waypoint(self.img_orig, 'orig')
        self._waypoint(img, 'input')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_gray = np.int16(img_gray)
        img_gray = np.float32(img_gray)
        img_float = np.float32(img)
        ys, ls, rs = self._boxes(img_float, signal=signal)
        ys = [y+1 for y in ys]
        ls = [y+1 for y in ls]
        rs = [x for x in rs]
        print(ys, ls, rs, sep='\n')
        for y,l,r in zip(ys, ls, rs):
          cv2.rectangle(img, (l-1, y-1), (r+1, y+self.box_height), (0, 255, 0), 1)
        drop = []
        '''i = 0
        for y,l,r in zip(ys, ls, rs):
          drop.append(self._process_box(img_gray[y:y+self.box_height, l:r+1], img[y:y+self.box_height, l:r+1], i))
          print('box {:2d}: y = {} l = {} r = {} color = {} conf = {} item = {}'.format(i, y, l, r, drop[-1][2], drop[-1][1], drop[-1][0]))
          i += 1
        print('drop {}: {}'.format(len(drop), drop))'''
        self._waypoint(img, 'output', 1)
        return drop

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

    def _boxes(self, img, signal=[]):
      assert img.shape[2] == 3 # bgr
      box_x = 31
      n_box_x = 7
      #n_box_x = 1
      box_y = self.box_height
      #box_y = 6
      mar_l, mar_r, mar_t, mar_b = 4, 4, 3, 5
      #mar_l, mar_r, mar_t, mar_b = 2, 2, 2, 2
      t = cv2.filter2D(img, -1, np.matrix('1; -2'), anchor=(0,0))
      b = cv2.filter2D(img, -1, np.matrix('-2; 1'), anchor=(0,0))
      l = cv2.filter2D(img, -1, np.matrix('1 -2'), anchor=(0,0))
      r = cv2.filter2D(img, -1, np.matrix('-2 1'), anchor=(0,0))
      ta = np.absolute(t)*-1
      ba = np.absolute(b)*-1
      la = np.absolute(l)*-1
      ra = np.absolute(r)*-1
      tsum = [cv2.filter2D(t, -1, np.matrix('1 '*box_x*i), anchor=(0,0))/(box_x*i) for i in range(1,n_box_x+1)]
      tsum_max = self._max_img_from_list(tsum)
      tasum = [cv2.filter2D(ta, -1, np.matrix('1 '*box_x*i), anchor=(0,0))/(box_x*i) for i in range(1,n_box_x+1)]
      tasum_max = self._max_img_from_list(tasum)
      bsum = [cv2.filter2D(b, -1, np.matrix('1 '*box_x*i), anchor=(0,0))/(box_x*i) for i in range(1,n_box_x+1)]
      bsum_max = self._max_img_from_list(bsum)
      basum = [cv2.filter2D(ba, -1, np.matrix('1 '*box_x*i), anchor=(0,0))/(box_x*i) for i in range(1,n_box_x+1)]
      basum_max = self._max_img_from_list(basum)
      lsum = cv2.filter2D(l, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0))/(box_y)
      lasum = cv2.filter2D(la, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0))/(box_y)
      rsum = cv2.filter2D(r, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0))/(box_y)
      rasum = cv2.filter2D(ra, -1, np.matrix(';'.join('1'*box_y)), anchor=(0,0))/(box_y)
      rasumtext = cv2.filter2D(ra, -1, np.matrix(';'.join('0'*mar_t)+';'+';'.join('1'*(box_y-mar_b-mar_t))+';'+';'.join('0'*mar_b)), anchor=(0,0))/(box_y)
      rasumtext = np.maximum(np.maximum(rasumtext[:, :, :1], rasumtext[:, :, 1:2]), rasumtext[:, :, 2:])
      dxa = np.absolute(cv2.filter2D(img, -1, np.matrix('1 -1'), anchor=(0,0)))
      dya = np.absolute(cv2.filter2D(img, -1, np.matrix('1; -1'), anchor=(0,0)))
      dxya = dxa + dya
      assert box_y > (mar_b+mar_t+1)
      assert box_x > (mar_l+mar_r+1)
      boxsum = [cv2.filter2D(dxya, -1, np.matrix(';'.join(['1 '*(box_x-mar_l-1)*i]*(box_y-mar_t-mar_b-1))), anchor=(0,0))/(2*(box_x-mar_l-1)*i*(box_y-mar_b-mar_t-1)) for i in range(1,n_box_x+1)]
      maxrgb = lambda img: np.maximum(np.maximum(img[:, :, :1], img[:, :, 1:2]), img[:, :, 2:])
      boxsum = [maxrgb(i) for i in boxsum]
      boxsum_max = self._max_img_from_list(boxsum)
      marlsum = maxrgb(cv2.filter2D(dxya, -1, np.matrix(';'.join(['1 '*(mar_l-1)]*(box_y-1))), anchor=(0,0))/(2*(mar_l-1)*(box_y-1)))
      marrsum = cv2.filter2D(dxya, -1, np.matrix(';'.join(['1 '*(mar_r-1)]*(box_y-1))), anchor=(0,0))/(2*(mar_r-1)*(box_y-1))
      num_rows, num_cols = marrsum.shape[:2]
      translation_matrix = np.float32([ [1,0,mar_r-1], [0,1,0] ])
      marrsum = cv2.warpAffine(marrsum, translation_matrix, (num_cols,num_rows))
      marrsum = maxrgb(marrsum)
      if len(self.vars_title) == 0:
        self.vars_title += ['tsum_max'+'['+c+']' for c in 'bgr']
        self.vars_title += ['tasum_max'+'['+c+']' for c in 'bgr']
        self.vars_title += ['bsum_max'+'['+c+']' for c in 'bgr']
        self.vars_title += ['basum_max'+'['+c+']' for c in 'bgr']
        self.vars_title += ['lsum'+'['+c+']' for c in 'bgr']
        self.vars_title += ['lasum'+'['+c+']' for c in 'bgr']
        self.vars_title += ['boxsum_max']
        self.vars_title += ['marlsum']
      img_vars = cv2.merge(
        () +
        tuple(tsum_max[:-box_y-1, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(tasum_max[:-box_y-1, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(bsum_max[box_y:-1, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(basum_max[box_y:-1, 1:-box_x, c] for c in range(img.shape[2])) +
        tuple(lsum[1:-box_y, :-box_x-1, c] for c in range(img.shape[2])) +
        tuple(lasum[1:-box_y, :-box_x-1, c] for c in range(img.shape[2])) +
        tuple((boxsum_max[mar_t+1:-box_y+mar_t, mar_l+1:-box_x+mar_l],)) +
        tuple((marlsum[1:-box_y, 1:-box_x]*-1,)) +
      ())
      #print(img[0:3, 0:3])
      if len(self.vars_title_x1) == 0:
        self.vars_title_x1 += ['rsum'+'['+c+']' for c in 'bgr']
        self.vars_title_x1 += ['rasum'+'['+c+']' for c in 'bgr']
        self.vars_title_x1 += ['marrsum']
      img_vars_x1 = cv2.merge(
        () +
        tuple(rsum[1:-box_y, :-1, c] for c in range(img.shape[2])) +
        tuple(rasum[1:-box_y, :-1, c] for c in range(img.shape[2])) +
        tuple((marrsum[1:-box_y, 0:-1]*-1,)) +
      ())
      assert img_vars.shape[0]+box_y+1 == img.shape[0] and img_vars.shape[1]+box_x+1 == img.shape[1]
      #assert img_vars.shape[2] == 18
      ys, xs, x1s = [], [], []
      sig = np.zeros(img_vars.shape[0:2])
      if self.flag_train:
        print_str = '{:4.0f}{:4.0f}{:4.0f}  {:25s}{:3s}' + '{:5.0f}'*len(self.vars_title) + '{:5.0f}'*len(self.vars_title_x1)
        print('Variables: {}'.format({iv: v for iv,v in enumerate(self.vars_title + self.vars_title_x1)}))
        print(print_str.replace('.0f}', 's}').replace('{:', '{:>').format('y','x0','x1','Drop','c',*('v'+str(i) for i in range(len(self.vars_title) + len(self.vars_title_x1)))))
        for s in signal:
          #print(s)
          assert s[0] > 0 and s[1] > 0
          y, x, x1 = s[0]-1, s[1]-1, s[2]
          if 0:
            my_tsum = sum(img[y, xx]-2*img[y+1, xx] for xx in range(x+1, x+1+box_x))/box_x
            my_tasum = -1*sum(abs(img[y, xx]-2*img[y+1, xx]) for xx in range(x+1, x+1+box_x))/box_x
            my_bsum = sum(-2*img[y+box_y, xx]+img[y+box_y+1, xx] for xx in range(x+1, x+1+box_x))/box_x
            my_basum = -1*sum(abs(2*img[y+box_y, xx]-img[y+box_y+1, xx]) for xx in range(x+1, x+1+box_x))/box_x
            my_lsum = sum(img[yy, x]-2*img[yy, x+1] for yy in range(y+1, y+1+box_y))/box_y
            my_lasum = -1*sum(abs(img[yy, x]-2*img[yy, x+1]) for yy in range(y+1, y+1+box_y))/box_y
            my_boxsum = max(self._calc_adxdysum(img, x+1+mar_l, x+box_x, y+1+mar_t, y+box_y-mar_b))
            my_marlsum = max(self._calc_adxdysum(img, x+1, x+mar_l, y+1, y+box_y))*-1
            my_vars = sum((list(img) for img in (my_tsum, my_tasum, my_bsum, my_basum, my_lsum, my_lasum, [my_boxsum], [my_marlsum])), [])
            my_diff = [a-b for a,b in zip(my_vars, img_vars[y, x])]
            if not all(abs(d) < 1e-5 for d in my_diff):
              print('img {}'.format(img_vars[y, x]))
              print('my {}'.format(my_vars))
              print('diff (len diff {}): {}'.format(len(my_vars)-len(img_vars[y, x]), my_diff))
              assert 0
          print(print_str.format(*s, *img_vars[y, x], *img_vars_x1[y, x]))
          self.sig.append(img_vars[y, x])
          sig[y, x]
          ys.append(y)
          xs.append(x)
          x1s.append(x1)
          for x1_scan in range(x+box_x, img_vars_x1.shape[1]):
            if x1_scan == x1:
              if 0:
                my_rsum = sum(-2*img[yy, x1]+img[yy, x1+1] for yy in range(y+1, y+1+box_y))/box_y
                my_rasum = -1*sum(abs(-2*img[yy, x1]+img[yy, x1+1]) for yy in range(y+1, y+1+box_y))/box_y
                my_marrsum = max(self._calc_adxdysum(img, x1-mar_r+1, x1, y+1, y+box_y))*-1
                my_vars = sum((list(img) for img in (my_rsum, my_rasum, [my_marrsum])), [])
                my_diff = [a-b for a,b in zip(my_vars, img_vars_x1[y, x1])]
                if not all(abs(d) < 1e-5 for d in my_diff):
                  print('img {}'.format(img_vars_x1[y, x1]))
                  print('my {}'.format(my_vars))
                  print('diff (len diff {}): {}'.format(len(my_vars)-len(img_vars_x1[y, x1]), my_diff))
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
            if x < x1_last:
              continue
            if all(v >= (c-1e-5) for v,c in zip(img_vars[y, x], self.cuts)):
              #print('x,y {} {}'.format(x, y))
              for x1 in range(x+box_x, sig.shape[1]):
                if all(v >= (c-1e-5) for v,c in zip(img_vars_x1[y, x1], self.cuts_x1)):
                  x1s.append(x1)
                  sig[y, x] = 1
                  ys.append(y)
                  xs.append(x)
                  x1_last = x1
                  #cv2.waitKey()
                  break
      return (ys, xs, x1s)
    
    def _calc_adxdysum(self, img, l, r, t, b):
      #print(l, r, t, b, r-l, b-t)
      assert r>=l and b>=t
      ret = []
      for c in range(img.shape[2]):
        #print([(abs(img[y,x,c]-img[y,x+1,c]),abs(img[y,x,c]-img[y+1,x,c])) for x in range(l,r) for y in range(t,b)])
        ret.append(sum(abs(img[y,x,c]-img[y,x+1,c])+abs(img[y,x,c]-img[y+1,x,c]) for x in range(l,r) for y in range(t,b))/(2*(r-l)*(b-t)))
      return ret

    def _waypoint(self, img, label, wait=False):
      cv2.imshow(label, img)
      fname = 'img/{}_{}.png'.format(self.timestamp, label)
      cv2.imwrite(fname, img)
      print('stored {}'.format(fname))
      if wait:
        cv2.waitKey()

    def _process_box(self, img_gray_orig, img_col, i=0):
      img_gray = np.uint8(img_gray_orig)
      ret,th = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
      if th is None:
        return '', -1, ''
      print('thresholding: ret = {}, th.shape = {}'.format(ret, th.shape))
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
      th_tes = th[1:-4, 4:-4]
      th_tes = cv2.resize(th_tes, (th_tes.shape[1]*4, th_tes.shape[0]*4))
      bsize = 5
      th_tes = cv2.copyMakeBorder(th_tes, top=bsize, bottom=bsize, left=bsize, right=bsize, borderType=cv2.BORDER_CONSTANT, value=255)
      self._waypoint(th_tes, 'tes_{:02d}'.format(i))
      item, conf = self._run_tesseract(th_tes)
      return item, conf, drop_color

    def _get_drop_from_img(self, img):
        return [self.items[random.randrange(len(self.items))] for i in range(random.randrange(1, 5))]
    
    def _run_tesseract(self, image):
      res = pytesseract.image_to_data(image, config=self.config_tesseract+' --psm 7')
      #print('res: {}'.format(res))
      #conf, text = res.splitlines()[-1].split()[-2:]
      conf, text = [], []
      for l in res.splitlines()[1:]:
        #print(l.split())
        c, t = l.split()[-2:]
        #print(c, t)
        if t != '-1' and float(c) > 0:
          conf.append(float(c))
          text.append(t)
      #print([l.split()[-2:] for l in res.splitlines()])
      #print(zip([l.split()[-2:] for l in res.splitlines()]))
      #conf, text = zip([l.split()[-2:] for l in res.splitlines()])
      print('tesseract: conf = {} text = {}'.format(conf, text))
      return ' '.join(text), sum(conf)/len(conf)


dropper = Drop()

#def py_droprec(img, timestamp=0, y0=20, y1=570, x0=None, x1=None, signal=[]):
def py_droprec(img, timestamp=0, signal=[], y0=None, y1=None, x0=None, x1=None):
  print('SZ droprec timestamp = {}, crop = [{}:{}, {}:{}], signal = {}'.format(timestamp, y0, y1, x0, x1, signal))
  signal_copy = signal.copy()
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
  s = '\n'.join(['({}){}[{}]'.format(d[2], d[0], d[1]) for d in drop])
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
        return basePath
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)

GHP = 'Greater Healing Potion'
GMP = 'Greater Mana Potion'
SHP = 'Super Healing Potion'
SMP = 'Super Mana Potion'
FRP = 'Full Rejuvenation Potion'
RP = 'Rejuvenation Potion'
def get_img(img_name, signal=None):
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
  return img, timestamp, signal

if __name__ == '__main__':
  np.set_printoptions(threshold=np.inf, linewidth=np.inf)
  #dropper.cuts = [-7.16129, -7.0, -10.4838705, -38.83871, -38.451614, -36.322582, -5.096774, -4.1827955, -5.419355, -21.655914, -20.903225, -22.838709, -39.5, -30.375, -22.625, -42.0, -33.625, -35.5, 44.62088, -19.088888]
  #dropper.cuts_x1 = [-9.1875, -12.9375, -12.875, -32.25, -37.25, -44.0625, -16.022223]
  dropper.cuts = [-7.16129, -7.0, -10.4838705, -38.83871, -38.451614, -36.322582, -7.2645164, -6.0903225, -8.858065, -21.941935, -22.754839, -26.393549, -39.5, -30.375, -22.625, -42.0, -33.625, -35.5, 41.008244, -19.088888]
  dropper.cuts_x1 = [-15.625, -16.5625, -13.1875, -32.25, -37.25, -44.0625, -16.022223]
  dropper.flag_train = 1
  if len(sys.argv) == 1:
    #ret = py_droprec(cv2.imread('../../502/screens_1/95621693075.png'), y0=300, y1=345, x0=275, x1=465, signal=[[4+300, 10+275], [22+300, 62+275], [1+300, 1+275]])
    #ret = py_droprec(cv2.imread('../158458820669.png'), signal=[[145,245,361,'Martel De Fer','y'],[145,363,553,GHP,'w'],[145,555,745,GHP,'w'],[160,221,319,'Tusk Sword','g'],[160,321,526,FRP,'w'],[175,232,408,GMP,'w'],[190,222,351,'Studded Leather','b'],[205,226,381,SMP,'w'],[220,249,425,GMP,'w'],[235,176,366,GHP,'w'],[251,270,403,'Conquest Sword','y'],[268,207,397,GHP,'w']])
    # trained
    ret = py_droprec(*get_img('../../502/screens_1/94456900671.png', signal=[[249,328,419,'Bone Wand','y'],[249,421,576,SMP,'w'],[249,578,783,FRP,'w'],[264,268,407,'Flawed Amethyst','w'],[264,409,574,RP,'w'],[279,307,476,SHP,'w'],[294,312,400,'Bone Shield','b'],[309,289,458,SHP,'w'],[324,326,491,RP,'w'],[339,244,434,GHP,'w'],[357,386,432,'Jewel','y'],[376,296,451,SMP,'w']]))
    #
    ret = py_droprec(*get_img('../../502/screens_1/95217013080.png'))
    print(ret)
  else:
    for img_name in list_images(sys.argv[1]):
      print(img_name)
      img = cv2.imread(img_name)
      timestamp = os.path.splitext(os.path.basename(img_name))[1]
      ret = py_droprec(*get_img(img_name))
      print(ret)
  dropper.stats()
