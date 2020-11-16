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
      #self.box_margin_lr = 4
      #self.box_margin_b = 4
      self.basedir = '/home/zenaiev/games/Diablo2/ssr/'
      self.config_tesseract = "-c tessedit_char_whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -'"
      with open(self.basedir + 'd2-drop-items.txt') as f:
          self.items = [l[:-1] for l in f.readlines()]
      self.current_drop = []
      self.sig = []
      self.bkg = []
      self.sig_x1 = []
      self.bkg_x1 = []
      self.cuts = []
      self.cuts_x1 = []
      self.flag_train = False
      self.vars_title = []
      self.vars_title_x1 = []
      self.xy_sig = []
      self.xy_bkg = []
      self.xy_sig_t_pix = []
      self.xy_bkg_t_pix = []
      self.xy_sig_t_pix_rat = []
      self.xy_bkg_t_pix_rat = []
      self.xy_sig_t_pix_arb = []
      self.xy_bkg_t_pix_arb = []
      self.xy_bkg_t_30 = []
      self.xy_sig_t_30 = []
      self.xy_bkg_t_quad_30 = []
      self.xy_sig_t_quad_30 = []
      self.xy_bkg_t_abs_30 = []
      self.xy_sig_t_abs_30 = []
      self.xy_bkg_t_quad_30_20 = []
      self.xy_sig_t_quad_30_20 = []
      self.xy_bkg_t_abs_30_20 = []
      self.xy_sig_t_abs_30_20 = []
      self.use_signal_x1 = False
    
    def stats(self):
      if len(self.sig):
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
            else:
              fig = plt.figure(title[i])
              ax = None
            bkg_sample = random.sample(bkg, min(10000, len(bkg)))
            cuts.append(self._plot(title[i], [v[i] for v in sig], [v[i] for v in bkg_sample], np.linspace(0, 300, 100), [1.0, 0.90], ax, col=['b','g','r'][i%3]))
        print('cuts = {}'.format(cuts_xy))
        print('cuts_x1 = {}'.format(cuts_x1))
        '''for i in range(len(self.sig_x1[0])):
          tit = self.vars_title[i+len(self.sig[0])]
          if len(self.vars_title[i].split('[')) > 1:
            if i%3 == 0:
              fig, axes = plt.subplots(3, 1, sharex=True, squeeze=True)
              fig.suptitle(tit.split('[')[0])
              fig.canvas.set_window_title(tit.split('[')[0])
              plt.subplots_adjust(hspace = 0.0, left = 0.01, right = 0.99)
            ax = axes[i%3]
          else:
            fig = plt.figure(tit)
            ax = None
          bkg = random.sample(self.bkg_x1, min(10000, len(self.bkg_x1)))
          cuts.append(self._plot(tit, [v[i] for v in self.sig_x1], [v[i] for v in bkg], np.linspace(-200, 50, 100), [1.0, 0.90], ax, col=['b','g','r'][i%3]))'''
#        print('cuts = {}'.format(cuts))
        plt.show(block=False)
        cv2.waitKey()
      return
      if len(self.xy_bkg) == 0 and len(self.xy_sig) == 0:
        return
      ncuts = len(self.xy_bkg[0]) if len(self.xy_bkg) != 0 else len(self.xy_sig[0])
      print('ncuts = {}'.format(ncuts))
      #print('bkg:\n{}'.format(dropper.xy_bkg))
      bkg_aver = [0] * ncuts
      sig_aver = [0] * ncuts
      sig_extr = [0] * ncuts
      for i in range(ncuts):
        bkg_aver[i] = sum(x[i] for x in dropper.xy_bkg) / len(dropper.xy_bkg)
        sig_aver[i] = sum(x[i] for x in dropper.xy_sig) / len(dropper.xy_sig)
        sig_extr[i] = min(x[i] for x in dropper.xy_sig)
      #print([all(x[i] > sig_extr[i] for i in range(ncuts)) for x in dropper.xy_bkg])
      cuts = sig_extr
      #cuts[0] = 25
      #cuts[1] = 25
      #cuts = [20, 20, 35, 220, 20]
      fp = sum(all(x[i] >= cuts[i] for i in range(ncuts)) for x in dropper.xy_bkg)
      tp = sum(all(x[i] >= cuts[i] for i in range(ncuts)) for x in dropper.xy_sig)
      fn = sum(any(x[i] < cuts[i] for i in range(ncuts)) for x in dropper.xy_sig)
      tn = sum(any(x[i] < cuts[i] for i in range(ncuts)) for x in dropper.xy_bkg)
      #print('sig:\n{}'.format(dropper.xy_sig))
      print('sig')
      str_sig = '{:8.1f}'*ncuts
      print(str_sig)
      for s in dropper.xy_sig:
        print(str_sig.format(*s))
      print('average bkg: {}'.format(bkg_aver))
      print('average sig: {}'.format(sig_aver))
      print('extreme sig: {}'.format(sig_extr))
      print('FP, FN, TP, TN: {}, {}, {}, {}'.format(fp, fn, tp, tn))
      print('cuts = {}'.format(cuts))
      # t pix
      #print(self.xy_sig_t_pix)
      self._plot('Top edge gradient pix', self.xy_sig_t_pix, self.xy_bkg_t_pix, np.linspace(-120, 120, 100), 0.99)
      self._plot('Top edge gradient pix ratio', self.xy_sig_t_pix_rat, self.xy_bkg_t_pix_rat, np.linspace(0, 5, 100), 0.99)
      self._plot('Top edge gradient pix A-r*B', self.xy_sig_t_pix_arb, self.xy_bkg_t_pix_arb, np.linspace(-150, 100, 100), 0.99)
      self._plot('Top edge sum30', self.xy_sig_t_30, self.xy_bkg_t_30, np.linspace(-100, 150, 100))
      self._plot('Top edge quad sum30', self.xy_sig_t_quad_30, self.xy_bkg_t_quad_30, np.linspace(0, 250, 100))
      self._plot('Top edge abs sum30', self.xy_sig_t_abs_30, self.xy_bkg_t_abs_30, np.linspace(0, 250, 100))
      self._plot('Top edge quad sum30[20]', self.xy_sig_t_quad_30_20, self.xy_bkg_t_quad_30_20, np.linspace(0, 200, 100))
      self._plot('Top edge abs sum30[20]', self.xy_sig_t_abs_30_20, self.xy_bkg_t_abs_30_20, np.linspace(0, 200, 100))
      #self._plot('', self.xy_sig_t_pix, self.xy_bkg_t_pix, , 0.99)
      plt.show(block=False)
      #plt.draw()
      cv2.waitKey()
      #input("Press Enter to continue...")

    def _plot(self, label, sig, bkg, bins=None, qs=[0.90], ax=None, col=None):
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
      #print('{}: sig mean[min,max] {.0f}[{}{}] min {}; bkg mean {}; eff q = {} c = {} sig {} bkg {}'.format(label, sum(sig)/len(sig), min(sig), sum(bkg)/len(bkg), q, c, sig_eff, bkg_eff))
      if ax is None:
        #ax = plt
        ax = plt.gca()
      #ax.figure(label)
      bins = np.linspace(min(sig_min, bkg_min), max(sig_max, bkg_max), 100)
      if bins is not None:
        ax.hist(bkg, bins, alpha=0.5, density=True, label='bkg({}) {:.0f}[{:.0f},{:.0f}]'.format(len(bkg), bkg_mean,bkg_min,bkg_max), color='k')
      else:
        _,bins,_ = ax.hist(bkg, alpha=0.5, density=True, label='bkg({}) {:.0f}[{:.0f},{:.0f}]'.format(len(bkg), bkg_mean,bkg_min,bkg_max), color=col)
      ax.hist(sig, bins, alpha=0.5, density=True, label='sig({}) {:.0f}[{:.0f},{:.0f}]'.format(len(sig), sig_mean,sig_min,sig_max), color=col)
      ax.legend(loc=2)
      ax.text(0.0,0.45,'\n'.join(q_str),horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
      ax.axes.get_yaxis().set_visible(False)
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

    def _convert(self, img):
      print(img.dtype)
      info = np.iinfo(img.dtype) # Get the information of the incoming image type
      img = img.astype(np.float64) / info.max # normalize the data to 0 - 1
      img = 255 * img # Now scale by 255
      img = img.astype(np.uint8)

    def process_drop(self, img, signal=[]):
        self._waypoint(self.img_orig, 'orig')
        self._waypoint(img, 'input')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_gray = np.int16(img_gray)
        img_gray = np.float32(img_gray)
        #ys, ls = self._detect_boxes(img_gray, signal=signal)
        img_float = np.float32(img)
        ys, ls, rs = self._boxes(img_float, signal=signal)
        #a
        ys = [y+1 for y in ys]
        ls = [y+1 for y in ls]
        #rs = [x+32 for x in ls]
        rs = [x for x in rs]
        #return []
        #ys = self._detect_y_new(img_gray, signal)
        #ys = self._detect_y(img_gray, signal)
        print(ys, ls, rs)
        #ls, rs = self._detect_x(img_gray, ys, signal)
        #print(ls, rs)
        drop = []
        i = 0
        for y,l,r in zip(ys, ls, rs):
          #drop.append(self._process_box(img_gray[y:y+self.box_height, l:r+1], img[y:y+self.box_height, l:r+1], i))
          #print('box {:2d}: y = {} l = {} r = {} color = {} conf = {} item = {}'.format(i, y, l, r, drop[-1][2], drop[-1][1], drop[-1][0]))
          i += 1
        for y,l,r in zip(ys, ls, rs):
          cv2.rectangle(img, (l-1, y-1), (r+1, y+self.box_height), (0, 255, 0), 1)
        print('drop {}: {}'.format(len(drop), drop))
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
      assert img.shape[2] == 3
      box_x = 31
      n_box_x = 7
      #n_box_x = 1
      box_y = 16
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
      #print(';'.join(['1 '*(mar_r-1)]*(box_y-1)))
      #marrsum = cv2.filter2D(dxya, -1, np.matrix(';'.join(['1 '*(mar_r-1)]*(box_y-1))), anchor=(0,2))/(2*(mar_r-1)*(box_y-1))
      #marrsum = cv2.filter2D(dxya, -1, np.matrix(';'.join(['1 '*(mar_r-1)]*(box_y-1))), anchor=(1,0))/(2*(mar_r-1)*(box_y-1))
      marrsum = cv2.filter2D(dxya, -1, np.matrix(';'.join(['1 '*(mar_r-1)]*(box_y-1))), anchor=(0,0))/(2*(mar_r-1)*(box_y-1))
      num_rows, num_cols = marrsum.shape[:2]
      translation_matrix = np.float32([ [1,0,mar_r-1], [0,1,0] ])
      marrsum = cv2.warpAffine(marrsum, translation_matrix, (num_cols,num_rows))
      marrsum = maxrgb(marrsum)
      #print('marrsum: {}'.format(marrsum[247:252, 417:422]))
      #print('marrsum: {}'.format(marrsum[249, 419]))
      #a
      
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
                #print(self._calc_adxdysum(img, 419, 420, 249, 254))
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
              print('x,y {} {}'.format(x, y))
              for x1 in range(x+box_x, sig.shape[1]):
                #print('x1 {} vars {} cut {} -> {}'.format(x1, img_vars_x1[y,x1], self.cuts_x1, all(v >= (c-0.001) for v,c in zip(img_vars_x1[y, x1], self.cuts_x1))))
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

    def _detect_boxes(self, img, signal=[]):
      #print('img.shape: {}', img.shape)
      len_x = 30
      len_y = self.box_height
      start_x = 4
      grady = cv2.filter2D(img, -1, np.matrix('1; -1'), anchor=(0,0))[:-1, :]
      grady_t = grady[:-len_y, :]
      grady_b = -1 * grady[len_y:, :]
      grady_t_slice = cv2.filter2D(grady_t, -1, np.matrix('1 '*len_x), anchor=(0,0))[:, :-len_x-1] / len_x
      grady_b_slice = cv2.filter2D(grady_b, -1, np.matrix('1 '*len_x), anchor=(0,0))[:, :-len_x-1] / len_x
      gradx = cv2.filter2D(img, -1, np.matrix('1 -1'), anchor=(0,0))[:, :-1]
      gradx_l_slice = cv2.filter2D(gradx, -1, np.matrix(';'.join(['1']*len_y)), anchor=(0,0))[:-len_y-1, :-len_x] / len_y
      gradx_abs = np.absolute(gradx)
      gradx_abs_start = cv2.filter2D(gradx_abs, -1, np.matrix(';'.join(['1 '*start_x]*len_y)), anchor=(0,0))[:-len_y-1, 1:-len_x+1] / len_y / start_x
      gradx_abs_start = 255 - gradx_abs_start
      gradx_abs_text = cv2.filter2D(gradx_abs, -1, np.matrix(';'.join(['1 '*(len_x-start_x)]*len_y)), anchor=(0,0))[:-len_y-1, 1+start_x:-len_x+start_x+1] / len_y / (len_x-start_x)
      l_minus1 = cv2.filter2D(img, -1, np.matrix('1;1;1;1;-1;-1;-1;-1;-1;-1;-1;-1;1;1;1;1'), anchor=(0,0))[:-17, :-len_x-1] / 16.
      #print('img.shape {} vars {} {} {} {} {}'.format(img.shape, grady_t_slice.shape, grady_b_slice.shape, gradx_l_slice.shape, gradx_abs_start.shape, gradx_abs_text.shape))
      ys, xs = [], []
      if len(self.cuts) != 0:
        sig = np.zeros(grady_t_slice.shape)
        for x in range(grady_t_slice.shape[1]):
          for y in range(grady_t_slice.shape[0]):
            sum3 = grady_t_slice[y, x] + grady_b_slice[y, x] + gradx_l_slice[y, x]
            sig[y, x] = self._cuts(grady_t_slice[y, x], grady_b_slice[y, x], gradx_l_slice[y, x], sum3, gradx_abs_start[y, x], gradx_abs_text[y, x], l_minus1[y, x])
            if sig[y, x]:
              ys.append(y)
              xs.append(x)
        return ys, xs
      elif len(signal) != 0:
        sig = np.zeros(grady_t_slice.shape)
        sig_pix = np.zeros(grady_t_slice.shape)
        for s in signal:
          y, x = s[0]-1, s[1]-1
          sum3 = grady_t_slice[y, x] + grady_b_slice[y, x] + gradx_l_slice[y, x]
          print('signal y,x {} {}: {}'.format(y, x, (grady_t_slice[y, x], grady_b_slice[y, x], gradx_l_slice[y, x], sum3, gradx_abs_start[y, x], gradx_abs_text[y, x], l_minus1[y, x])))
          sig[y, x] = 1
          self.xy_sig.append((grady_t_slice[y, x], grady_b_slice[y, x], gradx_l_slice[y, x], sum3, gradx_abs_start[y, x], gradx_abs_text[y, x], l_minus1[y, x]))
          ys.append(y)
          xs.append(x)
          x1 = s[2]+1
          self.xy_sig_t_pix += [img[y, xx]-img[y+1, xx] for xx in range(x+1, x1)]
          self.xy_sig_t_pix_rat += [img[y, xx]/img[y+1, xx] for xx in range(x+1, x1) if img[y+1, xx] != 0]
          self.xy_sig_t_pix_arb += [img[y, xx]-2*img[y+1, xx] for xx in range(x+1, x1)]
          self.xy_sig_t_30.append(sum(img[y, xx]-img[y+1, xx] for xx in range(x+1, x+1+30))/30)
          self.xy_sig_t_quad_30.append(math.sqrt(sum((img[y, xx]-2*img[y+1, xx])**2 for xx in range(x+1, x+1+30))/30))
          self.xy_sig_t_abs_30.append(sum(abs(img[y, xx]-2*img[y+1, xx]) for xx in range(x+1, x+1+30))/30)
          diff = [img[y, xx]-2*img[y+1, xx] for xx in range(x+1, x+1+30)]
          diff.sort()
          self.xy_sig_t_quad_30_20.append(math.sqrt(sum(xx**2 for xx in diff[5:25])/20))
          self.xy_sig_t_abs_30_20.append(sum(abs(v) for v in diff[5:25])/20)
          print(self.xy_sig_t_quad_30[-1], self.xy_sig_t_quad_30_20[-1], [(img[y, xx]-2*img[y+1, xx]) for xx in range(x+1, x+1+30)])
          sig_pix[y, x+1:x1] = 1
        for x in range(grady_t_slice.shape[1]):
          for y in range(grady_t_slice.shape[0]):
            if sig[y, x] == 0:
              sum3 = grady_t_slice[y, x] + grady_b_slice[y, x] + gradx_l_slice[y, x]
              self.xy_bkg.append((grady_t_slice[y, x], grady_b_slice[y, x], gradx_l_slice[y, x], sum3, gradx_abs_start[y, x], gradx_abs_text[y, x], l_minus1[y, x]))
            if sig_pix[y, x] == 0:
              self.xy_bkg_t_pix += [img[y, x]-img[y+1, x]]
              if img[y+1, x] != 0: self.xy_bkg_t_pix_rat += [img[y, x]/img[y+1, x]]
              self.xy_bkg_t_pix_arb += [img[y, x]-2*img[y+1, x]]
              self.xy_bkg_t_30.append(sum(img[y, xx]-img[y+1, xx] for xx in range(x+1, x+1+30))/30)
              self.xy_bkg_t_quad_30.append(math.sqrt(sum((img[y, xx]-2*img[y+1, xx])**2 for xx in range(x+1, x+1+30))/30))
              self.xy_bkg_t_abs_30.append(sum(abs(img[y, xx]-2*img[y+1, xx]) for xx in range(x+1, x+1+30))/30)
              diff = [img[y, xx]-2*img[y+1, xx] for xx in range(x+1, x+1+30)]
              diff.sort()
              self.xy_bkg_t_quad_30_20.append(math.sqrt(sum(xx**2 for xx in diff[5:25])/20))
              self.xy_bkg_t_abs_30_20.append(sum(abs(v) for v in diff[5:25])/20)
      return ys, xs
      #a

    def _cuts(self, t, b, l, sum3, start, text, lm1):
      if t >= self.cuts[0] and b >= self.cuts[1] and l >= self.cuts[2] and sum3 >= self.cuts[3] and start >= self.cuts[4] and text >= self.cuts[5] and lm1 >= self.cuts[6]:
        return True
      else:
        return False

    def _detect_y_new(self, img, signal=[]):
      slice_x = 30
      box_y = self.box_height
      print('img.shape: {}', img.shape)
      #'1 '*slice_x + ';' + ';'.join('-1')
      grady = cv2.filter2D(img, -1, np.matrix('1; -1'), anchor=(0,0)) / 2
      grady_abs = np.absolute(grady)
      print('grady.shape: {}', grady.shape)
      grady_p = grady[:-self.box_height, :]
      grady_m = -1 * grady[self.box_height:, :]
      print('grady_p.shape: {}', grady_p.shape)
      y_sum_p = np.sum(grady_p, axis=1)
      y_sum_m = np.sum(grady_m, axis=1)
      y_slice_p = [0] * len(y_sum_p)
      y_slice_m = [0] * len(y_sum_m)
      grady_p_slicex = cv2.filter2D(grady_p, -1, np.matrix('1 '*slice_x), anchor=(0,0)) / slice_x
      grady_m_slicex = cv2.filter2D(grady_m, -1, np.matrix('1 '*slice_x), anchor=(0,0)) / slice_x
      gradx = cv2.filter2D(img, -1, np.matrix('1 -1'), anchor=(0,0)) / 2
      print('gradx.shape: {}', gradx.shape)
      gradx_slicey = cv2.filter2D(gradx, -1, np.matrix(';'.join(['1']*box_y)), anchor=(0,0))[:-self.box_height] / box_y
      print('grady_p_slicex.shape: {}', grady_p_slicex.shape)
      print('grady_m_slicex.shape: {}', grady_m_slicex.shape)
      print('gradx_slicey.shape: {}', gradx_slicey.shape)
      grady_p_slicex[grady_p_slicex<0] = 0
      grady_m_slicex[grady_m_slicex<0] = 0
      gradx_slicey[gradx_slicey<0] = 0
      grady_p_slicex = np.uint8(grady_p_slicex)
      grady_m_slicex = np.uint8(grady_m_slicex)
      gradx_slicey = np.uint8(gradx_slicey)
      #print(grady_p_slicex[:, :14], grady_m_slicex[:, :14], gradx_slicey[:, :14])
      img3 = cv2.merge((grady_p_slicex[:-1],grady_m_slicex[:-1],gradx_slicey[1:]))
      for s in signal:
        print('signal y,x0,x1 {} {}: {} {} {}'.format(s['y'], s['x0'], grady_p_slicex[s['y']-1, s['x0']-1], grady_m_slicex[s['y']-1, s['x0']-1], gradx_slicey[s['y']-1, s['x0']-1]))
      self._waypoint(img3, '3')
      mask = cv2.inRange(img3, np.array([7, 7, 7]), np.array([255, 255, 255]))
      print(mask)
      a
      gradx_abs = np.absolute(gradx)
      grady_abs_slicex = cv2.filter2D(gradx_abs, -1, np.matrix('1 '*slice_x), anchor=(0,0)) / slice_x
      gradx_abs_box = cv2.filter2D(gradx_abs, -1, np.matrix(';'.join(['1 '*slice_x]*box_y)), anchor=(0,0)) / slice_x / box_y
      grady_abs_box = cv2.filter2D(grady_abs, -1, np.matrix(';'.join(['1 '*slice_x]*box_y)), anchor=(0,0)) / slice_x / box_y
      print('gradx_abs_box.shape: {}', gradx_abs_box.shape)
      for y in range(grady_p.shape[0]):
        if y_sum_p[y] < 0 or y_sum_m[y] < 0: continue
        plt.plot(grady_p_slicex[y, :], label='grady_p_slicex')
        plt.plot(grady_m_slicex[y, :], label='grady_m_slicex')
        plt.plot(gradx_abs_box[y, :], label='gradx_abs_box')
        plt.plot(grady_abs_box[y, :], label='grady_abs_box')
        plt.plot(grady_abs_slicex[y, :], label='grady_abs_slicex')
        plt.legend()
        plt.show()
        #cv2.waitKey()
        slice_p, x0_p, x1_p = self._max_slice(grady_p[y, :], 5, -5)
        slice_m, x0_m, x1_m = self._max_slice(grady_m[y, :], 5, -5)
        print(y, slice_p, x0_p, x1_p, slice_m, x0_m, x1_m)
        if slice_p > slice_m:
          slice_m = sum(grady_m[y, x0_p:x1_p+1])
          x0, x1 = x0_p, x1_p
        else:
          slice_p = sum(grady_p[y, x0_m:x1_m+1])
          x0, x1 = x0_m, x1_m
        y_slice_p[y] = slice_p
        y_slice_m[y] = slice_m
      #grady_p_slice = cv2.filter2D(grady_p, -1, np.matrix('1 '*30))
      #print('grady_p_slice.shape: {}', grady_p_slice.shape)
      #grady_m_slice = cv2.filter2D(grady_m, -1, np.matrix('1 '*30))
      #self._set_to_zero(grady_p, y0=grady_p.shape[0]-1)
      grady_p_abs = np.absolute(grady_p)
      grady_m_abs = np.absolute(grady_m)
      fig, axes = plt.subplots(1, 1, sharex=True, squeeze=True)
      ax = [axes]
      ax.append(plt)
      plt.subplots_adjust(hspace = 0.0, left = 0.06, right = 0.97)
      #print(ax)
      ax[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
      #ax[0].plot(range(img.shape[1], np.sum(grady[:-self.box_height], axis=1).tolist()))
      #ax[0].plot(range(img.shape[1], np.sum(grady[self.box_height:], axis=1).tolist()))
      #ax[0].plot(np.sum(grady_p[:-self.box_height], axis=1), label='grady_p')
      #ax[0].plot(np.sum(grady_m[self.box_height:], axis=1), label='grady_m')
      ax[0].plot(y_slice_p, label='y_slice_p')
      ax[0].plot(y_slice_m, label='y_slice_m')
      #ax[0].plot(np.sum(grady_p_abs[:-self.box_height], axis=1), label='grady_abs_p')
      #ax[0].plot(np.sum(grady_m_abs[self.box_height:], axis=1), label='grady_abs_m')
      ax[0].legend()
      print(signal)
      for s in signal:
        ax[0].vlines(s['y'], *ax[0].get_ylim(), colors='r')
      #ax[0].set_xlim(100, 200)
      ax[0].set_ylim(0, None)
      plt.show()
      #cv2.waitKey()
      a
      return []

    def _max_slice(self, A0, x_min=None, x_max=None):
      #print(len(A0), A0, x_min, x_max)
      if x_min is None:
        x_min = 0
      if x_max is None:
        x_max = len(A0)
      A = A0[x_min:x_max]
      max = A[0]
      acc = 0
      x0, x1 = 0, 0
      for ie,e in enumerate(A):
        acc += e
        if acc > max:
          max = acc
          x1 = ie
        elif acc < 0:
          acc = 0
          x0 = ie + 1
      if x0 == len(A):
        x0 -= 1
      return (max, x0+x_min, x1+x_min)

    def _detect_y(self, img, signal=[]):
      npix_sum = 25
      min_mean = 20
      img_dy_t = cv2.filter2D(img[:-self.box_height, :], -1, np.matrix('1; -1'))
      img_dy_b = cv2.filter2D(img[self.box_height:, :], -1, np.matrix('-1; 1'))
      img_dy_tb = img_dy_t + img_dy_b
      img_dy_tb_mean = cv2.filter2D(img_dy_tb, -1, np.matrix('1 '*npix_sum)) / (2*npix_sum)
      assert npix_sum % 2 == 1
      dx = int((npix_sum - 1) / 2)
      if img.shape[0] > 500:
        # set to zero top of window
        self._set_to_zero(img_dy_tb_mean, y1=25)
        # set to zero merc
        self._set_to_zero(img_dy_tb_mean, y0=0, y1=75, x0=18, x1=65)
        # plugy msg
        self._set_to_zero(img_dy_tb_mean, y0=104, y1=134, x0=14, x1=293)
      if img.shape[0] > 700:
        # set to zero x margins
        self._set_to_zero(img_dy_tb_mean, x1=dx)
        self._set_to_zero(img_dy_tb_mean, x0=-dx)
      #img_dy_tb_mean[img_dy_tb_mean < min_value] = 0
      img_mask = np.zeros(img_dy_tb.shape, img.dtype)
      for y in range(img.shape[0]-16-1):
        for x in range(dx, img.shape[1]-dx):
          if img_dy_tb_mean[y, x] < min_mean:
            continue
          mean_tb = img_dy_tb_mean[y, x]
          rms_tb = sum([(x-mean_tb)**2 for x in img_dy_t[y, x-dx:x+dx+1]] + [(x-mean_tb)**2 for x in img_dy_b[y, x-dx:x+dx+1]])
          rms_tb = math.sqrt(rms_tb / (npix_sum*2))
          if mean_tb > rms_tb:
            img_mask[y, x] = 1
        #img_mean[y, 0] = max(img_mean[y, :])
      img_mask = cv2.erode(img_mask, np.matrix('1 '*8))
      ys = set(np.where(img_mask == 1)[0])
      # remove entering plateau msg
      if 164 in ys and 165 in ys:
        ys.remove(141)
        ys.remove(142)
      return ys
    
    def _detect_x(self, img, ys, signal):
      ls, rs = [], []
      xmargin = 6
      for y in ys:
        print('y = {}'.format(y))
        l, r = [0]*img.shape[1], [0]*img.shape[1]
        l_rms, r_rms = [0]*img.shape[1], [0]*img.shape[1]
        l_rat, r_rat = [0]*img.shape[1], [0]*img.shape[1]
        for x in range(xmargin, img.shape[1]-xmargin):
          l[x], l_rms[x] = self._calc_h_edge(img, y, x, 'l')
          r[x], r_rms[x] = self._calc_h_edge(img, y, x, 'r')
          if l_rms[x] != 0:
            l_rat[x] = l[x] / l_rms[x] * 10
          if r_rms[x] != 0:
            r_rat[x] = r[x] / r_rms[x] * 10
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
        plt.subplots_adjust(hspace = 0.0, left = 0.06, right = 0.97)
        ax1.plot(range(img.shape[1]), l)
        ax1.plot(range(img.shape[1]), l_rms)
        ax1.plot(range(img.shape[1]), l_rat)
        ax2.plot(range(img.shape[1]), r)
        ax2.plot(range(img.shape[1]), r_rms)
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax2.plot(range(img.shape[1]), r_rat)
        for s in signal:
          ax1.vlines(s['l'], *ax1.get_ylim(), colors='b')
          ax2.vlines(s['b'], *ax2.get_ylim(), colors='b')
        #plt.show()
        #cv2.waitKey()
        ls.append(l_rat.index(max(l_rat)))
        rs.append(r_rat.index(max(r_rat)))
        print('y = {} l = {} r = {}'.format(y, ls[-1], rs[-1]))
      return ls, rs

    def _calc_h_edge(self, img, y, x, mode):
      if mode == 'l':
        xp, xm = x-1, x
      elif mode == 'r':
        xp, xm = x+1, x
      diff = [p - m for p,m in zip(img[y:y+self.box_height, xp].tolist(), img[y:y+self.box_height, xm].tolist())]
      #print(diff)
      mean = sum(diff) / len(diff)
      rms = math.sqrt(sum((d-mean)**2 for d in diff) / len(diff))
      #xp_list = img[y+1:y+1+self.box_height, xp].tolist()
      #mean_p = sum(xp_list) / len(xp_list)
      #rms_p = math.sqrt(sum((d-mean_p)**2 for d in xp_list) / len(xp_list))
      return mean, rms#, rms_p

    def _set_y_to_zero(self, img, y, start_y, x, min_val):
      if img.shape[0] < y:
        return
      if len(np.where(img[y-start_y, :x] == min_val)[0]) > 0 and len(np.where(img[y-start_y, x:] == min_val)[0]) == 0:
        #print('removing')
        img[y-start_y, :] = 0
    
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
def py_droprec(img, timestamp=0, y0=None, y1=None, x0=None, x1=None, signal=[]):
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

def get_img(img_name):
  img = cv2.imread(img_name)
  timestamp = os.path.splitext(os.path.basename(img_name))[0]
  return img, timestamp

if __name__ == '__main__':
  '''print(dropper._max_slice([-2, 1, 2, 5, 1, -1, 3, 30, -5, -25, 35, -45]))
  print(dropper._max_slice([-1, 2, -5, 12, -21, 33, 30, -15, 25, 35, -10, 11, -12]))
  print(dropper._max_slice([-1, 2, -5, 12, -21, 33, 30, -15, 25, 35, -10, 11, -12, 45]))
  print(dropper._max_slice([-1, 2, -5, 12, -21, 33, 30, -15, 25, 35, -10, 11, -12, 45], 7, 10))
  print(dropper._max_slice([-1]))
  print(dropper._max_slice([1]))
  print(dropper._max_slice([0]))
  a'''
  np.set_printoptions(threshold=np.inf, linewidth=np.inf)
  #x0, x1, y0, y1 = None, None, None, None
  # nominal top and bottom
  #y0, y1 = 23, 550
  # crop_img = img[y:y+h, x:x+w]
  #start_y=23
  #fin = '/home/zenaiev/games/Diablo2/ssr/158458820669.png'
  #fin = '../../502/screens/95621693075.png'
  #fin = '../../502/screens/97525956388.png'
  #fin = '../../502/screens/95842955271.png'
  #fin = '../../502/screens/95509173077.png'
  '../../502/screens/95127533078.png'
  '../../502/screens/98362279672.png'
  '../../502/screens/95982595271.png'
  '../../502/screens/98606719675.png'
  '../../502/screens/95160973083.png'
  '../../502/screens/96366795271.png'
  '../../502/screens/98160479669.png'
  '../../502/screens/95637733084.png'
  '../../502/screens/95551813087.png'
  '../../502/screens/98216559704.png'
  '../../502/screens/97052075936.png'
  '../../502/screens/97539036388.png'
  '../../502/screens/95855915280.png'
  '../../502/screens/98245719669.png'
  '../../502/screens/98750640837.png'
  '../../502/screens/95953875271.png'
  '../../502/screens/96914395937.png'
  '../../502/screens/96011475271.png'
  '../../502/screens/97078835945.png'
  dropper.cuts = []
  dropper.cuts = [-7.16129, -7.0, -10.4838705, -38.83871, -38.451614, -36.322582, -5.096774, -4.1827955, -5.419355, -21.655914, -20.903225, -22.838709, -39.5, -30.375, -22.625, -42.0, -33.625, -35.5, 44.62088, -19.088888]
  #dropper.cuts_x1 = [-9.1875, -12.9375, -12.875, -32.25, -37.25, -44.0625, -20.25]
  dropper.cuts_x1 = [-9.1875, -12.9375, -12.875, -32.25, -37.25, -44.0625, -16.022223]
  dropper.flag_train = 0
  if len(sys.argv) == 1:
    GHP = 'Greater Healing Potion'
    GMP = 'Greater Mana Potion'
    SHP = 'Super Healing Potion'
    SMP = 'Super Mana Potion'
    FRP = 'Full Rejuvenation Potion'
    RP = 'Rejuvenation Potion'
    #ret = py_droprec(cv2.imread('../../502/screens_1/95621693075.png'), y0=300, y1=345, x0=275, x1=465, signal=[[4+300, 10+275], [22+300, 62+275], [1+300, 1+275]])
    #ret = py_droprec(cv2.imread('../158458820669.png'), signal=[[145,245,361,'Martel De Fer','y'],[145,363,553,GHP,'w'],[145,555,745,GHP,'w'],[160,221,319,'Tusk Sword','g'],[160,321,526,FRP,'w'],[175,232,408,GMP,'w'],[190,222,351,'Studded Leather','b'],[205,226,381,SMP,'w'],[220,249,425,GMP,'w'],[235,176,366,GHP,'w'],[251,270,403,'Conquest Sword','y'],[268,207,397,GHP,'w']])
    ret = py_droprec(*get_img('../../502/screens_1/94456900671.png'), signal=[[249,328,419,'Bone Wand','y'],[249,421,576,SMP,'w'],[249,578,783,FRP,'w'],[264,268,407,'Flawed Amethyst','w'],[264,409,574,RP,'w'],[279,307,476,SHP,'w'],[294,312,400,'Bone Shield','b'],[309,289,458,SHP,'w'],[324,326,491,RP,'w'],[339,244,434,GHP,'w'],[357,386,432,'Jewel','y'],[376,296,451,SMP,'w']])
    print(ret)
  else:
    for img_name in list_images(sys.argv[1]):
      print(img_name)
      img = cv2.imread(img_name)
      timestamp = os.path.splitext(os.path.basename(img_name))[1]
      ret = py_droprec(*get_img(img_name))
      print(ret)
  dropper.stats()
