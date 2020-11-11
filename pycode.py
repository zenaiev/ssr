import pytesseract
import numpy as np
import imutils
import cv2
import random
from imutils.object_detection import non_max_suppression
import time
import math
import pytesseract
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        ys = self._detect_y(img_gray, signal)
        print(ys)
        ls, rs = self._detect_x(img_gray, ys, signal)
        print(ls, rs)
        drop = []
        i = 0
        for y,l,r in zip(ys, ls, rs):
          #drop.append(self._process_box(img_gray[y:y+self.box_height, l:r+1], img[y:y+self.box_height, l:r+1], i))
          #print('box {:2d}: y = {} l = {} r = {} color = {} conf = {} item = {}'.format(i, y, l, r, drop[-1][2], drop[-1][1], drop[-1][0]))
          i += 1
        for y,l,r in zip(ys, ls, rs):
          cv2.rectangle(img, (l-1, y-1), (r+1, y+self.box_height), (0, 255, 0), 1)
        print('{}: {}'.format(len(drop), drop))
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

    def _detect_y(self, img, start_y = 0, signal = [222, 237, 252, 267, 282, 300], signal_x = {}):
      npix_sum = 25
      min_mean = 20
      img_dy_t = cv2.filter2D(img[:-self.box_height, :], -1, np.matrix('1; -1'))
      img_dy_b = cv2.filter2D(img[self.box_height:, :], -1, np.matrix('-1; 1'))
      img_dy_tb = img_dy_t + img_dy_b
      img_dy_tb_mean = cv2.filter2D(img_dy_tb, -1, np.matrix('1 '*npix_sum)) / (2*npix_sum)
      assert npix_sum % 2 == 1
      dx = int((npix_sum - 1) / 2)
      # set to zero top of window
      self._set_to_zero(img_dy_tb_mean, y1=25)
      # set to zero x margins
      self._set_to_zero(img_dy_tb_mean, x1=dx)
      self._set_to_zero(img_dy_tb_mean, x0=-dx)
      # set to zero merc
      self._set_to_zero(img_dy_tb_mean, y0=0, y1=75, x0=18, x1=65)
      # plugy msg
      self._set_to_zero(img_dy_tb_mean, y0=104, y1=134, x0=14, x1=293)
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
        fig.subplots_adjust(hspace = 0.0, left = 0.06, right = 0.97)
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
      cv2.imwrite('img/{}_{}.png'.format(self.timestamp, label), img)
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

def py_droprec(img, timestamp=0, y0=0, y1=570, x0=None, x1=None, signal=[]):
  dropper.img_orig = img
  dropper.timestamp = timestamp
  print('SZ droprec timestamp = {}, crop = [{}:{}, {}:{}], signal = {}'.format(timestamp, y0, y1, x0, x1, signal))
  img_cropped = img[y0:y1, x0:x1]
  drop = dropper.process_drop(img_cropped, signal)
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

if __name__ == '__main__':
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
  # remove 141, 142
  if len(sys.argv) == 1:
    #img = cv2.imread(fin)
    #img = np.int8(img)
    #img = np.int8(img)
    #img = img[140:169, 244:362] # one box extended
    #img = img[start_y:550, :] # cut only small top and bottom
    #img = img[320:340, 335:407]
    #img = img[300:340, 335:407]
    #img = img[300:340, 315:427]
    #img = img[300:340, 275:467] # two items
    #signal_x = {4: [9, 187], 22: [61, 132]}
    #ret = py_droprec(cv2.imread('../../502/screens/95621693075.png'), y0=300, y1=340, x0=275, x1=467)
    ret = py_droprec(cv2.imread('../../502/screens/95621693075.png'))
    print(ret)
  else:
    for img_name in list_images(sys.argv[1]):
      print(img_name)
      img = cv2.imread(img_name)
      ret = py_droprec(img)
      print(ret)
