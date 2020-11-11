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

    def process_drop(self, img, start_y=0, signal_x={}):
        #img = self._preprocess_img(img)
        #self._preprocess_img(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = np.int16(img_gray)
        #img_gray = np.int8(img_gray)
       #img_gray = img_gray.astype('float')
        ret = self._edge_b_t(img_gray, start_y=start_y, signal_x=signal_x)
        ys, ls, rs, img_sum, img_rms, img_final = ret
        #img_rms = np.uint8(img_rms)
        #print(img_sum)
        #img_sum = cv2.normalize(img_sum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #delf._convert(img_sum)
        img_sum = np.uint8(img_sum.clip(min=0))
        img_rms = np.uint8(img_rms)
        img_final = np.uint8(img_final)
        img_sum = cv2.bitwise_and(img_sum, img_sum, mask=img_final)
        img_rms = cv2.bitwise_and(img_rms, img_rms, mask=img_final)
        #print(img_sum)
        print('img.shape = {}, img_sum.shape = {}, img_rms.shape = {}'.format(img.shape, img_sum.shape, img_rms.shape))
        #img_mean_rms = cv2.merge((img_rms, img_rms, img_rms))
        #img_mean_rms = cv2.merge((img_sum, img_sum, img_sum))
        img_sum_255 = np.matrix(img_sum)
        img_sum_255[img_sum_255 > 0] = 255
        #img_mean_rms = cv2.merge((img_sum, img_rms, np.zeros(img_sum.shape, img_sum.dtype)))
        img_mean_rms = cv2.merge((img_sum, img_rms, img_sum_255))
        cv2.imshow('meanrms', img_mean_rms)
        cv2.imwrite('meanrms.png', img_mean_rms)
        if 0:
          #print(img.shape, edge_t.shape, edge_b.shape)
          print(edge_b)
          edge_b = cv2.cvtColor(edge_b, cv2.COLOR_GRAY2BGR)
          edge_b[:, :, 1:3] = 0
          img_with_edge = cv2.bitwise_or(img[:-16,:], edge_b)
          edge_t = cv2.cvtColor(edge_t, cv2.COLOR_GRAY2BGR)
          edge_t[:, :, 0:2] = 0
          #img_with_edge = cv2.bitwise_or(img_with_edge[:-16,:], edge_t)
          cv2.imshow('img_with_edge', img_with_edge)
          cv2.imwrite('img_with_edge.png', img_with_edge)
        for y,l,r in zip(ys,ls,rs):
          #if l > r:
          #  l,r = r,l
          print(y,l,r)
          #cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
          cv2.rectangle(img, (l, y-1), (r, y+16), (0, 255, 0), 1)
        cv2.imshow('output', img)
        cv2.imwrite('output.png', img)
        cv2.waitKey(0)
        return ''
        a
        #print(img.shape)
        #edge_l = self._edge(img, 'l')
        #edge_r = self._edge(img, 'r')
        edge_b = self._edge(img, 'b')
        edge_t = self._edge(img, 't')
        '''edge_4 = cv2.bitwise_or(edge_l, edge_r)
        edge_4 = cv2.bitwise_or(edge_4, edge_b)
        edge_4 = cv2.bitwise_or(edge_4, edge_t)
        cv2.imshow('edge_4', edge_4)'''
        edge1 = cv2.bitwise_and(edge_b[16:, :], edge_t[:-16, :])
        edge1 = cv2.dilate(edge1, np.matrix('1 '*edge1.shape[0]))
        cv2.imshow('edge1', edge1)
        cv2.waitKey(0)
        return ''
        a
        #edge_fill = self._edgefill(edge_4, 'l')
        cv2.imshow('edge_fill', edge_fill)
        cv2.waitKey(0)
        a
        img = self._threshold(img)
        print(self._run_tesseract(img))
        a
        new_drop = self._get_drop_from_img(img)
        old_drop = self.current_drop.copy()
        for d in new_drop:
            if d in old_drop:
                old_drop.remove(d)
        self.current_drop = old_drop + new_drop
        return self.current_drop
    
    def _edge_b_t(self, img, start_y = 0, signal = [222, 237, 252, 267, 282, 300], signal_x = {}):
      np.set_printoptions(threshold=np.inf, linewidth=np.inf)
      print_size = 30
      line_size_erode = 0
      line_size_sum = 25
      av_grad = 20
      min_val = av_grad * line_size_sum
      min_val = 20
      img_dy_t = cv2.filter2D(img[:-16, :], -1, np.matrix('1; -1'))
      img_dy_b = cv2.filter2D(img[16:, :], -1, np.matrix('-1; 1'))
      img_dy_tb = img_dy_t + img_dy_b
      #print(img_dy_tb[0, :])
      img_dy_tb_sum = cv2.filter2D(img_dy_tb, -1, np.matrix('1 '*line_size_sum)) / (2*line_size_sum)
      dx = int((line_size_sum - 1) / 2)
      img_dy_tb_sum[:, :dx] = np.zeros(img_dy_tb_sum[:, :dx].shape)
      img_dy_tb_sum[:, -dx:] = np.zeros(img_dy_tb_sum[:, -dx:].shape)
      # merc
      img_dy_tb_sum[0:75-start_y, 18:65] = np.zeros(img_dy_tb_sum[0:75-start_y, 18:65].shape)
      # plugy msg
      img_dy_tb_sum[104-start_y:134-start_y, 14:293] = np.zeros(img_dy_tb_sum[104-start_y:134-start_y, 14:293].shape)
      #img_dy_tb_sum.rowRange(0, dx).setTo(0)
      #img_dy_tb_sum.rowRange(img_dy_tb_sum.shape[1]-dx, img_dy_tb_sum.shape[1]).setTo(0)
      img_dy_tb_sum[img_dy_tb_sum < min_val] = 0
      #img_dy_tb_sum = cv2.filter2D(img_dy_tb, -1, np.matrix('1 '*line_size_sum))
      #print(img_dy_tb_sum[0, :])
      #ret = np.where(img_dy_tb_sum > 20)
      img_rms = np.zeros(img_dy_t.shape, img.dtype)
      img_mean = np.zeros(img_dy_t.shape, img.dtype)
      img_mean_rms = np.zeros(img_dy_t.shape, object)
      img_final = np.zeros(img_dy_t.shape, img.dtype)
      #print(dx)
      for y in range(img.shape[0]-16-1):
        for x in range(dx, img.shape[1]-dx):
          #if img_dy_tb_sum[y, x] < 20:
          #  continue
          #print(img_dy_t[y, x-dx:x+dx])
          #if img_dy_tb_sum[y, x] < min_val:
          #  continue
          if img_dy_tb_sum[y, x] == 0:
            continue
          mean_tb = img_dy_tb_sum[y, x]
          #sum_tb = sum(img_dy_t[y, x-dx:x+dx+1]) + sum(img_dy_b[y, x-dx:x+dx+1])
          #mean_tb = sum_tb / (line_size_sum*2)
          #if mean_tb != img_dy_tb_sum[y, x]:
          #  print('x, y = {}, {} mean, mean_img, rms = {}, {}, {}'.format(x, y, mean_tb, img_dy_tb_sum[y, x], None))
          #if mean_tb < 20:
          #  continue
          rms_tb = sum([(x-mean_tb)**2 for x in img_dy_t[y, x-dx:x+dx]] + [(x-mean_tb)**2 for x in img_dy_b[y, x-dx:x+dx]])
          rms_tb = math.sqrt(rms_tb / (line_size_sum*2))
          #print(sum_tb, rms_tb)
          img_mean[y, x] = int(mean_tb)
          img_rms[y, x] = int(rms_tb)
          img_mean_rms[y, x] = '{},{}'.format(img_mean[y, x], img_rms[y, x])
          #if mean_tb > rms_tb and rms_tb < 40 and mean_tb > 20:
          if mean_tb > rms_tb:
            img_final[y, x] = 1
            #img_final[y, x] = mean_tb
            #img_final[y, x] = rms_tb
        img_mean[y, 0] = max(img_mean[y, :])
      #print('img_sum:\n{}'.format(img_mean))
      #print('img_rms:\n{}'.format(img_rms))
      #print('img_mean_rms:\n{}'.format(img_mean_rms))
      #print('img_final:\n{}'.format(img_final))
      img_final = cv2.erode(img_final, np.matrix('1 '*8))
      '''self._set_y_to_zero(img_final, 3, 0, 60, 1)
      self._set_y_to_zero(img_final, 40, 0, 60, 1)
      self._set_y_to_zero(img_final, 85, 0, 200, 1)
      self._set_y_to_zero(img_final, 104, start_y, 200, 1)
      self._set_y_to_zero(img_final, 115, start_y, 300, 1)
      self._set_y_to_zero(img_final, 119, start_y, 300, 1)'''
      ys = set(np.where(img_final == 1)[0])
      # remove entering plateau
      if 141 in ys and 142 in ys:
        ys.remove(141)
        ys.remove(142)
      ls, rs = [], []
      #print("img:\n{}".format(img))
      drop = []
      for y in ys:
        print('y = {}'.format(y))
        l, r = [0]*img.shape[1], [0]*img.shape[1]
        l_rms, r_rms = [0]*img.shape[1], [0]*img.shape[1]
        l_rms_p, r_rms_p = [0]*img.shape[1], [0]*img.shape[1]
        l_best, r_best = [0]*img.shape[1], [0]*img.shape[1]
        for x in range(4, img.shape[1]-4):
          l[x], l_rms[x], l_rms_p[x] = self._calc_h_edge(img, y, x, 'l')
          r[x], r_rms[x], r_rms_p[x] = self._calc_h_edge(img, y, x, 'r')
          if l_rms[x] != 0:
            l_best[x] = l[x] / l_rms[x] * 10
          if r_rms[x] != 0:
            r_best[x] = r[x] / r_rms[x] * 10
          #l.append(np.sum(img[y+1:y+16+1, x]) - np.sum(img[y:y+16+1, x+1]))
          #r.append(np.sum(img[y+1:y+16+1, x+1]) - np.sum(img[y:y+16+1, x]))
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
        fig.subplots_adjust(hspace = 0.0, left = 0.06, right = 0.97)
        #plt.figure('l')
        ax1.plot(range(img.shape[1]), l)
        ax1.plot(range(img.shape[1]), l_rms)
        ax1.plot(range(img.shape[1]), l_rms_p)
        ax1.plot(range(img.shape[1]), l_best)
        #plt.figure('r')
        ax2.plot(range(img.shape[1]), r)
        ax2.plot(range(img.shape[1]), r_rms)
        ax2.plot(range(img.shape[1]), r_rms_p)
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax2.plot(range(img.shape[1]), r_best)
        #print(ax2.get_ylim())
        if y in signal_x:
          ax1.vlines(signal_x[y][0], *ax1.get_ylim(), colors='b')
          ax2.vlines(signal_x[y][1], *ax2.get_ylim(), colors='b')
        #plt.show()
        #cv2.waitKey()
        l_max = max(l_best)
        l_max_index = l_best.index(l_max)
        ls.append(l_max_index)
        r_max = max(r_best)
        r_max_index = r_best.index(r_max)
        rs.append(r_max_index)
        print('y = {} l = {}[{}] r = {}[{}]'.format(y, l_max, l_max_index, r_max, r_max_index))
        drop.append(self._box(img[y:y+16+1, l_max_index+1:r_max_index], self.img_orig[y:y+16+1, l_max_index+1:r_max_index]))
        #print(l)
        #print(r)
        #img_edge_l = cv2.filter2D(img[y:y+16+1, :-1], -1, np.matrix('0 0' + ';1 -1'*16)
        #img_edge_r = cv2.filter2D(img[y:y+16+1, 1:], -1, np.matrix('0 0' + ';-1 1'*16)
      #max_in_row = [max(img_dy_tb_sum[y, :]) for y in range(0, img_dy_tb_sum.shape[0])]
      #print(max_in_row)
      #for y in signal:
      #  print('y = {}'.format(max(img_dy_tb_sum[y, :])))
      #max_pix = np.iinfo(img.dtype).max
      #img_dy_tb_thr = cv2.inRange(img_dy_tb_sum, min(line_size_sum*av_grad*2, max_pix), max_pix)
      print('ys = {}'.format(ys))
      return (ys, ls, rs, img_dy_tb_sum, img_rms, img_final)
      img_edge_b = self._edge_new(img[16:, :], 'b', print_size=print_size, line_size_erode=line_size_erode, line_size_sum=line_size_sum, av_grad=av_grad)
      cv2.imshow('edge_b', img_edge_b)
      #print(img.shape, img_edge_b.shape)
      #print(img[start_y:, :].shape, img_edge_b[:-start_y, :].shape)
      #img_with_edge = cv2.bitwise_or(img[:, :], img_edge_b[:, :])
      #cv2.imshow('img_with_edge', img_with_edge)
      img_edge_t = self._edge_new(img[:-16, :], 't', print_size=print_size, line_size_erode=line_size_erode, line_size_sum=line_size_sum, av_grad=av_grad)
      cv2.imshow('edge_t', img_edge_t)
      img_edge = cv2.bitwise_and(img_edge_b, img_edge_t)
      img_edge = cv2.erode(img_edge, np.matrix('1 '*15))
      #print('edged:\n{}'.format(img_edge[:, :print_size]))
      # suppress players set and soj
      #print(img_edge[104, 16:26] == 255*np.ones((1, 10)))
      #ret = np.where(img_edge == 255)
      #print(*zip(ret[0], ret[1]))
      #a
      self._set_y_to_zero(img_edge, 104, start_y, 200, min_val)
      self._set_y_to_zero(img_edge, 119, start_y, 300, min_val)
      self._set_y_to_zero(img_edge, 115, start_y, 300, min_val)
      #img_edge = cv2.dilate(img_edge, np.matrix('1 '*img.shape[0]))
      ys = set(np.where(img_edge >= min_val)[0])
      print('ys: {} [{}]'.format(len(ys), ys))
      cv2.imshow('edged', img_edge)
      #cv2.waitKey(0)
      return (ys, img_edge_t, img_edge_b)
      #a
      #line = np.matrix('1 '*line_size_erode)
    
    def _calc_h_edge(self, img, y, x, mode):
      if mode == 'l':
        xp, xm = x, x+1
      elif mode == 'r':
        xp, xm = x, x-1
      diff = [p - m for p,m in zip(img[y:y+16, xp].tolist(), img[y:y+16, xm].tolist())]
      #print(diff)
      mean = sum(diff) / len(diff)
      rms = math.sqrt(sum((d-mean)**2 for d in diff) / len(diff))
      xp_list = img[y+1:y+1+16, xp].tolist()
      mean_p = sum(xp_list) / len(xp_list)
      rms_p = math.sqrt(sum((d-mean_p)**2 for d in xp_list) / len(xp_list))
      return mean, rms, rms_p

    def _set_y_to_zero(self, img, y, start_y, x, min_val):
      if img.shape[0] < y:
        return
      if len(np.where(img[y-start_y, :x] == min_val)[0]) > 0 and len(np.where(img[y-start_y, x:] == min_val)[0]) == 0:
        #print('removing')
        img[y-start_y, :] = 0
    
    def _box(self, img_orig, img_col):
      img = np.uint8(img_orig)
      #print(img.shape)
      #print(img)
      ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
      if th is None:
        return {}
      print('thresholding: ret = {}, th.shape = {}'.format(ret, th.shape))
      cv2.imshow('th', th)
      img_col_th = cv2.bitwise_and(img_col, img_col, mask=cv2.bitwise_not(th))
      cv2.imshow('th_col', img_col_th)
      mean_col = cv2.mean(img_col, mask=cv2.bitwise_not(th))[0:3]
      img_1 = np.zeros((1, 1, 3), dtype='uint8')
      img_1[0, 0] = mean_col
      #print('mean_col: {}'.format(mean_col))
      #print(np.matrix(mean_col))
      hsv = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
      print('mean_col: {} hsv: {}'.format(mean_col, hsv))
      #hue_c = {'g': 45, 'y': 60, 'g': 120, 'b': 240}
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
      th = th[1:-4, 4:-4]
      th = cv2.resize(th, (th.shape[1]*4, th.shape[0]*4))
      cv2.imshow('col', img_col)
      #print(img_col.shape, th.shape, cv2.bitwise_not(th).shape)
      #cv2.waitKey()
      bsize = 5
      th_with_edge = cv2.copyMakeBorder(th, top=bsize, bottom=bsize, left=bsize, right=bsize, borderType=cv2.BORDER_CONSTANT, value=255)
      #th_with_edge = th
      #th_with_edge = 255*np.ones((th.shape[0] + 4, th.shape[1] + 4))
      #th_with_edge[2:-2, 2:-2] = th
      #th = th_with_edge
      #print(th)
      #print(ret)
      drop = self._run_tesseract(th_with_edge)
      print('drop: {} [{}]'.format(drop, drop_color))
      #cv2.waitKey()
      return {'drop': drop, 'drop_color': drop_color}

    def _get_drop_from_img(self, img):
        #ret = []
        #for i in range(random.randrange(4, start=1)):
        #    ret.append(self.items[random.randrange(len(self.items))])
        #return ret
        self._run_east(img)
        return [self.items[random.randrange(len(self.items))] for i in range(random.randrange(1, 5))]
    
    def _run_tesseract(self, image):
      cv2.imshow('tesseract', image)
      cv2.imwrite('tesseract.png', image)
      #cv2.waitKey()
      res = pytesseract.image_to_data(image, config=self.config_tesseract+' --psm 7')
      #print('res: {}'.format(res))
      #conf, text = res.splitlines()[-1].split()[-2:]
      conf, text = [], []
      for l in res.splitlines()[1:]:
        #print(l.split())
        c, t = l.split()[-2:]
        #print(c, t)
        if t != '-1' and float(c) > 0:
          conf.append(c)
          text.append(t)
      #print([l.split()[-2:] for l in res.splitlines()])
      #print(zip([l.split()[-2:] for l in res.splitlines()]))
      #conf, text = zip([l.split()[-2:] for l in res.splitlines()])
      print('detected[{}]: {}'.format(conf, text))
      return ' '.join(text)


dropper = Drop()

def py_droprec(img, timestamp=0, start_y=0, signal_x={}):
  dropper.img_orig = img
  print('SZ droprec timestamp = {}'.format(timestamp))
  #return ''
  x1,x2 = 137,510 # basic
  #img = img[x1:x2, :]
  #img = img[140:295, 167:765] # all drop range
  #cv2.imshow(timestamp + ' cropped', img)
  #print ("Contents of a :")
  #print (a)
  #c = '42str'
  #return c
  #img = img[244:144, 362:161]
  #img = img[145:161, 244:362] # one box
  #img = img[146:163, 244:362] # one box extended
  #img = img[:580, :] # cut only small bottom
  cv2.imshow('input', img)
  cv2.imwrite('input.png', img)
  drop = dropper.process_drop(img, start_y, signal_x)
  dropper.reset_drop()
  #print(drop)
  s = '\n'.join(drop)
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
  # crop_img = img[y:y+h, x:x+w]
  start_y=23
  #fin = '/home/zenaiev/games/Diablo2/ssr/158458820669.png'
  fin = '../../502/screens/95621693075.png'
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
    img = cv2.imread(fin)
    #img = np.int8(img)
    #img = np.int8(img)
    #img = img[140:169, 244:362] # one box extended
    #img = img[start_y:550, :] # cut only small top and bottom
    #img = img[320:340, 335:407]
    #img = img[300:340, 335:407]
    #img = img[300:340, 315:427]
    img = img[300:340, 275:467] # two items
    signal_x = {4: [9, 187], 22: [61, 132]}
    ret = py_droprec(img, start_y=start_y, signal_x=signal_x)
    print(ret)
  else:
    for img_name in list_images(sys.argv[1]):
      print(img_name)
      img = cv2.imread(img_name)
      img = img[start_y:550, :] # cut only small top and bottom
      ret = py_droprec(img, start_y=start_y)
      print(ret)
