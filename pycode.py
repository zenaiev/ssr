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

    def process_drop(self, img, start_y=0):
        #img = self._preprocess_img(img)
        #self._preprocess_img(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = np.int16(img_gray)
        #img_gray = np.int8(img_gray)
       #img_gray = img_gray.astype('float')
        ret = self._edge_b_t(img_gray, start_y=start_y)
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
          if l > r:
            l,r = r,l
          print(y,l,r)
          #cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
          cv2.rectangle(img, (l, y), (r, y+16), (0, 255, 0), 1)
        cv2.imshow('output', img)
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
    
    def _edge_b_t(self, img, start_y = 0, signal = [222, 237, 252, 267, 282, 300]):
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
      for y in ys:
        l, r = [], []
        for x in range(4, img.shape[0]-1-4):
          l.append(np.sum(img[y-16:y, x]) - np.sum(img[y-16:y, x+1]))
          r.append(np.sum(img[y-16:y, x+1]) - np.sum(img[y-16:y, x]))
        l_max = max(l)
        l_max_index = l.index(l_max)
        ls.append(l_max_index)
        r_max = max(r)
        r_max_index = r.index(r_max)
        rs.append(r_max_index)
        print('y = {} l = {}[{}] r = {}[{}]'.format(y, l_max, l_max_index, r_max, r_max_index))
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
    
    def _set_y_to_zero(self, img, y, start_y, x, min_val):
      if img.shape[0] < y:
        return
      if len(np.where(img[y-start_y, :x] == min_val)[0]) > 0 and len(np.where(img[y-start_y, x:] == min_val)[0]) == 0:
        #print('removing')
        img[y-start_y, :] = 0

    def _edge_new(self, img, mode, print_size = 18, line_size_erode = 4, line_size_sum = 25, av_grad = 10, debug=0):
      np.set_printoptions(threshold=np.inf, linewidth=np.inf)
      max_pix = np.iinfo(img.dtype).max
      if debug:
        print('input shape: {}'.format(img.shape))
        print('input\n', img[:, :print_size])
      if mode == 'b':
        kernel = np.matrix('-1; 1')
      elif mode == 't':
        kernel = np.matrix('1; -1')
      img_edge = cv2.filter2D(img, -1, kernel)
      #cv2.imshow('tmp', img_edge)
      #cv2.waitKey()
      #a
      if debug:
        print('der {}\n'.format(mode), img_edge[:, :print_size])
      if line_size_erode > 0:
        img_edge = cv2.erode(img_edge, np.matrix('1 '*line_size_erode))
        if debug:
          print('der {} erode\n'.format(mode), img_edge[:, :print_size])
      img_edge = cv2.filter2D(img_edge, -1, np.matrix('1 '*line_size_sum))
      if debug:
        print('der {} sum\n'.format(mode), img_edge[:, :print_size])
      img_edge = cv2.inRange(img_edge, min(line_size_sum*av_grad, max_pix), max_pix)
      if debug:
        print('der {} threshold\n'.format(mode), img_edge[:, :print_size])
      return img_edge

    def _edgefill(self, img, mode):
        if mode == 'l':
          kernel = np.matrix('1 -1')
          line_size = 3
          line = np.matrix(';'.join('1'*line_size))
        elif mode == 'r':
          kernel = np.matrix('-1 1')
          line_size = 3
          line = np.matrix(';'.join('1'*line_size))
        elif mode == 'b':
          kernel = np.matrix('-1; 1')
          line_size = 35
          line = np.matrix('1 '*line_size)
        elif mode == 't':
          kernel = np.matrix('1; -1')
          line_size = 35
          line = np.matrix('1 '*line_size)
        kernel = np.matrix('1;' + ';'.join('0'*14) + ';1')
        #print(kernel)
        img1 = cv2.filter2D(img, -1, kernel)
        print(img1.shape)
        cv2.imwrite('conv_fill.png', img1)
        return img1

    def _edge(self, img, mode):
        #vert = cv.getStructuringElement(cv.MORPH_RECT, (1, 7))
        if mode == 'l':
          line_size = 3
          kernel = np.matrix('1 -1')
          line = np.matrix(';'.join('1'*line_size))
        elif mode == 'r':
          line_size = 3
          kernel = np.matrix('-1 1')
          line = np.matrix(';'.join('1'*line_size))
        elif mode == 'b':
          line_size = 6
          kernel = np.matrix('-1; 1')
          line = np.matrix('1 '*line_size)
        elif mode == 't':
          line_size = 6
          kernel = np.matrix('1; -1')
          line = np.matrix('1 '*line_size)
        #print(kernel)
        img1 = cv2.filter2D(img, -1, kernel)
        print(img1.shape)
        #cv2.imwrite('conv1.png', img1)
        #img1 = cv2.inRange(img1, 38, 114)
        img1 = cv2.inRange(img1, 30, 120)
        img1 = img1 / line_size
        img1 = cv2.filter2D(img1, -1, line)
        #print(img1)
        img1 = cv2.inRange(img1, 255, 255)
        #img1 = cv2.bitwise_and(img1, img1, mask=mask1)
        #cv2.imshow('conv_'+mode, img1)
        #cv2.waitKey(0)
        #a
        return img1
    
    def _threshold(self, img):
        img = imutils.resize(img, img.shape[1] * 5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', img)
        #return img
        #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        #_,img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        #cv2.imshow('threshold0', img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        #img = cv2.dilate(img,kernel,iterations = 1)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.bitwise_not(img)
        cv2.imshow('threshold', img)
        cv2.waitKey(0)
        return img

    def _preprocess_img(self, img):
        pass
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
        gradY = np.absolute(gradY)
        (minVal, maxVal) = (np.min(gradY), np.max(gradY))
        gradY = 255 * ((gradY - minVal) / (maxVal - minVal))
        gradY = gradY.astype("uint8")
        cv2.imshow("Preprocessed", img)
        cv2.imshow("GradY", gradY)
        cv2.waitKey(0)
        a
        return img

    def _get_drop_from_img(self, img):
        #ret = []
        #for i in range(random.randrange(4, start=1)):
        #    ret.append(self.items[random.randrange(len(self.items))])
        #return ret
        self._run_east(img)
        return [self.items[random.randrange(len(self.items))] for i in range(random.randrange(1, 5))]
    
    def _run_tesseract(self, image):
        res = pytesseract.image_to_data(image, config=self.config_tesseract+' --psm 7')
        conf, text = res.splitlines()[-1].split()[-2:]
        print('detected[{}]: {}'.format(conf, text))
        return text

    def _run_east(self, image):
        min_confidence = 0.5
        orig = image.copy()

        # resize the image and grab the new image dimensions
        (H, W) = image.shape[:2]
        newW = int(math.ceil(W/32)*32)
        newH = int(math.ceil(H/32)*32)
        print('[INFO] resizing {} -> {}'.format((W,H), (newW,newH)))
        rW = W / float(newW)
        rH = H / float(newH)
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(self.basedir + 'frozen_east_text_detection.pb')

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()

        # show timing information on text prediction
        print("[INFO] text detection took {:.6f} seconds".format(end - start))

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < min_confidence:
                    continue

                # compute the offset factor as our resulting feature maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        #boxes = rects

        # loop over the bounding boxes
        drop = []
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective ratios
            #startX = int(startX * rW)
            #startY = int(startY * rH)
            #endX = int(endX * rW)
            #endY = int(endY * rH)

            # draw the bounding box on the image
            #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # get text
            img_box = orig[startY:endY, startX:endX]
            #(H, W) = img_box.shape[:2]
            #print(startX, endX, startY, endY, H, W)
            drop.append(self._run_tesseract(img_box))
            #cv2.imshow("Text line", img_box)
            #cv2.waitKey(0)

        # show the output image
        #cv2.imshow("Text Detection", orig)
        cv2.imshow("Text Detection", image)
        cv2.waitKey(0)


dropper = Drop()

def py_droprec(img, timestamp=0, start_y=0):
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
    drop = dropper.process_drop(img, start_y)
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
        print('dupa')
        return basePath
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)

if __name__ == '__main__':
  # crop_img = img[y:y+h, x:x+w]
  start_y=23
  #fin = '/home/zenaiev/games/Diablo2/ssr/158458820669.png'
  #fin = '../../502/screens/95621693075.png'
  #fin = '../../502/screens/97525956388.png'
  #fin = '../../502/screens/95842955271.png'
  fin = '../../502/screens/95509173077.png'
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
    img = img[start_y:550, :] # cut only small top and bottom
    #img = img[320:340, 335:407]
    #img = img[300:340, 335:407]
    ret = py_droprec(img, start_y=start_y)
    print(ret)
  else:
    for img_name in list_images(sys.argv[1]):
      print(img_name)
      img = cv2.imread(img_name)
      img = img[start_y:550, :] # cut only small top and bottom
      ret = py_droprec(img, start_y=start_y)
      print(ret)
