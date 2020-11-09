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

    def process_drop(self, img):
        #img = self._preprocess_img(img)
        #self._preprocess_img(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

def py_droprec(img, timestamp=0):
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
    img = img[:580, :] # cut only small bottom
    cv2.imshow('cropped', img)
    drop = dropper.process_drop(img)
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
    if len(sys.argv) == 1:
      img = cv2.imread('/home/zenaiev/games/Diablo2/ssr/158458820669.png')
      ret = py_droprec(img)
      print(ret)
    else:
      for img_name in list_images(sys.argv[1]):
        print(img_name)
        img = cv2.imread(img_name)
        ret = py_droprec(img)
        print(ret)
