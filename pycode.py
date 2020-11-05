import pytesseract
import numpy as np
import imutils
import cv2
import random

class Drop:
    def __init__(self):
        with open('/home/zenaiev/games/Diablo2/ssr/d2-drop-items.txt') as f:
            self.items = [l[:-1] for l in f.readlines()]
        self.current_drop = []
    
    def reset_drop(self):
        self.current_drop = []

    def process_drop(self, img):
        new_drop = Drop._get_drop_from_img(img, self.items)
        old_drop = self.current_drop.copy()
        for d in new_drop:
            if d in old_drop:
                old_drop.remove(d)
        self.current_drop = old_drop + new_drop
        return self.current_drop

    @staticmethod
    def _get_drop_from_img(img, items_all=None):
        #ret = []
        #for i in range(random.randrange(4, start=1)):
        #    ret.append(self.items[random.randrange(len(self.items))])
        #return ret
        return [items_all[random.randrange(len(items_all))] for i in range(random.randrange(1, 5))]

dropper = Drop()

def py_droprec(img, timestamp=0):
    print('SZ droprec timestamp = {}'.format(timestamp))
    x1,x2 = 137,510
    #img = img[:, x1:x2]
    #cv2.imshow(timestamp + ' cropped', img)
    #print ("Contents of a :")
    #print (a)
    #c = '42str'
    #return c
    drop = dropper.process_drop(img)
    dropper.reset_drop()
    #print(drop)
    s = '\n'.join(drop)
    return s


if __name__ == '__main__':
    img = cv2.imread('/home/zenaiev/games/Diablo2/502/screens/158458820669.png')
    ret = py_droprec(img)
    print(ret)