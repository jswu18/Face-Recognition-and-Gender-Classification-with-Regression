
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from scipy.misc import imsave as save

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result
testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

filenames = ["subset_actresses.txt", "subset_actors.txt"]
for subset in range(len(filenames)):
    act = list(set([a.split("\t")[0] for a in open(filenames[subset]).readlines()]))
    for a in act:
        name = a.split()[1].lower()
        i = 0
        j = 0
        for line in open(filenames[subset]):
            if a in line:
                sentence = line.split()
                filename = sentence[1]+str(i)+'.jpg'
                dimensions = sentence[-2]
                dimensions = dimensions.replace(",", " ")
                dimensions = dimensions.split()
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                # urllib.urlretrieve(line.split()[4], "uncropped/"+filename)
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                try:
                    im = imread("uncropped/"+filename)
                    im = im[int(dimensions[1]):int(dimensions[3]),int(dimensions[0]):int(dimensions[2])]
                    im = imresize(im, (32,32))
                    im = rgb2gray(im)
                    save("cropped/"+filename, im)
                    print "success:", filename
                    print "number of corrupt images:", j                    
                    i += 1
                except:
                    j+=1
                    print "read failed:", filename
                    print "number of corrupt images:", j
                    i+=1
                    continue



                


            
        

    