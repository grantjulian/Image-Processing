# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:32:54 2020

@author: Grant Julian

"""

from PIL import Image
from numba import jit
import numpy as np
import argparse
import time

# gpu acceleration speeds up runtime by ~10x
@jit(nopython=True)
# goes through every pixel and applies kernel (matrix)
def process(pcopy, pixels, matrix):
    for i in range(hoff, height - hoff):
        for j in range(woff, width - woff):
            # k handles r,g,b channels - no support for alpha
            for k in range(3):
                pcopy[i][j][k] = np.sum(np.multiply(matrix, pixels[i-hoff:i+hoff+1, j-woff:j+woff+1, k]))
                if pcopy[i,j,k] > 255:
                    pcopy[i,j,k] = 255
                elif pcopy[i,j,k] < 0:
                    pcopy[i,j,k] = 0

# keeps track of runtime
start_time = time.time()
# handles input arguments
parser = argparse.ArgumentParser(description="kernel convolution image processing algorithm")
parser.add_argument("input", help="name of the input file")
parser.add_argument("output", help="name of the output file")
parser.add_argument("algorithm", help="blur3 edge1 edge2 edge3 edge4 sharpen")
args = parser.parse_args()

# matrix is the kernel of the specificied algorithm (a 3x3 or 5x5 grid of weights)
matrix = np.genfromtxt("algorithms/" + args.algorithm + ".csv", delimiter=',')
# width offset and height offset- how far around a given pixel to apply kernel
woff, hoff = np.floor_divide(np.subtract(matrix.shape, (1,1)),(2,2))

# open up input image
img = Image.open(args.input)
width, height = img.size
pixels = np.reshape(img.getdata(),(height,width,3))
# makes a copy of the images pixels that will go on to be the output
pcopy = pixels.copy()
img.close()
print("processing image %s with %s kernel" % (args.input, args.algorithm))
process(pcopy, pixels, matrix)

        
output = Image.fromarray(pcopy.astype(np.uint8))
output.save(args.output)
print("created %s" % args.output)
print("finished in %.4f seconds" %(time.time() - start_time))
