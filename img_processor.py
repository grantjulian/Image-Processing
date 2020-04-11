# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:32:54 2020

@author: Grant Julian

"""

from PIL import Image
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="kernel convolution image processing algorithm")
parser.add_argument("input", help="name of the input file")
parser.add_argument("output", help="name of the output file")
parser.add_argument("algorithm", help="blur3 blur5 edge1 edge2 edge3 edge4 sharpen")
args = parser.parse_args()

matrix = np.genfromtxt("algorithms/" + args.algorithm + ".csv", delimiter=',')
woff, hoff = np.floor_divide(np.subtract(matrix.shape, (1,1)),(2,2))

img = Image.open(args.input)
width, height = img.size
pixels = np.reshape(img.getdata(),(height,width,3))
pcopy = pixels.copy()
img.close()

for i in range(hoff, height - hoff):
    for j in range(woff, width - woff):
        for k in range(3):
            pcopy[i,j,k] = np.sum(np.multiply(matrix,pixels[i-hoff:i+hoff+1, j-woff:j+woff+1, k]))
            if pcopy[i,j,k] > 255:
                pcopy[i,j,k] = 255
            elif pcopy[i,j,k] < 0:
                pcopy[i,j,k] = 0

        
output = Image.fromarray(pcopy.astype(np.uint8))
output.save(args.output)

