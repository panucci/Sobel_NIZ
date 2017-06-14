import sys
from PIL import Image
import math
import numpy
import scipy
from scipy import ndimage
from skimage import filters
import time
from scipy import signal as sg
from skimage import feature
import matplotlib.pyplot as plt

argImageName = sys.argv[1]
argBrightness = float(sys.argv[2])
canny = int(sys.argv[3])
T1 = 0.9
T2 = 0.3

if(argBrightness > 0 and argBrightness <=1):
	brightness = argBrightness
else:
	brightness = 0.7

imageName = "images/" + argImageName
img = Image.open(imageName)
pixelMatrix = img.load()

# GRAYSCALE
print("Image size: %s x %s  " % img.size)
print("------------------------------------------------------------")
start = time.time()  
grayScaleMatrix = numpy.zeros((img.width, img.height))
grayScaleRoberts = numpy.zeros((img.width, img.height))

for i in range(0, img.width):
    for j in range(0, img.height):
        grayScale = (pixelMatrix[i,j][0] * 0.2126 + pixelMatrix[i,j][1] * 0.7152 + pixelMatrix[i,j][2] * 0.0722) 
        grayScaleRoberts[i,j] = math.ceil(grayScale)
        grayScale = math.ceil(grayScale * brightness)
        if(grayScale > 255):
            grayScale = 255
        pixelMatrix[i,j] = grayScale, grayScale, grayScale
        grayScaleMatrix[i,j] = grayScale
img.save("results/0grayScale.png")     
print('GRAYSCALE:')
end = time.time()
time1 = end - start
print("Elapsed time: %s" % time1)  
print("------------------------------------------------------------") 

# CANNY
start = time.time()  
edges1 = feature.canny(grayScaleMatrix, canny, low_threshold=T1, high_threshold=T2)
scipy.misc.imsave('results/1canny.jpg', numpy.transpose(edges1))
print('CANNY:')
end = time.time()
time2 = end - start
print("Elapsed time: %s" % time2)  
print("------------------------------------------------------------")

# ROBERTS
start = time.time()  
roberts = filters.roberts(grayScaleRoberts)
scipy.misc.imsave('results/2roberts.jpg', numpy.transpose(roberts))
print('ROBERT-CROSS:')
end = time.time()
time3 = end - start
print("Elapsed time: %s" % time3)  
print("------------------------------------------------------------") 
       
# MOJA IMPLEMENTACIJA
start = time.time()
xAxisSobel = [[-1,0,1],
              [-2,0,2],
              [-1,0,1]]

yAxisSobel = [[-1,-2,-1],
              [0,0,0],
              [1,2,1]]



for x in range(1, img.width-2):
  for y in range(1, img.height-2):
    xPixel =  (xAxisSobel[0][0] * grayScaleMatrix[x-1,y-1]) +\
              (xAxisSobel[0][1] * grayScaleMatrix[x,y-1])   +\
              (xAxisSobel[0][2] * grayScaleMatrix[x+1,y-1]) +\
              (xAxisSobel[1][0] * grayScaleMatrix[x-1,y])   +\
              (xAxisSobel[1][1] * grayScaleMatrix[x,y])     +\
              (xAxisSobel[1][2] * grayScaleMatrix[x+1,y])   +\
              (xAxisSobel[2][0] * grayScaleMatrix[x-1,y+1]) +\
              (xAxisSobel[2][1] * grayScaleMatrix[x,y+1])   +\
              (xAxisSobel[2][2] * grayScaleMatrix[x+1,y+1]) 

    yPixel =  (yAxisSobel[0][0] * grayScaleMatrix[x-1,y-1]) +\
              (yAxisSobel[0][1] * grayScaleMatrix[x,y-1])   +\
              (yAxisSobel[0][2] * grayScaleMatrix[x+1,y-1]) +\
              (yAxisSobel[1][0] * grayScaleMatrix[x-1,y])   +\
              (yAxisSobel[1][1] * grayScaleMatrix[x,y])     +\
              (yAxisSobel[1][2] * grayScaleMatrix[x+1,y])   +\
              (yAxisSobel[2][0] * grayScaleMatrix[x-1,y+1]) +\
              (yAxisSobel[2][1] * grayScaleMatrix[x,y+1])   +\
              (yAxisSobel[2][2] * grayScaleMatrix[x+1,y+1]) 
    pixelValue = math.sqrt(math.pow(xPixel, 2) + math.pow(yPixel, 2))
    pixelValue = math.ceil(pixelValue)
    pixelMatrix[x,y] = pixelValue, pixelValue, pixelValue 
        
img.save("results/3sobelBasic.png")    
print('SOBEL BASIC IMPLEMETATION:')
end = time.time()
time4 = end - start
print("Elapsed time: %s" % time4)  

# MOJA IMPLEMENTACIJA UZ POMOC CONVOLVE
print("------------------------------------------------------------")
start = time.time()

#valX = sg.convolve(grayScaleMatrix, xAxisSobel, "valid")
#valY = sg.convolve(grayScaleMatrix, yAxisSobel, "valid")

valX = ndimage.filters.convolve(grayScaleMatrix, xAxisSobel, origin=-1)
valY = ndimage.filters.convolve(grayScaleMatrix, yAxisSobel, origin=-1)

tmpMatrix = numpy.zeros((valX.shape[0], valX.shape[1]))

for x in range(0, valX.shape[0]-1):
  for y in range(0, valX.shape[1]-1):
    tmp = math.sqrt(math.pow(valX[x,y], 2) + math.pow(valY[x,y], 2))
    tmpMatrix[x,y] = math.ceil(tmp * 1.5)
    
#scipy.misc.imsave('4convolve.jpg', numpy.transpose(tmpMatrix))
print("SOBEL IMPLEMETATION USING SCIPY.NDIMAGE.FILTERS.CONVOLVE:")
end = time.time()
time5 = end - start
print("Elapsed time: %s" % time5)  
print("------------------------------------------------------------")    

# POMOCU SCIPY BIBLIOTEKE
start = time.time() 
scipySobelImage = scipy.misc.imread(imageName)
scipySobelImage = scipySobelImage.astype('int32')

xAxis = ndimage.sobel(scipySobelImage, 0)  
yAxis = ndimage.sobel(scipySobelImage, 1)  
scipyFinal = numpy.hypot(xAxis, yAxis)  
scipyFinal *= 255.0 / numpy.max(scipyFinal) 
scipy.misc.imsave('results/4scipySobel.jpg', scipyFinal)
print ('SCIPY SOBEL:')
end = time.time()
time6 = end - start
print("Elapsed time: %s" % time6)  
print("------------------------------------------------------------")
      
plt.bar(1, time1, label = 'Grayscale')    
plt.bar(2, time2, label = "Canny")
plt.bar(3, time3, label = 'Robert-Cross')
plt.bar(4, time4, label = 'Sobel Basic')
plt.bar(5, time5, label = 'Sobel Improved')
plt.bar(6, time6, label = 'Sobel Scipy')
plt.xlabel("Algorithm")
plt.ylabel("Time (s)")
plt.legend()

plt.show()