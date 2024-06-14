import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import math as m
import cv2


#Grayscale function
def grayscale(image):
    #make blurring kernel
    newImage = np.zeros((image.shape[0], image.shape[1]))
    for y in range(image.shape[1]):  
        for x in range(image.shape[0]):
             r, g, b = image[x,y,0], image[x,y,1], image[x,y,2]
             gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
             newImage[x,y] = gray    

    return newImage

def gaussianFilter(input_array, size):
    mean_filter_kernel = np.ones((size,size))/(size**2)
    blur_filtered_image = np.zeros(input_array.shape)
    
    for i in range (input_array.shape[0]): 
        for j in range (input_array.shape[1]):
            summation = 0
            for k in range (mean_filter_kernel.shape[0]):
                for l in range (mean_filter_kernel.shape[1]):
                    if (i + k - 1 < input_array.shape[0]) and j + l - 1 < input_array.shape[1]:
                        summation += mean_filter_kernel[k][l] * input_array[i + k - 1][j + l - 1]
            blur_filtered_image[i][j]  = summation
    return(blur_filtered_image)


def edgeDetection(img):
# Sobel Edge Detection
    x = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis

    y = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis

    xy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    # Canny Edge Detection
    edge = cv2.Canny(image=img, threshold1=50, threshold2=150)  # Canny Edge Detection

    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edge)
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()