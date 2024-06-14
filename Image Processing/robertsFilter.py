#1. greyscale
#2. gaussian blur 
#3. prewitt

#importing all libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

    original_image = cv2.imread("/Users/divyamanvikar/Desktop/Test3.png") # reads and saves the image we want to save
    greyscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) # converts original image to greyscale using openCV
    gaussian_blur_image = cv2.GaussianBlur(greyscale_image, (9, 9), 0) # converts greyscale image to gaussian blurred image

    # prewitt filter kernels
    roberts_kernelx = np.array([[1,0],[0,-1]])
    roberts_kernely = np.array([[0,1],[-1,0]])

    # convolving filter kernels with gaussian blurred image
    robertsx_image = cv2.filter2D(src = gaussian_blur_image, ddepth = -1, kernel = roberts_kernelx)
    robertsy_image = cv2.filter2D(src = gaussian_blur_image,ddepth = -1, kernel = roberts_kernely)
    roberts_total_image = robertsx_image + robertsy_image # adds together the images found from convolving the x and y direction kernels

    edge = cv2.Canny(image=roberts_total_image, threshold1=50, threshold2=150)  # applies canny edge detection to prewitt filtered images to create an image that has only edges in a certain threshold

    # display all images
    cv2.imshow('Canny Edge Detection', edge)
    cv2.imshow('Roberts X', robertsx_image)
    cv2.imshow('Roberts Y', robertsy_image)
    cv2.imshow('Roberts Total Image', roberts_total_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()