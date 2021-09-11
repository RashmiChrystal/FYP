# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import pandas as pd
import os
import glob
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd

img_dir = r"F:\G\UOM\Level 4\FYP\datasets\Deforestation Dataset\converted images"
os.chdir(img_dir)
#hist_path = os.path.abspath(__file__)
hist_path = r"C:\Users\user\PycharmProjects\FYP"
df = pd.DataFrame()

#To read gray image
def readImage(img_list, img):
    n = cv2.imread(img, 0)
    img_list.append(n)
    return img_list

path = glob.glob("*.jpg")
list_ = []
cv_image = [readImage(list_, img) for img in path]

def readOneImage(img_list, num):
    image = img_list[num]
    return image

#pre-processing of the gray-scale image
def performContrastStretching(images):
    contrast_stretched_list_ = []
    for img in images:
        frame = img.copy()
        xp = [0,64,128,192,255]
        fp = [0,16,128,240,255]
        x = np.arange(256)
        table = np.interp(x,xp,fp).astype('uint8')
        img = cv2.LUT(img, table)
        contrast_stretched_list_.append(img)
    return contrast_stretched_list_[0]

def contrastStretching(image):
    frame = image.copy()
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    image = cv2.LUT(image, table)
    return image

#it'll be better to perform histogram equilization too

def medianFilter(img):
    median = cv2.medianBlur(img, 5)
    return median

def gaussianBlur(image):
    gaussian = cv2.GaussianBlur(image,(5,5),0)
    return gaussian

def gaborFilter(img):
    num = 1
    fimg_list = []
    for theta in range(2):
        theta = theta/4. * np.pi
        for sigma in (3,5):
            for lamda in np.arange(0, np.pi, np.pi/4.):
                for gamma in (0.05, 0.5):
                    gabor_label = 'Gabor' + str(num)
                    kernel = cv2.getGaborKernel((5,5), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img
                    fimg_list.append(fimg)
                    #cv2.imwrite(os.path.join(hist_path, gabor_label), fimg)
                    num += 1
    return fimg_list

def cannyEdge(img):
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite(os.path.join(hist_path, "Canny.jpg"), edges)
    edges2 = edges.reshape(-1)
    df['Canny edges'] = edges2

def edge_roberts(img):
    edge_robert = roberts(img)
    cv2.imwrite(os.path.join(hist_path, "Roberts.jpg"), edge_robert)
    edge_robert2 = edge_robert.reshape(-1)
    df['Roberts'] = edge_robert2

def edge_sobel(img):
    edge_sobel1 = sobel(img)
    cv2.imwrite(os.path.join(hist_path, "Sobel.jpg"), edge_sobel1)
    edge_sobel2 = edge_sobel1.reshape(-1)
    df['Sobel'] = edge_sobel2

def edge_scharr(img):
    edge_scharr1 = scharr(img)
    cv2.imwrite(os.path.join(hist_path, "Sobel.jpg"), edge_scharr1)
    edge_scharr2 = edge_scharr1.reshape(-1)
    df['Scharr'] = edge_scharr2

def edge_prewitt(img):
    edge_prewitt1 = prewitt(img)
    cv2.imwrite(os.path.join(hist_path, "Sobel.jpg"), edge_prewitt1)
    edge_prewitt2 = edge_prewitt1.reshape(-1)
    df['Prewitt'] = edge_prewitt2

def gaussianFeature(img):
    gaussian_img = nd.gaussian_filter(img, sigma = 3)
    gaussian_img1 = gaussian_img.reshape(-1)
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    cv2.imwrite(os.path.join(hist_path, "GS3.jpg"), gaussian_img)
    cv2.imwrite(os.path.join(hist_path, "GS7.jpg"), gaussian_img2)
    df['Gaussian S3'] = gaussian_img1
    df['Gaussian S7'] = gaussian_img3

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    #cv2.imshow("pic", list_[0])
    #cv2.waitKey(5000)
    image = readOneImage(list_, 50)
    #cv2.imshow("pic2", image)
    #cv2.waitKey(5000)

    contrast_stretched_image = contrastStretching(image)
    cv2.imwrite(os.path.join(hist_path, "contrast_stretched.jpg"), contrast_stretched_image)
    median_image = medianFilter(contrast_stretched_image)
    cv2.imwrite(os.path.join(hist_path, "median.jpg"), median_image)
    gaussian_image = gaussianBlur(contrast_stretched_image)
    cv2.imwrite(os.path.join(hist_path, "gaussian.jpg"), gaussian_image)
    gaussian_image2 = gaussian_image.reshape(-1)
    df['Original pixel values'] = gaussian_image2
    print(df.head(10))

    gabor_images = gaborFilter(gaussian_image)
    num1 = 1
    for img in gabor_images:
        gabor_label = "Gabor" + str(num1) + ".jpg"
        cv2.imwrite(os.path.join(hist_path, gabor_label), img)
        num1 += 1

    cannyEdge(gaussian_image)
    edge_roberts(gaussian_image)
    edge_sobel(gaussian_image)
    edge_scharr(gaussian_image)
    edge_prewitt(gaussian_image)
    gaussianFeature(img)
    print(df.head(10))
    df.to_csv('C:/Users/user/PycharmProjects/FYP/Features.csv')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
