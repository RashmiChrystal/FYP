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
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt

img_dir = r"F:\G\UOM\Level 4\FYP\datasets\Deforestation Dataset\converted images"
os.chdir(img_dir)
#hist_path = os.path.abspath(__file__)
hist_path = r"C:\Users\user\PycharmProjects\FYP"
df = pd.DataFrame()
df_sift = pd.DataFrame()
df_newFeatures1 = pd.DataFrame()

#To read gray image
def readImage(img_list, img):
    n = cv2.imread(img, 0)
    img_list.append(n)
    return img_list

path = glob.glob("*.jpg")
list_ = []

cv_image = [readImage(list_, img) for img in path]

def readImagesFromFolder():
    list_1 = []
    list_2 = []
    img_dir = r"F:\G\UOM\Level 4\FYP\datasets\Deforestation Dataset\converted images"
    for img in os.listdir(img_dir):
        read_img = cv2.imread(os.path.join(img_dir,img))
        if read_img is not None:
            list_1.append(read_img)
            list_2.append(img)
            path1 = hist_path + '\CS_images\Original'
            cv2.imwrite(os.path.join(path1, img), read_img)
            #print("When being read:", img)
    return list_2

def readImagesFromFolder2():
    list_3 = []
    list_4 = []
    imdir = "F:/G/UOM/Level 4/FYP/datasets/Deforestation Dataset/converted images/"
    filenames = glob.glob(imdir+"*.jpg")
    filenames.sort()
    print("filenames", filenames)
    for img in filenames:
        n = cv2.imread(img, 0)
        if n is not None:
            list_3.append(n)
            list_4.append(img)
    return list_3

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
    plt.hist(image.ravel(), 256, [0, 256])
    #plt.show()
    frame = image.copy()
    frame1 = image.copy()
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    xp1 = [5, 50, 145, 250]
    fp1 = [5, 10, 245, 250]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    frame = cv2.LUT(frame, table)
    plt.hist(frame.ravel(), 256, [0, 256])
    #plt.show()
    y = np.arange(256)
    table1 = np.interp(y, xp1, fp1).astype('uint8')
    frame1 = cv2.LUT(frame1, table1)
    plt.hist(frame1.ravel(), 256, [0, 256])
    #plt.show()
    return image

def contrastStretching2(image):
    minmax_img = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            minmax_img[i,j] = 255*(image[i,j]-np.min(image))/(np.max(image)-np.min(image))
    plt.hist(image.ravel(), 256, [0, 256])
    #plt.show()
    return minmax_img

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
    #cv2.imwrite(os.path.join(hist_path, "Canny.jpg"), edges)
    edges2 = edges.reshape(-1)
    df['Canny edges'] = edges2

def edge_roberts(img):
    edge_robert = roberts(img)
    #cv2.imwrite(os.path.join(hist_path, "Roberts.jpg"), edge_robert)
    #cv2.imshow("Roberts.jpg", edge_robert)
    #cv2.waitKey()
    edge_robert2 = edge_robert.reshape(-1)
    df['Roberts'] = edge_robert2

def edge_sobel(img):
    edge_sobel1 = sobel(img)
    #cv2.imwrite(os.path.join(hist_path, "Sobel.jpg"), edge_sobel1)
    #cv2.imshow("Sobel.jpg", edge_sobel1)
    #cv2.waitKey()
    edge_sobel2 = edge_sobel1.reshape(-1)
    df['Sobel'] = edge_sobel2

def edge_scharr(img):
    edge_scharr1 = scharr(img)
    #cv2.imwrite(os.path.join(hist_path, "Scharr.jpg"), edge_scharr1)
    #cv2.imshow("Scharr.jpg", edge_scharr1)
    #cv2.waitKey()
    edge_scharr2 = edge_scharr1.reshape(-1)
    df['Scharr'] = edge_scharr2

def edge_prewitt(img):
    edge_prewitt1 = prewitt(img)
    #cv2.imwrite(os.path.join(hist_path, "Prewitt.jpg"), edge_prewitt1)
    #cv2.imshow("Prewitt.jpg", edge_prewitt1)
    #cv2.waitKey()
    edge_prewitt2 = edge_prewitt1.reshape(-1)
    df['Prewitt'] = edge_prewitt2

def gaussianFeature(img):
    gaussian_img = nd.gaussian_filter(img, sigma = 3)
    gaussian_img1 = gaussian_img.reshape(-1)
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    #cv2.imshow("GS3.jpg", gaussian_img)
    #cv2.waitKey()
    #cv2.imshow("GS7.jpg", gaussian_img2)
    #cv2.waitKey()
    df['Gaussian S3'] = gaussian_img1
    df['Gaussian S7'] = gaussian_img3

def varianceFeature(img):
    variance_img = nd.generic_filter(img, np.var, size =3)
    #cv2.imshow("variance_img", variance_img)
    #cv2.waitKey()
    variance_img1 = variance_img.reshape(-1)
    df['Variance'] = variance_img1

def siftAlgo(img):
    print(img.shape[:2])
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img, keypoints, img)
    return img, keypoints, descriptors

def entropyFeature(img):
    entropy_img = entropy(img, disk(1))
    #cv2.imwrite(os.path.join(hist_path, "ENtropy image.jpg"), entropy_img)
    #cv2.imshow("Entropy Img", entropy_img)
    #cv2.waitKey()
    entropy_img1 = entropy_img.reshape(-1)
    df['Entropy'] = entropy_img1

def iteratePics():
    path = glob.glob("*.jpg")

def performPCA():
    print("Performaing PCA")
    #read_file = pd.read_excel(r'F:\G\UOM\Level 4\FYP\Implementation\main.pyfeatures.xls')
    #read_file.to_csv(r'F:\G\UOM\Level 4\FYP\Implementation\features.csv', index = None, header = True)
    featureset = pd.read_csv(r'C:\Users\user\PycharmProjects\FYP\Features.csv')

    #Standardize the data (The first step of PCA)
    X_std = StandardScaler().fit_transform(featureset)

    #compute covariance matrix
    mean_vec= np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std-mean_vec)) / (X_std.shape[0]-1)
    print('Covariance matrix: \n%s' %cov_mat)

    #Get eigen values
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    # for i in eig_pairs:
    #     print('eig pair: ', i[0])

    pca = sklearnPCA(n_components=2)
    pca.fit_transform(featureset)
    print('PCA Variance ratio: ', pca.explained_variance_ratio_)

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
    #contrast_stretched_image = contrastStretching2(image)
    #cv2.imwrite(os.path.join(hist_path, "contrast_stretched.jpg"), contrast_stretched_image)
    #cv2.imshow("contrast_stretched", contrast_stretched_image)
    #cv2.waitKey(0)
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
    gaussianFeature(gaussian_image)
    varianceFeature(gaussian_image)

    siftimg, keypoints, descriptors = siftAlgo(gaussian_image)
    # cv2.imshow("sift image", siftimg)
    # cv2.waitKey()
    print("keypoints: ", keypoints)
    print("descriptors: ", np.size(descriptors[0]))
    print("descriptors: ", np.size(descriptors[1]))
    print("descriptors: ", np.size(descriptors[60]))
    # num2 = 1
    # for item in descriptors:
    #     df_sift[str(num2)] = item
    #     num2 += 1
    entropyFeature(gaussian_image)
    performPCA()
    print(df.head(10))
    df.to_csv('C:/Users/user/PycharmProjects/FYP/Features.csv')
    #df_sift.to_csv('C:/Users/user/PycharmProjects/FYP/Sift_Descriptors.csv')

    #**************************************************************************************************************
    #for multiple images
    list_3 = readImagesFromFolder2()
    contrast_stretched_image_list = []
    num_CS = 0
    for img in list_3:
        CS_img = contrastStretching(img)
        contrast_stretched_image_list.append(CS_img)
        path = hist_path + '\CS_images'
        label = 'CS_' + str(num_CS) + '.jpg'
        cv2.imwrite(os.path.join(path, label), CS_img)
        gaussian_img = gaussianBlur(CS_img)
        path1 = hist_path + '\Gaussian_imgs'
        label1 = 'G_' + str(num_CS) + '.jpg'
        cv2.imwrite(os.path.join(path1, label1), gaussian_img)
        #use the original pixel values of the gaussian images
        num_CS = num_CS + 1

    #list_4 = readImagesFromFolder2()
    #for img in list_4:
    #    print(img)

    print("end")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
