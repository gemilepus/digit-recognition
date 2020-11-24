# coding=utf-8
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
# 讀取分類器
clf = joblib.load("digits_cls.pkl")
im = cv2.imread("photo_1.jpg")  # 讀取輸入圖片
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 灰度圖化
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)  # 高斯模糊（去噪）

ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)  # 閾值：二值化
ctrs, hier = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查詢圖像輪廓
rects = [cv2.boundingRect(ctr) for ctr in ctrs]  # 框出目標

# 對查詢的目標識別：計算HOG特徵圖並且使用SVM預測數字
for rect in rects:
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]

    # resize 圖片
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # 計算 HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

