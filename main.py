# -*- coding:utf-8 -*-
from Scan import Scansheet
import os
import cv2 as cv
from e import ContourCountError, ContourPerimeterSizeError, PolyNodeCountError
# ROW=20
# Num = (int(input('please input the start number:')) - 1)  # The first number of ques -1
# file = open(input('题块号:')+".txt")
file = open('1-4.txt')
ANSWER_KEY=eval(file.read())
file.close()
# print(ANSWER_KEY)
# 对横向照片旋转
# img=cv.imread('test10.jpg')
# dct=cv.transpose(img)
# img=cv.flip(dct,0)

# Scansheet(img)
# 设定摄像头参数
cap=cv.VideoCapture(cv.CAP_DSHOW+2)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
# img=cv.imread("test6.jpg")
# 可以
if(cap.isOpened()):
    ret, img = cap.read()
    # dct=cv.transpose(img)
    # img=cv.flip(dct,0)
    # print(img.shape,end='\r')
    # print(img.shape)
    # cv.imshow("img", img)
    try:
        Scansheet(img,ANSWER_KEY)
        # cv.waitKey(0)
    except Exception as e:
        print(e)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     print('hjhjh')
    #     #     break
        # continue
cap.release()
# cv.destroyAllWindows()
