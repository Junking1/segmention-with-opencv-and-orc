import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image
import random
import os
from PIL import Image
import pytesseract
import glob as go

def myimrite(pathname,file):
    #避免文件名相同导致的图片被覆盖
    pic = os.path.exists(pathname + '.jpg')
    if not pic:
        cv2.imwrite(pathname + '.jpg', result)
        file.write(pathname+'.jpg     ')  # 将文件夹名称写入文件
    else:
        pathname=pathname+'_1'
        myimrite(pathname,file)

def MyNMS(result1):
    # fei'ji'da'yi'zhi
    kernel3 = np.ones((80, 80), np.uint8)
    gray_result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
    gray_result1 = cv2.morphologyEx(gray_result1, cv2.MORPH_OPEN, kernel3)
    ret_result1, binary_result1 = cv2.threshold(gray_result1, 1, 255, cv2.THRESH_BINARY)
    cv2.bitwise_not(binary_result1, binary_result1)
    contours_result1, hierarchy_result1 = cv2.findContours(binary_result1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(result1.copy(), contours_result1, -1, (0, 0, 255), 3)

    if len(contours_result1) > 1 :
        area = []
        for c_result1 in range(len(contours_result1)):
            area.append(cv2.contourArea(contours_result1[c_result1]))
        max_idx = np.argmax(area)
        for i_result1 in range(len(contours_result1)):
            if i_result1 != max_idx:
                rect_result1 = cv2.minAreaRect(contours_result1[i_result1])
                box_result1 = np.int0(cv2.boxPoints(rect_result1))

                draw_img_result1 = cv2.drawContours(result1.copy(), [box_result1], -1, (0, 0, 255), 3)
                Xs_result1 = [i_result1[0] for i_result1 in box_result1]
                Ys_result1 = [i_result1[1] for i_result1 in box_result1]
                x1_result1 = abs(min(Xs_result1))
                x2_result1 = abs(max(Xs_result1))
                y1_result1 = abs(min(Ys_result1))
                y2_result1 = abs(max(Ys_result1))
                for a in range(x1_result1,x2_result1-1):
                    for b in range(y1_result1, y2_result1-1):
                        c_result=1
                        if c_result > 0:
                            result1[b, a, 0] = 255
                            result1[b, a, 1] = 255
                            result1[b, a, 2] = 255
        result = result1
    else :
        result=result1
    return result



#定义开闭卷积核
kernel1 = np.ones((100,100),np.uint8)
kernel2 = np.ones((50,50),np.uint8)

full_path ='Check.txt'  # 也可以创建一个.doc的word文档
img_path = go.glob("D:\SpareFish\SE_pic\\*.jpg")


for a_path in img_path:
    # #图片读取及size获取
    print(a_path)
    img = cv2.imread(a_path)
    pictue_size = img.shape
    picture_hight = pictue_size[0]
    picture_width = pictue_size[1]
    img2 = img[picture_hight - 360:picture_hight - 100, 1400:picture_width - 1300]
    image2 = Image.fromarray(img2)
    text = pytesseract.image_to_string(image2, lang='eng',config="--psm 7 digit")
    print(text)

    folder = os.path.exists(text)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(text)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print
        "---  new folder...  ---"
        print
        "---  OK  ---"
    else:
        print
        "---  There is this folder!  ---"

    #灰度转换后进行开闭运算并二值化后二值反转
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel1)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)
    ret, binary = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    cv2.bitwise_not(binary,binary);


    #如果输入图像分辨率过大进行缩放方便观察调试
    scale_percent = 100  # percent of original size
    width = int(binary.shape[1] * scale_percent / 100)
    height = int(binary.shape[0] * scale_percent / 100)
    dim = (width, height)
    rebinary = cv2.resize(binary, dim, interpolation=cv2.INTER_NEAREST)
    #边界提取，contours包含边界值的坐标，hierarchy包含分类
    contours, hierarchy = cv2.findContours(rebinary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


    cv2.drawContours(img.copy(),contours,-1,(0,0,255),3)
    file = open(full_path, 'a',encoding='utf-8')
    file.write('\n')  # 将回车写入文件
    file.write(a_path)  # 路径写入文件
    file.write('       ')  # 将空格写入文件
    file.write('folder： ' + text+'     pic： ')  # 将文件夹名称写入文件


    print(len(contours))
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = abs(min(Xs))-20
        x2 = abs(max(Xs))+20
        y1 = abs(min(Ys))-200
        y2 = abs(max(Ys))+500
        hight = y2 - y1
        width = x2 - x1
        crop_img = img.copy()[y1:y1 + hight-1, x1:x1 + width-1]
        image = Image.fromarray(crop_img)
        text_c = pytesseract.image_to_string(image, lang='eng')
        print(text_c)
        result1= img.copy()[y1+180:y1 + hight-480, x1:x1 + width]


        #########特殊NMS

        result = MyNMS(result1)

        pathname=text+'\\'+text_c
        myimrite(pathname,file)

    file.close

