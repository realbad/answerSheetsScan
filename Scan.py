import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2 as cv
from e import ContourCountError, ContourPerimeterSizeError, PolyNodeCountError

ANS=["A","B","C","D"]
CHOICE=4
MARKSIZE=130 # 涂黑块像素大小，一般无需调整
def Scansheet(img,ANSWER_KEY):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Num = (int(input('please input the start number:')) - 1)
    Num = list(ANSWER_KEY.keys())[0]-1
    # 打印灰度图
    # cv.imshow("gray",gray)

    # 高斯滤波，清除一些杂点
    blur = cv.GaussianBlur(gray, (3, 3), 0)

    # 自适应二值化算法
    thresh2 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 13)

    # 打印二值化后的图
    # cv.imshow("thresh2",thresh2)

    # 寻找轮廓
    image, cts, hierarchy = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 打印找到的轮廓
    # print("轮廓数：", len(cts))

    # 对拷贝的原图进行轮廓标记
    contour_flagged = cv.drawContours(img.copy(), cts, -1, (0, 0, 255), 1)
    # 打印轮廓图
    # cv.imshow("contours_flagged", contour_flagged)
    # 按像素面积降序排序
    List = sorted(cts, key=cv.contourArea, reverse=True)
    # 尝试先对原图进行切割，失败
    # X,Y,W,H = cv.boundingRect(list[0])
    # RectImg = cv.rectangle(blur.copy(), (X, Y), (X + W, Y + H), (0, 255, 0), 2)
    # thresh2=thresh2[Y:Y+H,X:X+W]
    # cv.imshow("thresh22",thresh2)
    # img=img[Y:Y+H,X:X+W]
    # 遍历轮廓
    # image, cts, hierarchy = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # list = sorted(cts, key=cv.contourArea, reverse=True)

    for ct in List:
        # 周长，第1个参数是轮廓，第二个参数代表是否是闭环的图形,实际上是下个函数的准确度
        peri = 0.03 * cv.arcLength(ct,True)
        # 获取多边形的所有定点，如果是四个定点，就代表是矩形
        approx = cv.approxPolyDP(ct, peri, True)
        # 只考虑矩形
        # print(len(approx))
        # cv.imshow('rec',RectImg)

        if ((len(approx) == 4) ):

            # 从原图中提取所需的矫正图片
            ox = four_point_transform(img, approx.reshape(4, 2))
            # cv.imwrite('ox.jpg',ox)
            shape = (ox.shape)
            ox=cv.resize(ox,(936,shape[0]),interpolation=cv.INTER_CUBIC)
            cv.imwrite('oxext.jpg',ox)
            # 从原图中提取所需的矫正图片
            tx = four_point_transform(gray, approx.reshape(4, 2))
            tx = cv.resize(tx, (936, shape[0]), interpolation=cv.INTER_CUBIC)
            # 打印矫正后的灰度图
            # cv.imshow("tx",tx)

            # 对矫正图进行高斯模糊
            blur = cv.GaussianBlur(tx, (3, 3), 0)
            # cv.imshow('blur',blur)

            # 对矫正图做自适应二值化
            thresh2 = cv.adaptiveThreshold(tx, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 13)

            # 打印矫正后的二值化图
            # cv.imshow("tx_thresh2", thresh2)
            # cannyimg=cv.Canny(thresh2.copy(),100,300)
            # cv.imshow('cannyimg',cannyimg)
            # 获取轮廓
            r_image, r_cts, r_hierarchy = cv.findContours(thresh2, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
            # 打印得到轮廓数量
            # print("第二层轮廓数：", len(r_cts))
            contour_flagged_2 = cv.drawContours(ox.copy(), r_cts, -1, (0, 0, 255), 1)
            # 打印轮廓图
            # cv.imshow("contours_flagged_2", contour_flagged_2)

            # 用于存储答案的python list变量
            question_list=[]
            hulls=[]
            quesBorders=[]
            Circleimg=ox.copy()
            for (i,r_ct) in enumerate(r_cts) :
                # 转为矩形，分别获取 x，y坐标，及矩形的宽和高
                (x, y), radius = cv.minEnclosingCircle(r_ct)
                # 获取凸包
                hull = cv.convexHull(r_ct)
                # r_peri = 0.01 * cv.arcLength(r_ct, True)
                # # 获取多边形的所有定点，如果是四个定点，就代表是矩形
                # quesBorder = cv.approxPolyDP(r_ct, r_peri, True)
                # area = cv.contourArea(quesBorder)
                # if(len(quesBorder) == 4 and area > 100):
                #     quesBorders.append(r_ct)
                #     hulls.append(hull)
                # area = cv.contourArea(r_ct)
                # perimeter = cv.arcLength(r_ct, True)
                # 过滤掉不符合答案坐标和长宽的选项,过滤规则？还需进一步研究一些狗屁填涂无法识别

                hulls.append(hull)
                if ((245>x>130 or 410>x>290 or 570>x>450 or 734>x>610 or 900>x>775) and y>40 and 7<radius<15 ):
                    # cv.drawContours(ox, r_ct, -1, (0, 0, 255), 1)
                    Circleimg = cv.circle(Circleimg, (int(x), int(y)), int(radius), (0, 0, 255), 1)

                    # print(area)
                    # print(x)
                    # 对轮廓线根据hierarchy属性进行进一步筛选
                    if(r_hierarchy[0][i][3]==-1):
                        question_list.append(r_ct)
                        # print(r_hierarchy[0][i])
                        Circleimg = cv.circle(Circleimg, (int(x), int(y)), int(radius), (0, 0, 255), 1)

            # hullimg = cv.drawContours(tx.copy(), hulls,-1, (0, 0, 255), 1)


            # quesBorderimg = cv.drawContours(tx.copy(),quesBorders,-1,(255,255,255),1)
            # cv.imshow('quesborder',quesBorderimg)
            # quesBorderimg=cv.GaussianBlur(quesBorderimg, (3, 3), 0)
            # offBorder = cv.adaptiveThreshold(quesBorderimg,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,31,15)
            # r_image, r_cts, r_hierarchy = cv.findContours(offBorder, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
            # offBorderimg = cv.drawContours(ox.copy(), r_cts, -1, (0, 0, 255), 1)
            # for (i,r_ct) in enumerate(r_cts) :
            #     (x, y), radius = cv.minEnclosingCircle(r_ct)
            #     if ((245>x>130 or 410>x>290 or 570>x>450 or 734>x>610 or 900>x>775) and y>30 and 7<radius<15 ):
            #         # cv.drawContours(ox, r_ct, -1, (0, 0, 255), 1)
            #         Circleimg = cv.circle(Circleimg, (int(x), int(y)), int(radius), (0, 0, 255), 1)
            #         if(r_hierarchy[0][i][3]==-1):
            #             question_list.append(r_ct)
            # cv.imshow('hullimg',hullimg)
            # cv.imshow("circleimg",Circleimg)
            # cv.imshow('offBorder', offBorder)
            # cv.imshow('offBorderimg',offBorderimg)
            print("答总数",len(question_list))
            # print(question_list)

            # grayques=cv.cvtColor(quespic, cv.COLOR_BGR2GRAY)
            # circles = cv.HoughCircles(grayques, cv.HOUGH_GRADIENT, 1, 1, param1=10, param2=15, minRadius=6, maxRadius=10)
            # print(len(circles[0]))
            # for circle in circles[0]:
            #     xh = int(circle[0])
            #     yh = int(circle[1])
            #     # 半径
            #     rh = int(circle[2])
            #     # 在原图用指定颜色标记出圆的位置
            #     tx = cv.circle(tx, (xh, yh), rh, (0, 0, 255), -1)
            # cv.imshow('hough', tx)
            # 按坐标从上到下排序
            questionCnts = contours.sort_contours(question_list, method="top-to-bottom")[0]

            def sortCircle(cnts):
                circles=[]
                # rows=[[] for i in range(ROW)]
                rows=[]
                # row1=np.empty(2)
                for cnt in cnts:
                    circles.append(cv.minEnclosingCircle(cnt))
                # print(cnts[0])
                rows.append([cnts[0]])
                # np.append(row1[0], cnts[0])
                j=0
                # 判断同一行
                for i,circle in enumerate(circles):
                    if((i+1)<len(cnts)):
                        if((circles[i+1][0][1]-circle[0][1])<10 ):
                            rows[j].append(cnts[i + 1])
                        else:
                            # j += 1
                            rows.append([cnts[i + 1]])
                            j += 1
                    else:
                        break
                    # elif((i+1)==len(cnts)):
                    #     # 判断是不是同一行
                    #     if((circle[0][1]-circles[i-1][0][1])<10):
                    #         rows[j].append(cnts[i])
                    #     else:
                    #         # j += 1
                    #         rows.append([cnts[i]])
                    #         j += 1
                i=0
                rowSort=[]
                # row为一行
                for row in rows:
                    row=contours.sort_contours(rows[i], method="left-to-right")[0]
                    # CX=[]
                    # for circ in row:
                    #     m=(cv.moments(circ))
                    #     CX.append(int(m['m10'] / m['m00']))
                    # if len(row)%CHOICE != 0:
                    #     for i,cx in enumerate(CX):
                    #         if(CX[i+1]-cx>50):
                    rowSort.append(row)
                    i += 1
                return rowSort
            # 得到所有顺序边界

            quesSorted=sortCircle(questionCnts)
            # print(quesSorted)
            quespic = cv.drawContours(ox.copy(), questionCnts, -1, (0, 0, 255), 1)
            cv.imshow("ques", quespic)
            cv.imwrite('ques.jpg', quespic)
            # for questionCnt in questionCnts:
            #     print(len(questionCnt))
            # print(questionCnts)
            # questionCnt里面的应当是轮廓的点的个数

            #  使用np函数，按5个元素，生成一个集合
            answer = {}
            #对r行
            for r in range(len(quesSorted)):
                # 中的每道题
                for (q, i) in enumerate(np.arange(0, len(quesSorted[r]), CHOICE)):

                    # 每一个行CHOICE个答案，从左到右排序
                    cnts = contours.sort_contours(quesSorted[r][i:i + CHOICE])[0]
                    # 存储一行题里面的每个答案
                    ans_list = []
                    # 对每个答案
                    for (j, cc) in enumerate(cnts):

                        # 生成全黑画布
                        mask = np.zeros(thresh2.shape, dtype="uint8")
                        # 将每一个答案按轮廓写上去，并将填充颜色设置成白色
                        tpp = cv.drawContours(mask, [cc], -1, 255, -1)
                        # cv.imshow('black',tpp)
                        # 两个图片做位运算
                        mask = cv.bitwise_and(thresh2, thresh2, mask=mask)

                        # 统计每个答案的像素
                        total = cv.countNonZero(mask)

                        # 添加到集合里面
                        ans_list.append( (total,j) )

                    # print(ans_list)
                    # 按像素大小排序
                    # ans_list=sorted(ans_list,key=lambda  x:x[0],reverse=True)

                    Num += 1
                    marks=[] #存储所有涂黑的,另外似乎还做了一遍二次筛选？
                    for ans in ans_list:
                        if(ans[0]>MARKSIZE): # MAgic number
                            marks.append(ans)
                    # max_ans_num=ans_list[0][1]
                    # max_ans_size=ans_list[0][0]
                    # print("题号：",Num,end='')
                    # print("答案：",end='')#,"列表：",ans_list)
                    # marktmp = [] 保存答案
                    # for mark in marks:
                    #     print(ANS[mark[1]],end="")
                    #     marktmp.append(ANS[mark[1]])
                    # answer.update({Num:marktmp})
                    # print('') #相当于最后加一个\n，很神奇，不然没东西了

                    for mark in (marks):
                        # cv.drawContours(ox, cnts[mark[1]], -1, (0, 0, 255), 2)
                    # 给正确答案，标记成错误红色正确绿色mark->(223,0)
                        if (ANS[mark[1]] not in ANSWER_KEY[Num]):
                            cv.drawContours(ox, cnts[mark[1]], -1, (0, 0, 255), 2)
                        else:
                            cv.drawContours(ox, cnts[mark[1]], -1, (0, 255, 0), 3)
                    for ABCD in ANSWER_KEY[Num]:
                        z=int(chr(ord(ABCD)-17))
                        # 正确答案green
                        try:
                            cv.drawContours(ox, cnts[z], -1, (144, 0, 50), 2)#紫色的代表正确但缺失
                        except:
                            continue
                            # cv.drawContours(ox, cnts[mark[1]], -1, (0, 255, 0), 2)
            # 作保存答案之用，平时不用
            # file = open('answer.txt', 'w')
            # file.write(str(answer))
            # file.close()
            cv.imshow("answer_flagged", ox)
            # cv.waitKey(0)

            # 最大的轮廓就是我们想要的，之后的就可以结束循环了
            break
        # 阻塞等待窗体关闭
    cv.waitKey(0)
    cv.destroyAllWindows()