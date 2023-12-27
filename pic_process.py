import cv2
import numpy as np
from skimage import measure
import os
import sys
import time


class Processor:

    def __int__(self, pic_path: str) -> None:
        super().__init__()

    @staticmethod
    def pic_init(self, pic_path):
        # 读取图片信息
        pic = cv2.imread(pic_path)
        # 图片灰度化
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        # 图片二值化
        ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        # print(type(binary))
        return ret, binary

    def pic_show(self, pic):
        cv2.imshow('1', pic)
        # 等待按键
        cv2.waitKey(0)

    def find_position_binary(self, binary):
        contours_list = []
        # 找到二值化图像轮廓坐标
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        # print(len(contours))
        for i, val in enumerate(contours):
            # 找到轮廓最左坐标，宽高
            x, y, w, h = cv2.boundingRect(val)
            print([i, x, y, w, h])
            contours_list.append([i, x, y, w, h])
        return contours, contours_list

    # 返回模板contours， 和对应名称
    def get_module_contours(self):
        img = cv2.imread('module.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 滤波腐蚀膨胀
        # 图片二值化
        ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        # 获取边缘数据
        c, h = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_shape_name = ["heart", "star", "triangle", "rectangle", "arrow", "circular", "else"]
        return c, contours_shape_name
        # for index, i in enumerate(c):
        #     x, y, w, h = cv2.boundingRect(i)
        #     cv2.rectangle(img, (x, y),
        #                   (x + w, y + h),
        #                   (0, 0, 255), 3)
        #     cv2.putText(img, str(index), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 (255, 0, 0), 1)
        # cv2.imshow('1', img)
        # cv2.waitKey(0)

    def draw(self, pic, contours_lis):
        for i in range(len(contours_lis)):
            # 画矩形框
            cv2.rectangle(pic, (contours_lis[i][1], contours_lis[i][2]),
                          (contours_lis[i][1] + contours_lis[i][3], contours_lis[i][2] + contours_lis[i][4]),
                          (0, 0, 255), 3)
            cv2.putText(pic, str(i), (contours_lis[i][1] + 10, contours_lis[i][2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)

    def shape_match(self, dir_path):
        # 获取匹配模板的数据
        module_contours, module_name = self.get_module_contours()
        # 灰度化
        for dir_item in os.scandir(dir_path):
            img = cv2.imread(dir_item.path)
            print(dir_item.path)
            # 获取图像宽度和高度
            height, width, _ = img.shape
            print(width, height)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            # self.pic_show(binary)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 滤波腐蚀膨胀
            # to do
            # 坐标和宽高信息
            contours_lis = []
            # for i, val in enumerate(contours):
            #     # 找到轮廓最左坐标，宽高
            #     x, y, w, h = cv2.boundingRect(val)
            #     contours_lis.append([x, y, w, h])
            #     # cv2.rectangle(img, (x, y),
            #     #               (x + w, y + h),
            #     #               (0, 0, 255), 3)
            #     # cv2.putText(img, '1', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #     #             0.5,
            #     #             (255, 0, 0), 1)
            # print(type(contours))
            similar_thresh = 0.02
            area_thresh = 3000
            for index, contour in enumerate(contours):
                similar = []
                if cv2.contourArea(contour) < area_thresh:
                    continue
                for module_index, module_contour in enumerate(module_contours):
                    similar_one = cv2.matchShapes(contour, module_contour, cv2.CONTOURS_MATCH_I1, 0)
                    # 计算Hu矩
                    # 计算contour1和contour2的边界框
                    # bbox1 = measure.regionprops(contour.astype(int))[0].bbox
                    # bbox2 = measure.regionprops(module_contour.astype(int))[0].bbox
                    #
                    # # 计算contour1和contour2的宽度和高度
                    # width1 = bbox1[3] - bbox1[1]
                    # height1 = bbox1[2] - bbox1[0]
                    # width2 = bbox2[3] - bbox2[1]
                    # height2 = bbox2[2] - bbox2[0]
                    #
                    # # 将contour1和contour2缩放到相同的尺寸
                    # contour = measure.rescale(contour, (height2 / height1, width2 / width1))
                    # module_contour = module_contour

                    # moments1 = cv2.moments(contour)
                    # hu_moments1 = cv2.HuMoments(moments1)
                    #
                    # moments2 = cv2.moments(module_contour)
                    # hu_moments2 = cv2.HuMoments(moments2)
                    # # 归一化处理
                    # hu_moments1_normalized = -np.sign(hu_moments1) * np.log10(np.abs(hu_moments1))
                    # hu_moments2_normalized = -np.sign(hu_moments2) * np.log10(np.abs(hu_moments2))
                    #
                    # # 计算归一化后的Hu矩之间的差异
                    # similar_one = np.sum(np.abs(hu_moments1_normalized - hu_moments2_normalized))
                    # # 比较Hu矩
                    # # hu_diff = cv2.compareHu(hu_moments1, hu_moments2)
                    # print("hu:", similar_one)
                    # similarity = measure.compare_shapes(contour, contour)

                    # 输出形状相似性
                    # print("形状相似性：", similarity)
                    similar.append(similar_one)
                # print(similar)
                # max_i = max(similar)
                # # min_i = min(similar)
                # for i in range(len(similar)):
                #     similar[i] = similar[i] / max_i
                min_similar = min(similar)
                min_index = similar.index(min_similar)
                print(similar)
                approx = cv2.approxPolyDP(contour, 0.015 * cv2.arcLength(contour, True), True)
                sides = len(approx)
                print(sides)
                if min_similar < similar_thresh:
                    x, y, w, h = cv2.boundingRect(contour)
                    if (module_name[min_index] == "triangle") and (sides != 3):
                        if sides == 10:
                            min_index = 1
                        elif sides == 4:
                            min_index = 3
                        elif sides == 7:
                            min_index = 4
                    elif module_name[min_index] == "rectangle" and sides != 4:
                        if sides == 3:
                            min_index = 2
                        elif sides == 10:
                            min_index = 1
                        elif sides == 7:
                            min_index = 4
                    elif module_name[min_index] == "arrow" and sides != 7:
                        if sides == 3:
                            min_index = 2
                        elif sides == 4:
                            min_index = 3
                        elif sides == 10:
                            min_index = 1
                    elif module_name[min_index] == "star" and sides != 10:
                        if sides == 3:
                            min_index = 2
                        elif sides == 4:
                            min_index = 3
                        elif sides == 7:
                            min_index = 4
                    # print((x, y, w, h))
                    cv2.rectangle(img, (x, y),
                                  (x + w, y + h),
                                  (0, 0, 255), 3)
                    cv2.putText(img, module_name[min_index], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0), 1)
                    txt_name = "labels/shape_match/" + dir_item.path[9:]
                    txt_name = txt_name.replace("jpg", "txt")
                    # print(txt_name)
                    yolo_txt = [str(min_index) + " ", str((x + w / 2) / width) + " ", str((y + h / 2) / height) + " ",
                                str(w / width) + " ", str(h / height) + "\n"]
                    with open(txt_name, mode="a", encoding="utf-8") as f:
                        for i in yolo_txt:
                            f.write(i)
                else:
                    if sides == 3:
                        min_index = 2
                    elif sides == 4:
                        min_index = 3
                    elif sides == 7:
                        min_index = 4
                    elif sides == 10:
                        min_index = 1
                    else:
                        min_index = 6
                    x, y, w, h = cv2.boundingRect(contour)
                    # print((x, y, w, h))
                    cv2.rectangle(img, (x, y),
                                  (x + w, y + h),
                                  (0, 0, 255), 3)
                    cv2.putText(img, module_name[min_index], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0), 1)
                    txt_name = "labels/shape_match/" + dir_item.path[9:]
                    txt_name = txt_name.replace("jpg", "txt")
                    # print(txt_name)
                    yolo_txt = [str(min_index) + " ", str((x + w / 2) / width) + " ", str((y + h / 2) / height) + " ",
                                str(w / width) + " ", str(h / height) + "\n"]
                    with open(txt_name, mode="a", encoding="utf-8") as f:
                        for i in yolo_txt:
                            f.write(i)
            self.pic_show(img)
        with open("./labels/shape_match/classes.txt", mode="w", encoding="utf-8") as f:
            for i in module_name:
                f.write(i)
                f.write("\n")
    # def shape_match(self, img):
    #     # 灰度化
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     ret, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    #     # self.pic_show(binary)
    #     contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # 滤波腐蚀膨胀
    #     # to do
    #     # 坐标和宽高信息
    #     contours_lis = []
    #     for i, val in enumerate(contours):
    #         # 找到轮廓最左坐标，宽高
    #         x, y, w, h = cv2.boundingRect(val)
    #         contours_lis.append([x, y, w, h])
    #         # cv2.rectangle(img, (x, y),
    #         #               (x + w, y + h),
    #         #               (0, 0, 255), 3)
    #         # cv2.putText(img, '1', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
    #         #             0.5,
    #         #             (255, 0, 0), 1)
    #     # print(type(contours))
    #     similar_thresh = 0.02
    #     for i in range(len(contours)):
    #         for index, h in enumerate(contours):
    #             similar = cv2.matchShapes(contours[i], h, cv2.CONTOURS_MATCH_I1, 0)
    #             # print(similar)
    #             if similar >= similar_thresh:
    #                 continue
    #             if similar < similar_thresh:
    #                 x, y, w, h = cv2.boundingRect(contours[index])
    #                 # print((x, y, w, h))
    #                 cv2.rectangle(img, (x, y),
    #                               (x + w, y + h),
    #                               (0, 0, 255), 3)
    #                 cv2.putText(img, '1', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
    #                             0.5,
    #                             (255, 0, 0), 1)
    #             # else:
    #             #     x, y, w, h = cv2.boundingRect(contour)
    #             #     # print((x, y, w, h))
    #             #     cv2.rectangle(img, (x, y),
    #             #                   (x + w, y + h),
    #             #                   (0, 0, 255), 3)
    #             #     cv2.putText(img, module_name[6], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
    #             #                 0.5,
    #             #                 (255, 0, 0), 1)
    #             self.pic_show(img)

    def color_match(self, dir_path):
        # img = cv2.imread(p_path)
        # 读取图片信息
        # 图片hsv
        color_str = ['black', 'gray', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
        for dir_item in os.scandir(dir_path):
            img = cv2.imread(dir_item.path)
            height, width, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 滤波腐蚀膨胀
            # 图片二值化
            ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            # contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 用于存储类型，坐标信息
            class_position_lis = []
            # kernel = np.ones((20, 20), np.uint8)
            # eroded = cv2.erode(img, kernel, iterations=1)
            # img = cv2.dilate(eroded, kernel, iterations=1)
            # self.pic_show(img)
            # img = cv2.GaussianBlur(img, (5, 5), 0)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # self.pic_show(img_hsv)
            # print(img_hsv)
            # for i, val in enumerate(contours):
            #     # 找到轮廓最左坐标，宽高
            #     x, y, w, h = cv2.boundingRect(val)
            #     contours_lis.append([x, y, w, h])
            # print(contours_lis)
            low = np.array([
                [0, 0, 0],
                [0, 0, 46],
                # [0, 0, 221],
                [156, 43, 46],
                # [0, 43, 46],
                [11, 43, 46],
                [26, 43, 46],
                [35, 43, 46],
                [78, 43, 46],
                [100, 43, 46],
                [125, 43, 46]
            ])
            high = np.array([
                [180, 255, 46],
                [180, 43, 220],
                # [180, 30, 255],
                [180, 255, 255],
                [25, 255, 255],
                [34, 255, 255],
                [77, 255, 255],
                [99, 255, 255],
                [124, 255, 255],
                [155, 255, 255]
            ])
            color_code = 0  # 给颜色一个编号，方便写入YOLO格式
            area_thresh = 3000
            for j in range(9):
                mask = cv2.inRange(img_hsv, low[j], high[j])
                # mask = cv2.GaussianBlur(mask, (5, 5), 0)
                c, hie = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # self.pic_show(c[0])
                # print(type(c))
                if not c:
                    continue

                for i, contour in enumerate(c):
                    area = cv2.contourArea(contour)
                    print(area)
                    if area < area_thresh:
                        continue
                    x, y, w, h = cv2.boundingRect(c[i])
                    class_position_lis.append(
                        [str(color_code) + " ", str((x + w / 2) / width) + " ", str((y + h / 2) / height) + " ",
                         str(w / width) + " ", str(h / height) + "\n"])
                    cv2.rectangle(img, (x, y),
                                  (x + w, y + h),
                                  (0, 0, 255), 3)
                    cv2.putText(img, color_str[j], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0), 1)
                color_code += 1  # 颜色类别加1，进入下一次循环判断
            print(class_position_lis)
            txt_name = "labels/color_match/" + dir_item.path[9:]
            txt_name = txt_name.replace("jpg", "txt")
            # print(txt_name)

            with open(txt_name, mode="a", encoding="utf-8") as f:
                for lis in class_position_lis:
                    for i in lis:
                        f.write(i)
            self.pic_show(img)
            # print(class_position_lis)
        with open("./labels/color_match/classes.txt", mode="w", encoding="utf-8") as f:
            for i in color_str:
                f.write(i)
                f.write("\n")

'''
    def color_match(self, img):
            # 图片灰度化
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 图片二值化
            ret, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

            # 腐蚀和膨胀操作
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)

            # 查找轮廓
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 坐标和宽高信息
            contours_lis = []
            for i, val in enumerate(contours):
                # 找到轮廓最左坐标，宽高
                x, y, w, h = cv2.boundingRect(val)
                contours_lis.append([x, y, w, h])
            print(contours_lis)

            low = np.array([
                [0, 0, 0],
                [0, 0, 46],
                # [0, 0, 221],
                [0, 43, 46],
                [11, 43, 46],
                [26, 43, 46],
                [35, 43, 46],
                [78, 43, 46],
                [100, 43, 46],
                [125, 43, 46]
            ])
            high = np.array([
                [180, 255, 46],
                [180, 43, 220],
                # [180, 30, 255],
                [10, 255, 255],
                [25, 255, 255],
                [34, 255, 255],
                [77, 255, 255],
                [99, 255, 255],
                [124, 255, 255],
                [155, 255, 255]
            ])
            color_str = ['black', 'gray', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
            for j in range(9):
                mask = cv2.inRange(img_hsv, low[j], high[j])
                c, hie = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not c:
                    continue

                for i in range(len(c)):
                    x, y, w, h = cv2.boundingRect(c[i])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(img, color_str[j], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    self.pic_show(img)'''

'''
        for i, contours_simple in enumerate(contours_lis):
            print(i)
            center_x = (contours_simple[0] + contours_simple[2]) // 2
            center_y = (contours_simple[1] + contours_simple[3]) // 2
            # h, s, v = cv2.split(img_hsv)
            # print(img_hsv[center_y, center_x, 0])
            # 黑色识别
            if (0 <= img_hsv[center_y, center_x, 0] and img_hsv[center_y, center_x, 0] <= 180) and (
                    0 <= img_hsv[center_y, center_x, 1] and img_hsv[center_y, center_x, 1] <= 255) and (
                    0 <= img_hsv[center_y, center_x, 2] and img_hsv[center_y, center_x, 2] <= 46):
                cv2.rectangle(img, (contours_lis[i][0], contours_lis[i][1]),
                              (contours_lis[i][0] + contours_lis[i][2], contours_lis[i][1] + contours_lis[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, 'Black', (contours_lis[i][0] + 10, contours_lis[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            # 灰色
            if (0 <= img_hsv[center_y, center_x, 0] and img_hsv[center_y, center_x, 0] <= 180) and (
                    0 <= img_hsv[center_y, center_x, 1] and img_hsv[center_y, center_x, 1] <= 43) and (
                    46 <= img_hsv[center_y, center_x, 2] and img_hsv[center_y, center_x, 2] <= 220):
                cv2.rectangle(img, (contours_lis[i][0], contours_lis[i][1]),
                              (contours_lis[i][0] + contours_lis[i][2], contours_lis[i][1] + contours_lis[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, 'Black', (contours_lis[i][0] + 10, contours_lis[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            #
            if (0 <= img_hsv[center_y, center_x, 0] and img_hsv[center_y, center_x, 0] <= 180) and (
                    0 <= img_hsv[center_y, center_x, 1] and img_hsv[center_y, center_x, 1] <= 255) and (
                    0 <= img_hsv[center_y, center_x, 2] and img_hsv[center_y, center_x, 2] <= 46):
                cv2.rectangle(img, (contours_lis[i][0], contours_lis[i][1]),
                              (contours_lis[i][0] + contours_lis[i][2], contours_lis[i][1] + contours_lis[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, 'Black', (contours_lis[i][0] + 10, contours_lis[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            # 红色
            if (0 <= img_hsv[center_y, center_x, 0] and img_hsv[center_y, center_x, 0] <= 10) and (
                    43 <= img_hsv[center_y, center_x, 1] and img_hsv[center_y, center_x, 1] <= 255) and (
                    46 <= img_hsv[center_y, center_x, 2] and img_hsv[center_y, center_x, 2] <= 255):
                cv2.rectangle(img, (contours_lis[i][0], contours_lis[i][1]),
                              (contours_lis[i][0] + contours_lis[i][2], contours_lis[i][1] + contours_lis[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, 'Black', (contours_lis[i][0] + 10, contours_lis[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            # 橙色
            if (11 <= img_hsv[center_y, center_x, 0] and img_hsv[center_y, center_x, 0] <= 25) and (
                    43 <= img_hsv[center_y, center_x, 1] and img_hsv[center_y, center_x, 1] <= 255) and (
                    46 <= img_hsv[center_y, center_x, 2] and img_hsv[center_y, center_x, 2] <= 255):
                cv2.rectangle(img, (contours_lis[i][0], contours_lis[i][1]),
                              (contours_lis[i][0] + contours_lis[i][2], contours_lis[i][1] + contours_lis[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, 'Black', (contours_lis[i][0] + 10, contours_lis[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            # 黄色
            if (26 <= img_hsv[center_y, center_x, 0] and img_hsv[center_y, center_x, 0] <= 34) and (
                    43 <= img_hsv[center_y, center_x, 1] and img_hsv[center_y, center_x, 1] <= 255) and (
                    46 <= img_hsv[center_y, center_x, 2] and img_hsv[center_y, center_x, 2] <= 255):
                cv2.rectangle(img, (contours_lis[i][0], contours_lis[i][1]),
                              (contours_lis[i][0] + contours_lis[i][2], contours_lis[i][1] + contours_lis[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, 'Black', (contours_lis[i][0] + 10, contours_lis[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            # 绿色
            if (35 <= img_hsv[center_y, center_x, 0] and img_hsv[center_y, center_x, 0] <= 77) and (
                    43 <= img_hsv[center_y, center_x, 1] and img_hsv[center_y, center_x, 1] <= 255) and (
                    46 <= img_hsv[center_y, center_x, 2] and img_hsv[center_y, center_x, 2] <= 255):
                cv2.rectangle(img, (contours_lis[i][0], contours_lis[i][1]),
                              (contours_lis[i][0] + contours_lis[i][2], contours_lis[i][1] + contours_lis[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, 'Black', (contours_lis[i][0] + 10, contours_lis[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            # 青色
            if (78 <= img_hsv[center_y, center_x, 0] and img_hsv[center_y, center_x, 0] <= 99) and (
                    43 <= img_hsv[center_y, center_x, 1] and img_hsv[center_y, center_x, 1] <= 255) and (
                    46 <= img_hsv[center_y, center_x, 2] and img_hsv[center_y, center_x, 2] <= 255):
                cv2.rectangle(img, (contours_lis[i][0], contours_lis[i][1]),
                              (contours_lis[i][0] + contours_lis[i][2], contours_lis[i][1] + contours_lis[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, 'Black', (contours_lis[i][0] + 10, contours_lis[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            # 蓝色
            if (100 <= img_hsv[center_y, center_x, 0] and img_hsv[center_y, center_x, 0] <= 124) and (
                    43 <= img_hsv[center_y, center_x, 1] and img_hsv[center_y, center_x, 1] <= 255) and (
                    46 <= img_hsv[center_y, center_x, 2] and img_hsv[center_y, center_x, 2] <= 255):
                cv2.rectangle(img, (contours_lis[i][0], contours_lis[i][1]),
                              (contours_lis[i][0] + contours_lis[i][2], contours_lis[i][1] + contours_lis[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, 'Black', (contours_lis[i][0] + 10, contours_lis[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            # 紫色
            if (125 <= img_hsv[center_y, center_x, 0] and img_hsv[center_y, center_x, 0] <= 155) and (
                    43 <= img_hsv[center_y, center_x, 1] and img_hsv[center_y, center_x, 1] <= 255) and (
                    46 <= img_hsv[center_y, center_x, 2] and img_hsv[center_y, center_x, 2] <= 255):
                cv2.rectangle(img, (contours_lis[i][0], contours_lis[i][1]),
                              (contours_lis[i][0] + contours_lis[i][2], contours_lis[i][1] + contours_lis[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, 'Black', (contours_lis[i][0] + 10, contours_lis[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            self.pic_show(img)
            '''

if __name__ == "__main__":
    pass
