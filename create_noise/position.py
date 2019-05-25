# -*- coding: utf-8 -*-

import cv2
import glob
import os
import csv

class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):

        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent

    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])


if __name__ == "__main__":

    images = glob.glob('./sample/*.jpg')
    positions = {}

    for image in images:
        image_name = os.path.basename(image)
        position = []
        #入力画像
        read = cv2.imread(image)
        # ウィンドウのサイズを変更可能にする
        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        #表示するWindow名
        window_name = image_name

        #画像の表示
        cv2.imshow(image_name, read)

        #コールバックの設定
        mouseData = mouseParam(image_name)
        print('Start {}'.format(image_name))
        while 1:
            cv2.waitKey(20)
            #左クリックがあったら表示
            if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
                print(mouseData.getPos())
                position.append(mouseData.getPos())

            #右クリックがあったら終了
            elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
                print('Finished {}'.format(image_name))
                position = list(set(position))
                positions.update({image_name : position})
                cv2.destroyAllWindows()
                break

    print("Finished")
    print(positions)
    keys = positions.keys()

    with open('position.txt', 'w') as f:
        for key in keys:
            data = positions[key]
            f.write(key)
            for i in range(len(data)):
                d = data[i]
                f.write(',' + str(d[0]) + '_' + str(d[1]))
            f.write('\n')
