import sys

import cv2

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import  QImage,QPixmap
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import QDir, Qt, QUrl
import time
import random

class detectCarSystem(QDialog):

    def __init__(self):
        super(detectCarSystem,self).__init__()
        loadUi('maingui.ui',self)
        self.image=None
        self.processedImage=None
        self.loadBut.clicked.connect(self.loadClicked)
        self.exitBut.clicked.connect(self.exitClicked)


    @pyqtSlot()
    def exitClicked(self):
        sys.exit(app.exec_())

    @pyqtSlot()
    def loadClicked(self):
        try:
            fileName, _ = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\', "Video Files(*.avi)")
            car_cascade = cv2.CascadeClassifier('cars.xml')
            vc = cv2.VideoCapture(fileName)
            try:
                if vc.isOpened():
                    rval, frame = vc.read()
                else:
                    rval = False



                max_speed = 45;

                if vc.isOpened():
                    rval, frame = vc.read()
                else:
                    rval = False

                length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
                print("Frame Count: {0}".format(length))

                width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
                print("Video Frame Width: {0}".format(width))

                height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print("Video Frame Height: {0}".format(height))

                # tiime start
                start_time = time.time()

                video_width = int(vc.get(3))
                video_height = int(vc.get(4))
                fr = int(vc.get(5))
                print("Fps:", fr)

                counter = 0

                value = 0;
                while rval:
                    rval, frame = vc.read()

                    # car detection.
                    cars = car_cascade.detectMultiScale(frame, 1.2, 2)

                    ncars = 0
                    fps_value = 0
                    pp = 1
                    value = length / fr;

                    for (x, y, w, h) in cars:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        # End time
                        end_time = time.time()
                        value = (length / fr) / (end_time - start_time) * 10 + r_value;
                        counter += 1
                        if end_time - start_time > pp:
                            fps_value = counter / (end_time - start_time);
                            counter = 0

                        diff_fps = end_time - start_time
                        speed_value = diff_fps * fps_value * value;


                        text_sample = str(round(speed_value)) + " km/h"
                        # text_sample=str(round(value)) + " km/h";

                        if speed_value > max_speed:
                            cv2.putText(frame, text_sample, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 0, 0))
                        else:
                            cv2.putText(frame, text_sample, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (255, 255, 255))

                        ncars = ncars + 1

                    # show result
                    cv2.imshow("Result", frame);
                    cv2.waitKey(1);

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    if cv2.waitKey(1) & 0xFF == ord('p'):
                        while True:
                            key2=cv2.waitKey(1) or 0xff
                            cv2.imshow("Result", frame);
                            if key2==ord('p'):
                                break

                vc.release()
            except cv2.error as e:
                print('Finished')
        except RuntimeError:
            print('Finished.')

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = detectCarSystem()
    window.setWindowTitle('Moving Vehicles Tracking and Speed Detection from Video')
    window.show()
    sys.exit(app.exec_())