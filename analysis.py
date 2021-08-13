import cv2
import csv
import os
import scipy
import numpy as np

from numpy.fft.helper import fftfreq
from matplotlib import pyplot as plt

FPS  = 30

class video_analysis(object):
    def __init__(self, path):
        self.data_path = path
        self.cap = ""
        self.roi = ""
        self.dir = "./data"

        self.setROI()

    def setROI(self):
        self.cap = cv2.VideoCapture(self.data_path)
        ret, img = self.cap.read()
        if ret is False:
            return

        self.roi = cv2.selectROI('vibration analysis based video', img, False)
        if self.roi is None:
            print('---- error ----')
            return

    def fourier_transform(self, magnitude):
        sig = magnitude.reshape(-1)

        yf = scipy.fft(sig) / len(sig)
        yf = np.abs(yf)
        xf = fftfreq(self.roi[2] * self.roi[3], 1 / FPS) # w*h

        return xf, yf

    def save_csv(self, data):
        # Make diretory if not exist
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        print("---- Write csv file ----")
        csv_path = os.path.join(self.dir, self.data_path.split('.')[0]+'.csv')
        f = open(csv_path, 'w', encoding='utf-8')
        w = csv.writer(f)
        for i in data:
            w.writerow(i)
        f.close()
            

    def opticalflow_dense(self):
        x, y, w, h = self.roi[0], self.roi[1], self.roi[2], self.roi[3]

        # For Drawing
        plt.ion()
        figure, ax = plt.subplots()

        plt.xlim(0, 15, 3)
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Fast Fourier Transform')
        plt.grid(True)

        # First frame
        _, frame = self.cap.read()
        crop = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Make mask for drawing 
        mask = np.zeros_like(crop)

        # Sets image saturation to maximum
        mask[..., 1] = 255

        fft_data = []

        first_frame = True
        while True:
            ret, frame = self.cap.read()
            if ret is False:
                print('End of videocapture')
                break

            # Croping frame : [y:y+h, x:x+w]
            crop = frame[y:y+h, x:x+w]

            prev_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
            # Calculates dense optical flow by Farneback method
            # The result 2-channel array of flow vectors (dx/dt, dy/dt)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Sets image hue according to the optical flow direction
            mask[..., 0] = angle * 180 / np.pi / 2
            
            # Sets image value according to the optical flow
            # magnitude (normalized)
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            # Converts HSV to RGB (BGR) color representation
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            
            # Updates previous frame
            prev_gray = gray

            # FFT
            xf, yf = self.fourier_transform(magnitude)

            print(xf.shape)            
            fft_data.append(xf)

            if first_frame is True:
                line, = ax.plot(xf, yf)
                first_frame = False
                continue

            line.set_xdata(xf)
            line.set_ydata(yf)

            figure.canvas.draw()
            figure.canvas.flush_events()

            cv2.rectangle(frame, (x,y),(x+w, y+h), (255,0,0), 2)
            cv2.imshow('dense optical flow', rgb)
            cv2.imshow('vibration analysis based video', frame)
            if cv2.waitKey(1) & 0xFF == 27: 
                # self.save_csv(fft_data)
                break

        self.cap.release()
        cv2.destroyAllWindows()

        # Save data
        # self.save_csv(fft_data)
        

# reference 
# - fourier transform
# https://realpython.com/python-scipy-fft/#scipyfft-vs-scipyfftpack
# https://underflow101.tistory.com/26

# - matplot
# https://www.delftstack.com/ko/howto/matplotlib/how-to-automate-plot-updates-in-matplotlib/
#
# - paper
# http://assets.ksnve.or.kr/mail_form/file/contents3001_102.pdf
#
# - anomal detection
# https://hoya012.github.io/blog/anomaly-detection-overview-1/
# 
# - lstm ae
# https://velog.io/@jonghne/LSTM-AE%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%8B%9C%EA%B3%84%EC%97%B4-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9D%B4%EC%83%81-%ED%83%90%EC%A7%80-1-%EA%B0%9C%EC%9A%94