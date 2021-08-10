import scipy
import cv2
import numpy as np
import matplotlib.animation as animation

from numpy.fft.helper import fftfreq
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = 'ipcam03-2021-07-12_15-00.mp4' # run
    # data = 'ipcam03-2021-07-12_19-30.mp4' # stop
    cap = cv2.VideoCapture(data)
    
    # select roi
    _, first_frame = cap.read()
    x, y, w, h = cv2.selectROI('origin', first_frame, False)

    # plot figure
    plt.ion()
    figure, ax = plt.subplots()

    plt.xlim(0, 15, 3)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Fast Fourier Transform')
    plt.grid(True)

    # check roi
    if x != 0 and y != 0:
        first_frame = first_frame[y:y+h, x:x+w]

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # Creates an image filled with zero
    # intensities with the same dimensions as the frame
    mask = np.zeros_like(first_frame)
    
    # Sets image saturation to maximum
    mask[..., 1] = 255

    # for plot    
    first_step = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        draw = frame.copy() # for drawing

        if x != 0 and y != 0:
            frame = frame[y:y+h, x:x+w] # cropping

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
        
        # Opens a new window and displays the output frame
        cv2.imshow("dense optical flow", rgb)
        
        # Updates previous frame
        prev_gray = gray
        
        # FFT analysis
        sig = magnitude.reshape(-1)

        yf = scipy.fft(sig) / len(sig)
        yf = np.abs(yf)

        xf = fftfreq(w*h, 1 / 30)
        
        # Drawing matplot
        if first_step:
            line, = ax.plot(xf, yf)
            first_step = False
            continue

        line.set_xdata(xf)
        line.set_ydata(yf)

        figure.canvas.draw()
        figure.canvas.flush_events()
        
        # Drawing ROI
        if h == 0:
            cv2.circle(draw, (x,y),2, (255,0,0), 4) # point
        else:
            cv2.rectangle(draw, (x,y), (x+w, y+h), (255,0,0), 2) # area

        cv2.imshow('origin', draw)
        if cv2.waitKey(5) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()


# reference 
# - fourier transform
# https://realpython.com/python-scipy-fft/#scipyfft-vs-scipyfftpack
#
# - matplot
# https://www.delftstack.com/ko/howto/matplotlib/how-to-automate-plot-updates-in-matplotlib/