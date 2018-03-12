import tkinter as tk
import cv2
import numpy as np
import random
from tkinter import Menu

win = tk.Tk()
win.title("Python GUI")
numFrame = 1

def _captureVideo():
    capature = cv2.VideoCapture(0)
    out = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (int(capature.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5), int(capature.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)))
    while(True):
        ret, frame = capature.read()
        out.write(frame)
        cv2.imshow("frame", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    fps = capature.get(cv2.CAP_PROP_FPS);
    var = 'FPS: ' + str(1/fps)
    out.release()
    capature.release()
    cv2.destroyAllWindows()
    t = tk.Text(win)
    t.insert('insert',var)
    t.pack()

def _selectFrame():
    capature = cv2.VideoCapture('test.mp4')
    ret, frame = capature.read()
    global numFrame
    numFrame = 1
    while(ret):
        cv2.imshow("frame", frame)
        if cv2.waitKey(10) & 0xFF == ord('s'):
            cv2.imwrite("test_%d.png" %numFrame, frame)
            numFrame += 1
        ret, frame = capature.read()
    capature.release()
    cv2.destroyAllWindows()

def _showFrame():
    for i in range(1, numFrame):
        imgSHOW = cv2.imread("test_%d.png" %i)
        imgR = cv2.resize(imgSHOW, (800, 450))
        cv2.imshow("Original Picture_%d"%i, imgR)
    if (cv2.waitKey(0) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()

def _bgrTohsv():
    for i in range(1, numFrame):
        img = cv2.imread("test_%d.png" %i)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite("test_hsv_%d.png" %i, imgHSV)
        imgSHOWHSV = cv2.imread("test_hsv_%d.png" %i)
        imgHSVR = cv2.resize(imgSHOWHSV, (800, 450))
        cv2.imshow("HSV Picture_%d"%i, imgHSVR)
    if (cv2.waitKey(0) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()

def _bgrTohsv2():
        img = cv2.imread("test_1.png", -1)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                b, g, r = img[i][j][0] / 255, img[i][j][1] / 255, img[i][j][2] / 255,
                maxi = max(r, g, b)
                mini = min(r, g, b)
                diff = maxi - mini
                if maxi == mini:
                    h = 0
                elif maxi == r:
                    #h = ((60 * (g - b) / diff) % 6 ) % 360
                    h = (60 * ((g - b) / diff) + 360) % 360
                elif maxi == g:
                    h = (60 * ((b - r) / diff) + 120) % 360
                elif maxi == b:
                    h = (60 * ((r - g) / diff) + 240) % 360
                if maxi == 0:
                    s = 0
                else:
                    s = diff/maxi
                v = maxi
                img[i][j][0] = np.round(h / 2)
                img[i][j][1] = np.round(255 * s)
                img[i][j][2] = np.round(255 * v)
        cv2.imwrite("test_hsv2.jpg", img)

def _traceObject():
    sum = 0
    for i in range(1, numFrame):
        lower = np.array([5, 150, 46])
        upper = np.array([28, 255, 255])
        t1= cv2.getTickCount()
        image = cv2.imread("test_hsv_%d.png" %i)
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)
        cv2.imwrite("test_bw_%d.png" %i, mask)
        t2 = cv2.getTickCount()
        maskR = cv2.resize(mask, (800, 450))
        cv2.imshow("TRACE_%d"%i, maskR)
        t=(t2-t1)/cv2.getTickFrequency()
        sum += t
    var = 'Average time for detecting the object: ' + str(sum/numFrame)
    t = tk.Text(win)
    t.insert('insert',var)
    t.pack()
    if (cv2.waitKey(0) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()

def _makeNoise():
    image = cv2.imread('test_1.png')
    output = np.zeros(image.shape,np.uint8)
    prob = 0.05
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    cv2.imwrite('test_noise.png', output)
    imgNOISE = cv2.imread("test_noise.png")
    imgNOISER = cv2.resize(imgNOISE, (800, 450))
    cv2.imshow('Noise', imgNOISER)
    if (cv2.waitKey(0) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()

def _removeNoise():
    image = cv2.imread('test_noise.png')
    median = cv2.medianBlur(image,5)
    cv2.imwrite('test_smooth1.jpg', median)
    imgRemoveNoiseR = cv2.resize(median, (800, 450))
    cv2.imshow('Median Smooth', imgRemoveNoiseR)
    if (cv2.waitKey(0) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()

def _realtimeTrack():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    ret = True
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([5, 150, 46])
        upper = np.array([28, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        cv2.imshow('frame',mask)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def _quit():
    """Exit the mainloop"""
    win.quit()
    win.destroy()
    exit()

def main():
    menuBar = Menu(win)
    win.config(menu=menuBar)
    VideoCapture = Menu(menuBar, tearoff=0)
    menuBar.add_cascade(label="Video", menu=VideoCapture)
    VideoCapture.add_separator()
    VideoCapture.add_command(label="Capture a Video",command=_captureVideo)
    VideoCapture.add_separator()
    VideoCapture.add_command(label="Real-Time Tracking",command=_realtimeTrack)
    VideoCapture.add_separator()
    VideoCapture.add_command(label="Exit", command=_quit)

    FrameSelect = Menu(menuBar, tearoff=0)
    menuBar.add_cascade(label="Frames", menu=FrameSelect)
    FrameSelect.add_command(label="Select the Frames", command=_selectFrame)
    FrameSelect.add_command(label="Show the Frames", command=_showFrame)

    ConvertHSV = Menu(menuBar, tearoff=0)
    menuBar.add_cascade(label="HSV", menu=ConvertHSV)
    ConvertHSV.add_command(label="BGR to HSV", command=_bgrTohsv)
    ConvertHSV.add_separator()
    ConvertHSV.add_command(label="BGR to HSV2",command=_bgrTohsv2)

    TraceBall = Menu(menuBar, tearoff=0)
    menuBar.add_cascade(label="Trace", menu=TraceBall)
    TraceBall.add_command(label="Trace an Object", command = _traceObject)

    MakeNoise = Menu(menuBar, tearoff=0)
    menuBar.add_cascade(label="Noises", menu=MakeNoise)
    MakeNoise.add_command(label="Make Noise", command = _makeNoise)
    MakeNoise.add_separator()
    MakeNoise.add_command(label="Remove Noise1", command = _removeNoise)


    win.mainloop()

if __name__ == "__main__":
    main()
