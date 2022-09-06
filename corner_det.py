#!/usr/bin/env python3
import numpy as np
import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1920, 1080)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)

def empty(a): #takes 1 argument
   pass

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    # create a window for the trackbar
    cv2.namedWindow("TrackBars") #create a new window named "TrackBars"
    cv2.resizeWindow("TrackBars", 640, 240)

    cv2.createTrackbar("Harris threshold", "TrackBars", 1, 100, empty)

    while True:
        videoIn = video.get()
        frame = videoIn.getCvFrame()

        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        cv2.imshow("video", frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        cv2.imshow("video gray", gray)

        # find Harris corners
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)

        harris_threshold = cv2.getTrackbarPos("Harris threshold", "TrackBars")

        frame[dst>(harris_threshold/100)*dst.max()]=[0,0,255]

        # this one is for subpixel corner detection
        # ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        # dst = np.uint8(dst)

        # # find centroids
        # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # # define the criteria to stop and refine the corners
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        # # Now draw them
        # res = np.hstack((centroids,corners))
        # res = np.int0(res)

        # try:
        #     frame[res[:,1],res[:,0]]=[0,0,255]
        #     frame[res[:,3],res[:,2]] = [0,255,0]
        # except: 
        #     continue

        cv2.imshow("preview",frame)

        if cv2.waitKey(1) == ord('q'):
            break
