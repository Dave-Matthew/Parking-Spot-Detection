from traceback import FrameSummary
import cv2
import depthai as dai
import numpy as np
import time
import blobconverter
import math

def remove_small_objects(img, min_size=150):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    img2 = img
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            img2[output == i + 1] = 0

    return img2 

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area>15000: #to get rid of outliar data maybe a hole in the table, etc
            cv2.drawContours(frame, cnt, -1, (255,0,0), 3)
            perimeter = cv2.arcLength(cnt,True)
            #print(perimeter)
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True) #get every point of corners into an array
            
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0)) #create a bounding rectangle
            
            if objCor == 4: 
                print(approx)
                cv2.putText(frame, "Parking Spot", (x + (w//2) - 10, y + (h//2) - 10),cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)

def empty(a): #takes 1 argument
   pass

def getHoughLines(src):

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    # Standard Hough Line Transform
    lines = cv2.HoughLines(src, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)
    
    # # Probabilistic Line Transform
    # linesP = cv2.HoughLinesP(src, 1, np.pi / 180, 50, None, 50, 10)
    
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         #getFullLine(l, cdstP)
    #         #getFullLineV2(l, cdstP)
    #         cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
    
    # cv2.imshow("Source", src)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    return cdst



########### MAIN FUNCTION ##############
# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1280, 720)
camRgb.setFps(60)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(4)

# Linking
camRgb.video.link(xoutVideo.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    startTime = time.monotonic()
    counter = 0
    fps = 0

    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    cv2.namedWindow("TrackBars") #create a new window named "TrackBars"
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty) # from 0 until 179 is the whole range of hue
    cv2.createTrackbar("Hue Max", "TrackBars", 120, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 85, 255, empty) # from 0 until 255 is the whole range of saturation
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 136, 255, empty) # from 0 until 255 is the whole range of value(colors)
    cv2.createTrackbar("Val Max", "TrackBars", 252, 255, empty)
    cv2.createTrackbar("Thres Min", "TrackBars", 150, 1000, empty)
    cv2.createTrackbar("Thres Max", "TrackBars", 550, 1000, empty)
    cv2.createTrackbar("Harris threshold", "TrackBars", 1, 100, empty)
    cv2.createTrackbar("Max corner", "TrackBars", 1, 100, empty)
    cv2.createTrackbar("Quality level", "TrackBars", 1, 1000, empty)
    cv2.createTrackbar("Min distance", "TrackBars", 10, 1000, empty)


    while True:
        videoIn = video.get()
        frame = videoIn.getCvFrame()
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1) #source image, kernel, sigma(the bigger the sigma the more blur)
        
        
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        t_min = cv2.getTrackbarPos("Thres Min", "TrackBars")
        t_max = cv2.getTrackbarPos("Thres Max", "TrackBars")
        #harris_threshold = cv2.getTrackbarPos("Harris threshold", "TrackBars")
        max_corner = cv2.getTrackbarPos("Max corner", "TrackBars")
        q_level = cv2.getTrackbarPos("Quality level", "TrackBars")
        min_dis = cv2.getTrackbarPos("Min distance", "TrackBars")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(frame, lower, upper)
        imgResult = cv2.bitwise_and(frame, frame, mask = mask) #to merge img to the img with mask named masked
        imgCanny = cv2.Canny(imgResult, t_max, t_min, apertureSize = 5)

        imgHoughLine = getHoughLines(imgCanny)

        ## getting the corner
        corners = cv2.goodFeaturesToTrack(imgGray, max_corner, (q_level/1000), min_dis, mask= mask )
        
        # print(corners)
        # if corners is not None:
        #     corners = np.int0(corners)
        #     for i in corners:
        #         x,y = i.ravel()
        #         print(x, " ", y)
        #         cv2.circle(imgCanny,(x,y),3,(0,0,255),-1)
        if corners is not None:       
            corners = np.int0(corners)
            for i in corners:
                x,y = i.ravel()
                #print(x, " ", y)
                #print(imgResult[y,x])
                cv2.circle(imgResult,(x,y),10,(0,0,255),-1)


        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        cv2.putText(frame, "fps: {:.2f}".format(fps), (100,100), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,255,0))

        ## clearing the canny image, but the FPS gets very low
        # imgCannyClear = remove_small_objects(imgCanny)
        # imgHoughLine = getHoughLines(imgCannyClear)
        getContours(imgCanny)

        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        cv2.imshow("Mask", imgResult)
        cv2.imshow("Canny", imgCanny)
        cv2.imshow("Video", frame)
        cv2.imshow("Hough Line",imgHoughLine)

        if cv2.waitKey(1) == ord('q'):
            break