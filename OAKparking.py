from curses.panel import bottom_panel
from traceback import FrameSummary
import cv2
import depthai as dai
import numpy as np
import time
import blobconverter
import math

# this function is to clear out the canny image(delete noise from background and small object)
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

# this function is to get the type of object(triangle or rectangle or etc) and the area(in pixel) of the box
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
                return [(approx[0][0][0],approx[0][0][1]),(approx[1][0][0],approx[1][0][1]), (approx[2][0][0],approx[2][0][1]),(approx[3][0][0],approx[3][0][1])]
    return None

# this function is needed to create the trackbar for the parameter of the masked HSV
def empty(a): #takes 1 argument
   pass

# although we didn't use this at the end, hough line is useful to predict line
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

# Define source
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

# Define output
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutVideo.setStreamName("video")
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setPreviewSize(640, 400)
camRgb.setFps(60)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(4)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

lrcheck = False
subpixel = True

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

# Spatial Calculator Config
topLeft = dai.Point2f(0.45, 0.45)
bottomRight = dai.Point2f(0.55, 0.55)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
camRgb.preview.link(xoutVideo.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    startTime = time.monotonic()
    counter = 0
    fps = 0

    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    color = (255, 255, 255)


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
        
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

        depthFrame = inDepth.getFrame() # depthFrame values are in millimeters

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        
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
        # yellow_lower = np.array([20, 100, 100])
        # yellow_upper = np.array([30, 255, 255])
        mask = cv2.inRange(frame, lower, upper)
        #mask = cv2.inRange(frame, lower, upper)
        imgResult = cv2.bitwise_and(frame, frame, mask = mask) #to merge img to the img with mask named masked
        imgCanny = cv2.Canny(imgResult, t_max, t_min, apertureSize = 5)

        #imgHoughLine = getHoughLines(imgCanny)

        ## getting the corner
        # corners = cv2.goodFeaturesToTrack(imgGray, max_corner, (q_level/1000), min_dis, mask= mask )
        
        # if corners is not None:       
        #     corners = np.int0(corners)
        #     for i in corners:
        #         x,y = i.ravel()
        #         #print(x, " ", y)
        #         #print(imgResult[y,x])
        #         cv2.circle(imgResult,(x,y),10,(0,0,255),-1)
        
        ## clearing the canny image, but the FPS gets very low
        # imgCannyClear = remove_small_objects(imgCanny)
        # imgHoughLine = getHoughLines(imgCannyClear)
        det = getContours(imgCanny)
        
        ##### get only 1 3d coordination which is the avg of the bbox
        if det is not None:
            tmp = sum(det[0]) 
            maxval = tmp
            maxidx = 0
            minval = tmp
            minidx = 0
            for i in range(0,len(det)):
                if sum(det[i]) > maxval:
                    maxval =  sum(det[i])
                    maxidx = i
                if sum(det[i]) < minval:
                    minval =  sum(det[i])
                    minidx = i 

            print(det[minidx], det[maxidx])
            topLeft = dai.Point2f(det[minidx][0]/frame.shape[1] , det[minidx][1]/frame.shape[0])
            bottomRight = dai.Point2f(det[maxidx][0]/frame.shape[1] , det[maxidx][1]/frame.shape[0])
            #print(topLeft, bottomRight)
        

        # Get the Spatial Coordination
        config.roi = dai.Rect(topLeft,bottomRight)
        config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.addROI(config)
        spatialCalcConfigInQueue.send(cfg)
        #spatialLocationCalculator.initialConfig.addROI(config)
        spatialData = spatialCalcQueue.get().getSpatialLocations()

        #print("go")
        for depthData in spatialData:
            #print("test")
            roi = depthData.config.roi
            roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)
            print(f"({xmin} {ymin}), ({xmax} {ymax})")

            depthMin = depthData.depthMin
            depthMax = depthData.depthMax

            selectedFrame = frame
            fontType = cv2.FONT_HERSHEY_TRIPLEX
            # cv2.rectangle(selectedFrame, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(selectedFrame, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, 255)
            cv2.putText(selectedFrame, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, 255)
            cv2.putText(selectedFrame, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, 255)

        # Output the fps
        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        cv2.putText(frame, "fps: {:.2f}".format(fps), (100,100), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,255,0))

        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        cv2.imshow("Mask", imgResult)
        cv2.imshow("Canny", imgCanny)
        cv2.imshow("Video", frame)
        cv2.imshow("depth", depthFrameColor)
        #cv2.imshow("Hough Line",imgHoughLine)

        if cv2.waitKey(1) == ord('q'):
            break