#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import argparse
import time

'''
Mobile object localizer demo running on device on RGB camera.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py

Link to the original model:
https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/default/1

Blob taken from:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/151_object_detection_mobile_object_localizer
'''

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float, help="Coonfidence threshold", default=0.2)

args = parser.parse_args()
THRESHOLD = args.threshold
NN_PATH = "models/generic_object_localizer_192x192.blob"
NN_WIDTH = 192
NN_HEIGHT = 192
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360


# --------------- Methods ---------------

def plot_boxes(frame, boxes, colors, scores):
    color_black = (0, 0, 0)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        y1 = (frame.shape[0] * box[0]).astype(int)
        y2 = (frame.shape[0] * box[2]).astype(int)
        x1 = (frame.shape[1] * box[1]).astype(int)
        x2 = (frame.shape[1] * box[3]).astype(int)
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 15), color, -1)
        cv2.putText(frame, f"{scores[i]:.2f}", (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color_black)


# --------------- Pipeline ---------------
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

##### The RGB, manip, and detection part ###########################################

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(NN_PATH)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Color camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
cam.setInterleaved(False)
cam.setFps(40)

# Create manip
manip = pipeline.createImageManip()
manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
manip.initialConfig.setKeepAspectRatio(False)

# Link preview to manip and manip to nn
cam.preview.link(manip.inputImage)
manip.out.link(detection_nn.input)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("cam")
xout_rgb.input.setBlocking(False)
cam.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)
detection_nn.out.link(xout_nn.input)

xout_manip = pipeline.createXLinkOut()
xout_manip.setStreamName("manip")
xout_manip.input.setBlocking(False)
manip.out.link(xout_manip.input)


######### The Depth and Spatial Location Calculator part ###############################

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

lrcheck = False
subpixel = False

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

# Config
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.setWaitForConfigInput(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    np.random.seed(0)
    colors_full = np.random.randint(255, size=(100, 3), dtype=int)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_manip = device.getOutputQueue(name="manip", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False

    while True:
        in_cam = q_cam.get()
        in_nn = q_nn.get()
        in_manip = q_manip.get()
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

        frame = in_cam.getCvFrame()
        frame_manip = in_manip.getCvFrame()
        depthFrame = inDepth.getFrame()

        # get outputs
        detection_boxes = np.array(in_nn.getLayerFp16("ExpandDims")).reshape((100, 4))
        detection_scores = np.array(in_nn.getLayerFp16("ExpandDims_2")).reshape((100,))

        # keep boxes bigger than threshold
        mask = detection_scores >= THRESHOLD
        boxes = detection_boxes[mask]
        colors = colors_full[mask]
        scores = detection_scores[mask]

        # draw boxes
        #plot_boxes(frame, boxes, colors, scores)
        #plot_boxes(frame_manip, boxes, colors, scores)
        color_black = (0, 0, 0)

        cfg = dai.SpatialLocationCalculatorConfig()
        configs = []
                
        for i in range(boxes.shape[0]):
            box = boxes[i]
            y1 = (frame.shape[0] * box[0]).astype(int)
            y2 = (frame.shape[0] * box[2]).astype(int)
            x1 = (frame.shape[1] * box[1]).astype(int)
            x2 = (frame.shape[1] * box[3]).astype(int)


            d_y1 = (depthFrame.shape[0] * box[0]).astype(int)
            d_y2 = (depthFrame.shape[0] * box[2]).astype(int)
            d_x1 = (depthFrame.shape[1] * box[1]).astype(int)
            d_x2 = (depthFrame.shape[1] * box[3]).astype(int)
            topLeft.x = d_x1
            topLeft.y = d_y1
            bottomRight.x = d_x2
            bottomRight.y = d_y2

            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 100
            config.depthThresholds.upperThreshold = 10000
            config.roi = dai.Rect(topLeft, bottomRight)
            #config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
            configs.append(config)

            color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 15), color, -1)
            cv2.putText(frame, f"{scores[i]:.2f}", (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color_black)

        cfg.setROIs(configs)
        spatialCalcConfigInQueue.send(cfg)

        # show fps
        # show fps and predicted count
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

        # show frame
        cv2.imshow("Localizer", frame)
        cv2.imshow("Manip + NN", frame_manip)


        
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        spatialData = spatialCalcQueue.get().getSpatialLocations()
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            depthMin = depthData.depthMin
            depthMax = depthData.depthMax

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, 255)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, 255)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, 255)
        # Show the frame
        cv2.imshow("depth", depthFrameColor)


        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()


        if cv2.waitKey(1) == ord('q'):
            break