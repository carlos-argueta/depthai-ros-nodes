#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

import rospy

import tf2_ros
import tf2_geometry_msgs

from depthai_ros_msgs.msg import SpatialDetection, SpatialDetectionArray
from vision_msgs.msg import ObjectHypothesis, BoundingBox2D
from geometry_msgs.msg import Pose, PoseArray, PointStamped
from sensor_msgs.msg import CameraInfo, Image

from pathlib import Path

import time

from cv_bridge import CvBridge, CvBridgeError

from camera_info_manager import CameraInfoManager

sequenceNum = 0
frameName = ""

def bboxToRosMsg(boxesData):
    global sequenceNum

    opDetectionMsg = SpatialDetectionArray ()

        
    # setting the header
    opDetectionMsg.header.seq = sequenceNum;
    opDetectionMsg.header.stamp = rospy.Time.now()
    opDetectionMsg.header.frame_id = frameName;

    for i, (bbox, score) in enumerate(boxesData):

        xMin = int(bbox[0])
        yMin = int(bbox[1])
        xMax = int(bbox[2])
        yMax = int(bbox[3])

        xSize = xMax - xMin;
        ySize = yMax - yMin;
        xCenter = xMin + xSize / 2.0;
        yCenter = yMin + ySize / 2.0;

        detection = SpatialDetection()
            
        result = ObjectHypothesis()
        #print(t.label)
        result.id = i
        result.score = score
        detection.results.append(result)

        detection.bbox.center.x = xCenter;
        detection.bbox.center.y = yCenter;
        detection.bbox.size_x = xSize;
        detection.bbox.size_y = ySize;
       

        detection.is_tracking = False;
        detection.tracking_id = "-1"

        opDetectionMsg.detections.append(detection)

    
    sequenceNum += 1

    return opDetectionMsg

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def create_pipeline():
    
    NN_PATH = str((Path(__file__).parent / Path("models/frozen_inference_graph_openvino_2021.4_5shave.blob")).resolve().absolute())
    NN_WIDTH = 300
    NN_HEIGHT = 300
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 360

    # --------------- Pipeline ---------------
    pipeline = dai.Pipeline()

    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    ##### The RGB, manip, and detection part ###########################################

    # Define a neural network that will make predictions based on the source frames
    #detection_nn = pipeline.createNeuralNetwork()
    #detection_nn.setBlobPath(NN_PATH)
    #detection_nn.setNumPoolFrames(2)
    #detection_nn.input.setBlocking(False)
    #detection_nn.setNumInferenceThreads(1)

    detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    detection_nn.setConfidenceThreshold(0.5)
    detection_nn.setBlobPath(NN_PATH)
    detection_nn.setNumInferenceThreads(1)
    detection_nn.setNumPoolFrames(2)
    detection_nn.input.setBlocking(False)

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


    return pipeline, stereo

def detections_publisher(camera_height_from_floor):

    rospy.init_node('DepthPublisher', anonymous=True)
    rate = rospy.Rate(5) # ROS Rate at 5Hz

    # Get the parameters:
    cam_id = ''
    if rospy.has_param('~cam_id'):
        cam_id = rospy.get_param('~cam_id')

    camera_name = rospy.get_param("~camera_name")

    debug = rospy.get_param("~debug")

    camera_param_uri = rospy.get_param("~camera_param_uri")

    #image_pub = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=100)
    #camera_pub = rospy.Publisher("/camera/depth/camera_info", CameraInfo,
    #                                            queue_size=100)

    #image_pub = rospy.Publisher('/image_raw', Image, queue_size=100)
    depth_image_pub = rospy.Publisher('/'+camera_name+'/depth/image_rect', Image, queue_size=5)

    depth_camera_pub = rospy.Publisher('/'+camera_name+'/depth/camera_info', CameraInfo,
                                                queue_size=5)
    rgb_image_pub = rospy.Publisher('/'+camera_name+'/rgb/image', Image, queue_size=5)
    
    rgb_camera_pub = rospy.Publisher('/'+camera_name+'/rgb/camera_info', CameraInfo,
                                                queue_size=5)

    dets_pub = rospy.Publisher('/'+camera_name+'/detections/object_detections', SpatialDetectionArray, queue_size=1)
    
    poses_pub = rospy.Publisher('/'+camera_name+'/detections/object_poses', PoseArray, queue_size=10)

    tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
    tf_listener = tf2_ros.TransformListener(tf_buffer)
        

    bridge = CvBridge()

    left_uri = camera_param_uri +"/" + "left.yaml";
  
    right_uri = camera_param_uri + "/" + "right.yaml";
    
    stereo_uri = camera_param_uri + "/" + "right.yaml";

    manager = CameraInfoManager(camera_name , stereo_uri)
    manager.loadCameraInfo()

    camera_info = None
    if manager.isCalibrated():
        camera_info = manager.getCameraInfo()
        
    else:
        raise ROSException("No CameraInfo")

    #print(camera_info)
        
    #print(left_uri, right_uri, stereo_uri)

    found, device_info = None, None
    if cam_id is not None and cam_id != '':
        found, device_info = dai.Device.getDeviceByMxId(cam_id)

        if not found:
            raise RuntimeError("Device not found!")
    else:
        print("No camera ID specified, finding one")
        for device in dai.Device.getAllAvailableDevices():
            print(f"{device.getMxId()} {device.state}")
            cam_id = device.getMxId()
        if cam_id != '':
            print("Using camera ",cam_id)
            found, device_info = dai.Device.getDeviceByMxId(cam_id)
        else:
            raise RuntimeError("No device found!")

    pipeline, depth = create_pipeline()

    # Pipeline defined, now the device is assigned and pipeline is started
    with dai.Device(pipeline) as device:

        THRESHOLD = 0.2

        np.random.seed(0)
        colors_full = np.random.randint(255, size=(100, 3), dtype=int)

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
        q_manip = device.getOutputQueue(name="manip", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

        start_time = time.time()
        counter = 0
        fps = 0
        layer_info_printed = False

        seq = 0

        labelMap = ['nothing','crosswalk ahead','give way','green light','priority road','red light','right turn','stop sign','traffic light','yellow light']
        
        while not rospy.is_shutdown():
            in_cam = q_cam.get()
            in_nn = q_nn.get()
            in_manip = q_manip.get()
            inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

            frame = in_cam.getCvFrame()
            frame_manip = in_manip.getCvFrame()
            depthFrame = inDepth.getFrame()

            detections = in_nn.detections
            # get outputs
            #detection_boxes = np.array(in_nn.getLayerFp16("ExpandDims")).reshape((100, 4))
            #detection_scores = np.array(in_nn.getLayerFp16("ExpandDims_2")).reshape((100,))

            # keep boxes bigger than threshold
            #mask = detection_scores >= THRESHOLD
            #boxes = detection_boxes[mask]
            #colors = colors_full[mask]
            #scores = detection_scores[mask]

            # draw boxes
            #plot_boxes(frame, boxes, colors, scores)
            #plot_boxes(frame_manip, boxes, colors, scores)
            color_black = (0, 0, 0)

            cfg = dai.SpatialLocationCalculatorConfig()
            configs = []
            
            det_boxes = []  
            color = (0, 0, 0)

            for detection in detections:
            
                print(labelMap[detection.label])
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                bbox_d = frameNorm(depthFrame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                
                y1 = bbox[1]
                y2 = bbox[3]
                x1 = bbox[0]
                x2 = bbox[2]

                d_y1 = bbox_d[1]
                d_y2 = bbox_d[3]
                d_x1 = bbox_d[0]
                d_x2 = bbox_d[2]
                
                det_box = [d_x1, d_y1, d_x2, d_y2]
                det_boxes.append((det_box,int(detection.confidence * 100)))

                # Config
                topLeft = dai.Point2f(d_x1, d_y1)
                bottomRight = dai.Point2f(d_x2, d_y2)
                #topLeft.x = 
                #topLeft.y = 
                #bottomRight.x = 
                #bottomRight.y = 

                config = dai.SpatialLocationCalculatorConfigData()
                config.depthThresholds.lowerThreshold = 100
                config.depthThresholds.upperThreshold = 10000
                config.roi = dai.Rect(topLeft, bottomRight)
                #config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
                configs.append(config)

                #color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))

                cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                #cv2.putText(frame, str(detection.label), (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
                
            if len(configs) > 0:
                cfg.setROIs(configs)
                spatialCalcConfigInQueue.send(cfg)

            # show fps
            # show fps and predicted count
            color_black, color_white = (0, 0, 0), (255, 255, 255)
            label_fps = "Fps: {:.2f}".format(fps)
            (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
            
            if debug:
                cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
                cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                            0.4, color_black)

                # show frame
                cv2.imshow("Localizer", frame)
                cv2.imshow("Manip + NN", frame_manip)


            
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            source_frame = camera_name+"_right_camera_optical_frame"
            base_poses = PoseArray()
            base_poses.header.frame_id = source_frame;
            base_poses.header.stamp = rospy.Time.now();

            spatialData = spatialCalcQueue.get().getSpatialLocations()
            for depthData in spatialData:
                roi = depthData.config.roi
                roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)

                camera_point = PointStamped()
                camera_point.header = base_poses.header
                camera_point.point.x = depthData.spatialCoordinates.x / 1000.0;
                camera_point.point.y = (depthData.spatialCoordinates.y + camera_height_from_floor) / 1000.0;
                camera_point.point.z = depthData.spatialCoordinates.z / 1000.0;


                # Convert point from camera optical frame to camera frame
                target_frame = camera_name+"_right_camera_frame"
                source_frame = camera_name+"_right_camera_optical_frame"

                transform1 = tf_buffer.lookup_transform(target_frame,
                    source_frame, #source frame
                    rospy.Time(0), #get the tf at first available time
                    rospy.Duration(1.0)) #wait for 1 second
                frame_point = tf2_geometry_msgs.do_transform_point(camera_point, transform1)

                # Convert the point from camera frame to target frame
                target_frame = "base_link"
                source_frame = camera_name+"_right_camera_frame"
                transform2 = tf_buffer.lookup_transform(target_frame,
                    source_frame, #source frame
                    rospy.Time(0), #get the tf at first available time
                    rospy.Duration(1.0)) #wait for 1 second
                base_point = tf2_geometry_msgs.do_transform_point(frame_point, transform2)

                base_pose =  Pose()
                base_pose.position.x = base_point.point.x;
                base_pose.position.y = base_point.point.y;
                base_pose.position.z = base_point.point.z;

                base_poses.poses.append(base_pose)

                depthMin = depthData.depthMin
                depthMax = depthData.depthMax

                fontType = cv2.FONT_HERSHEY_TRIPLEX

                #if debug:
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, 255)
                cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, 255)
                cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, 255)
            
            # Show the frame
            if debug:
                cv2.imshow("depth", depthFrameColor)


            counter += 1
            if (time.time() - start_time) > 1:
                fps = counter / (time.time() - start_time)

                counter = 0
                start_time = time.time()


            if cv2.waitKey(1) == ord('q'):
                break


            # Create and publish ROS messages
            cv_frame = (depthFrameColor * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

            # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
            cv_frame = cv2.applyColorMap(cv_frame, cv2.COLORMAP_JET)

            depth_image_msg = bridge.cv2_to_imgmsg(cv_frame, encoding="passthrough")
            depth_image_msg.header.stamp = rospy.Time.now()
            depth_image_msg.header.seq = seq
            depth_image_msg.header.frame_id = camera_name+"_rgb_camera_optical_frame"
            
            depth_image_pub.publish(depth_image_msg)
                
            if det_boxes:
                

                detections_msg = bboxToRosMsg(det_boxes)
                detections_msg.header = depth_image_msg.header
                
                dets_pub.publish(detections_msg)

                poses_pub.publish(base_poses)

                seq = seq + 1

            rate.sleep()


    
   
if __name__ == '__main__':
    try:
        detections_publisher(390)
    except rospy.ROSInterruptException:
        pass