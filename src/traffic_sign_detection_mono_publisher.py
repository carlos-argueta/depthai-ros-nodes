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

    for i, (bbox, position, class_id, score) in enumerate(boxesData):

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
        result.id = class_id
        result.score = score
        detection.results.append(result)

        detection.bbox.center.x = xCenter;
        detection.bbox.center.y = yCenter;
        detection.bbox.size_x = xSize;
        detection.bbox.size_y = ySize;

        detection.position.x = position[0]
        detection.position.y = position[1]
        detection.position.z = position[2]
       
        detection.is_tracking = False;
        detection.tracking_id = "-1"

        opDetectionMsg.detections.append(detection)

    
    sequenceNum += 1

    return opDetectionMsg

def create_pipeline():
    
    nnBlobPath = str((Path(__file__).parent / Path("models/frozen_inference_graph_openvino_2021.4_5shave.blob")).resolve().absolute())
    # MobilenetSSD label texts
    
    syncNN = True

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    imageManip = pipeline.create(dai.node.ImageManip)

    xoutManip = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)
    depthRoiMap = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)

    xoutManip.setStreamName("right")
    nnOut.setStreamName("detections")
    depthRoiMap.setStreamName("boundingBoxDepthMapping")
    xoutDepth.setStreamName("depth")

    # Properties
    imageManip.initialConfig.setResize(300, 300)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # StereoDepth
    stereo.initialConfig.setConfidenceThreshold(255)

    # Define a neural network that will make predictions based on the source frames
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    imageManip.out.link(spatialDetectionNetwork.input)
    if syncNN:
        spatialDetectionNetwork.passthrough.link(xoutManip.input)
    else:
        imageManip.out.link(xoutManip.input)

    spatialDetectionNetwork.out.link(nnOut.input)
    spatialDetectionNetwork.boundingBoxMapping.link(depthRoiMap.input)

    stereo.rectifiedRight.link(imageManip.inputImage)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)


    return pipeline

def detections_publisher(camera_height_from_floor):

    rospy.init_node('TrafficSignDetectionPublisher', anonymous=True)
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
    #depth_image_pub = rospy.Publisher('/'+camera_name+'/depth/image_rect', Image, queue_size=5)

    #depth_camera_pub = rospy.Publisher('/'+camera_name+'/depth/camera_info', CameraInfo,
    #                                            queue_size=5)
    rgb_image_pub = rospy.Publisher('/'+camera_name+'/rgb/image', Image, queue_size=5)
    
    rgb_camera_pub = rospy.Publisher('/'+camera_name+'/rgb/camera_info', CameraInfo,
                                                queue_size=5)

    dets_pub = rospy.Publisher('/'+camera_name+'/detections/traffic_sign_detections', SpatialDetectionArray, queue_size=1)
    
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

    pipeline = create_pipeline()

    # Pipeline defined, now the device is assigned and pipeline is started
    with dai.Device(pipeline) as device:

    
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        #previewQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        #detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        #xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
        #depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        depthRoiMapQueue = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        rectifiedRight = None
        detections = []

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)


        seq = 0

        labelMap = ['nothing','crosswalk ahead','give way','green light','priority road','red light','right turn','stop sign','traffic light','yellow light']
        
        while not rospy.is_shutdown():

            inRectified = previewQueue.get()
            inDet = detectionNNQueue.get()
            inDepth = depthQueue.get()

            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            rectifiedRight = inRectified.getCvFrame()

            depthFrame = inDepth.getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
            

            det_boxes = []  
            source_frame = camera_name+"_right_camera_optical_frame"
            rospy_time_now = rospy.Time.now();

            detections = inDet.detections
            if len(detections) != 0:
                boundingBoxMapping = depthRoiMapQueue.get()
                roiDatas = boundingBoxMapping.getConfigData()

                for roiData in roiDatas:
                    roi = roiData.roi
                    roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                    topLeft = roi.topLeft()
                    bottomRight = roi.bottomRight()
                    xmin = int(topLeft.x)
                    ymin = int(topLeft.y)
                    xmax = int(bottomRight.x)
                    ymax = int(bottomRight.y)
                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            # If the rectifiedRight is available, draw bounding boxes on it and show the rectifiedRight
            height = rectifiedRight.shape[0]
            width = rectifiedRight.shape[1]
            for detection in detections:
                # Denormalize bounding box
                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)

                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label

                cv2.putText(rectifiedRight, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(rectifiedRight, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(rectifiedRight, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(rectifiedRight, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(rectifiedRight, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                cv2.rectangle(rectifiedRight, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)


                camera_point = PointStamped()
                camera_point.header.frame_id = source_frame;
                camera_point.header.stamp = rospy_time_now;
                camera_point.point.x = detection.spatialCoordinates.x / 1000.0;
                camera_point.point.y = (detection.spatialCoordinates.y + camera_height_from_floor) / 1000.0;
                camera_point.point.z = detection.spatialCoordinates.z / 1000.0;


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

               
                det_box = [x1, y1, x2, y2]
                pos_box = [base_point.point.x,base_point.point.y,base_point.point.z ]
                det_boxes.append((det_box,pos_box,detection.label,int(detection.confidence * 100)))

                
            # Create and publish ROS messages
            #depth_frame = (depthFrameColor * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

            # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
            #cv_frame = cv2.applyColorMap(cv_frame, cv2.COLORMAP_JET)

            #depth_image_msg = bridge.cv2_to_imgmsg(depthFrameColor, encoding="passthrough")
            #depth_image_msg.header.stamp = rospy.Time.now()
            #depth_image_msg.header.seq = seq
            #depth_image_msg.header.frame_id = camera_name+"_rgb_camera_optical_frame"
            
            #depth_image_pub.publish(depth_image_msg)

            rgb_image_msg = bridge.cv2_to_imgmsg(rectifiedRight, encoding="passthrough")
            rgb_image_msg.header.stamp = rospy.Time.now()
            rgb_image_msg.header.seq = seq
            rgb_image_msg.header.frame_id = camera_name+"_rgb_camera_optical_frame"

            rgb_image_pub.publish(rgb_image_msg)
                
            if det_boxes:
                

                detections_msg = bboxToRosMsg(det_boxes)
                detections_msg.header = rgb_image_msg.header
                
                dets_pub.publish(detections_msg)

                seq = seq + 1


            if debug:
                cv2.putText(rectifiedRight, "NN fps: {:.2f}".format(fps), (2, rectifiedRight.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                cv2.imshow("depth", depthFrameColor)
                cv2.imshow("rectified right", rectifiedRight)
                        
            
            
 
            if cv2.waitKey(1) == ord('q'):
                break


            
            rate.sleep()


    
   
if __name__ == '__main__':
    try:
        detections_publisher(390)
    except rospy.ROSInterruptException:
        pass