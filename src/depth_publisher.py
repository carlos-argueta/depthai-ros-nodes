#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

import rospy
from sensor_msgs.msg import CameraInfo, Image

from cv_bridge import CvBridge, CvBridgeError

from camera_info_manager import CameraInfoManager


def create_pipeline():
    '''
    If one or more of the additional depth modes (lrcheck, extended, subpixel)
    are enabled, then:
     - depth output is FP16. TODO enable U16.
     - median filtering is disabled on device. TODO enable.
     - with subpixel, either depth or disparity has valid data.
    Otherwise, depth output is U16 (mm) and median is functional.
    But like on Gen1, either depth or disparity has valid data. TODO enable both.
    '''
    # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
    extended_disparity = False
    # Better accuracy for longer distance, fractional disparity 32-levels:
    subpixel = False
    # Better handling for occlusions:
    lr_check = False

    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(1280 , 720)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)


    # Define a source - two mono (grayscale) cameras
    left = pipeline.createMonoCamera()
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    #left.setImageOrientation(dai.CameraImageOrientation.HORIZONTAL_MIRROR)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)

    right = pipeline.createMonoCamera()
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    #right.setImageOrientation(dai.CameraImageOrientation.HORIZONTAL_MIRROR)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
    depth = pipeline.createStereoDepth()
    depth.setConfidenceThreshold(255)
    depth.setOutputDepth(True)
    depth.setOutputRectified(True)
    #depth.setRectifyMirrorFrame(False)
    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
    median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7 # For depth filtering
    depth.setMedianFilter(median)

    depth.setLeftRightCheck(lr_check)

    # Normal disparity values range from 0..95, will be used for normalization
    max_disparity = 95

    if extended_disparity: max_disparity *= 2 # Double the range
    depth.setExtendedDisparity(extended_disparity)

    if subpixel: max_disparity *= 32 # 5 fractional bits, x32
    depth.setSubpixel(subpixel)

    # When we get disparity to the host, we will multiply all values with the multiplier
    # for better visualization
    multiplier = 255 / max_disparity

    left.out.link(depth.left)
    right.out.link(depth.right)

    # Create output
    #xout = pipeline.createXLinkOut()
    #xout.setStreamName("disparity")
    #depth.disparity.link(xout.input)

    # Create output
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
    depth.depth.link(xoutDepth.input)

    # Create output
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)


    return pipeline, multiplier

def image_publisher():

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

    pipeline, multiplier = create_pipeline()

    # Pipeline defined, now the device is connected to
    with dai.Device(pipeline, device_info) as device:
        # Start pipeline
        device.startPipeline()

        # Output queue will be used to get the disparity frames from the outputs defined above
        q = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        # Output queue will be used to get the rgb frames from the output defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)


        seq = 0
        #while True:
        while not rospy.is_shutdown():
            inDepth = q.get()  # blocking call, will wait until a new data has arrived
            frame = inDepth.getCvFrame()

            inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

            # Retrieve 'bgr' (opencv format) frame
            frame_rgb = inRgb.getCvFrame()


            #print(frame.shape)
            
            cv_frame = (frame.copy()).astype(np.uint8)

            #print(np.min(frame), np.max(frame), i)
            #i = i + 1

            # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
            cv_frame = cv2.applyColorMap(cv_frame, cv2.COLORMAP_JET)

            depth_image_msg = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
            depth_image_msg.header.stamp = rospy.Time.now()
            depth_image_msg.header.seq = seq
            depth_image_msg.header.frame_id = camera_name+"_rgb_camera_optical_frame"
            #if image_message != None:
                #print "There's something here"
            #rospy.loginfo(image_message)
            #print(image_msg.header.seq, image_msg.header.stamp)

            depth_image_pub.publish(depth_image_msg)

            rgb_image_msg = bridge.cv2_to_imgmsg(frame_rgb, encoding="passthrough")
            rgb_image_msg.header.stamp = rospy.Time.now()
            rgb_image_msg.header.seq = seq
            rgb_image_msg.header.frame_id = camera_name+"_rgb_camera_optical_frame"
            #if image_message != None:
                #print "There's something here"
            #rospy.loginfo(image_message)
            #print(image_msg.header.seq, image_msg.header.stamp)

            rgb_image_pub.publish(rgb_image_msg)

            depth_camera_msg = camera_info
            depth_camera_msg.header = depth_image_msg.header  # Copy header from image message
            depth_camera_pub.publish(depth_camera_msg)

            rgb_camera_msg = camera_info
            rgb_camera_msg.header = rgb_image_msg.header  # Copy header from image message
            rgb_camera_pub.publish(rgb_camera_msg)


            # frame is ready to be shown
            if debug:
                cv2.imshow("Depth", cv_frame)
                #print("Show me")

                if cv2.waitKey(1) == ord('q'):
                    break

            rate.sleep()

            seq = seq + 1

        
if __name__ == '__main__':
    try:
        image_publisher()
    except rospy.ROSInterruptException:
        pass