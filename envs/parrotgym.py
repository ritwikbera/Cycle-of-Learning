#!/usr/bin/env python
""" parrotgym.py:
Defines a gym wrapper environment following OpenAI template for communicating and controlling the parrot drone.

__author__ = "Nicholas Waytowich"
__version__ = "0.0.1"
__status__ = "Prototype"
__date__ = "July 11, 2019"

"""

# import libraries
import sys
import os
import gym
from gym import spaces
import cv2
from PIL import Image
import threading
import apriltag
import numpy as np
from time import gmtime, strftime, localtime
import csv
import pandas

import olympe  # parrot api
from olympe.messages.ardrone3.Piloting import TakeOff, moveBy, Landing, PCMD
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.tools.logger import TraceLogger
from olympe.messages import gimbal
from olympe.messages.ardrone3.PilotingState import SpeedChanged
from olympe.messages.ardrone3.PilotingState import AttitudeChanged
from olympe.messages.ardrone3.PilotingState import AltitudeChanged
import keyboard

sys.path.insert(0, '../')
import SimpleJoystick


# Main class for interfacing with the parrot drone
class parrotgym(gym.Env):

    # parrotgym constructor
    def __init__(self, render=False, debug=False, num_tags=6, simulation=True):
        # debug mode
        self.debug = debug

        # drone speed
        self.speed = 50

        # render
        self.render = render

        # number of april tags
        self.num_tags = num_tags

        # initialize ip address
        self.simulation = simulation
        if self.simulation:
            ip = "10.202.0.1"    # ip address for virtual drone
        else:
            ip = "192.168.42.1"  # ip address for real drone

        self.drone = olympe.Drone(ip, loglevel=TraceLogger.level.error)

        # Initialize the apriltag detector
        self.detector = apriltag.Detector()

        # Initialize cv2frame
        self.cv2frame = None

        # connect to the xbox controller
        self.connect_controller()

    # main reset function
    def reset(self):
        # stop drone
        self.stop()
        # start drone
        self.start()

    # connect to the xbox controller
    def connect_controller(self):
        # start polling thread for xbox controller
        self.js = SimpleJoystick.SimpleJoystick()
        self.js_thread = threading.Thread(target=self.js.poll)
        self.js_thread.start()
        print('controller connected')

    # connect to drone
    def connect(self):
        # connect to drone
        self.drone.connection()
        print('drone connected')

        # set callbacks for capturing image data
        self.drone.set_streaming_callbacks(
            raw_cb=self.yuv_frame_cb,
            h264_cb=self.h264_frame_cb
            )

        # start video streaming
        self.drone.start_video_streaming()

    # main starting procedure for drone
    def start(self):       
        
        # connect to drone and initialize camera feed
        self.connect()
        # launch drone and go into piloting mode
        self.takeoff()

    # launch drone and go into piloting mode
    def takeoff(self):
        # Take off and then wait for a maximum of 5 seconds for the drone to start hovering
        print('taking off')
        self.drone(
            TakeOff() >> FlyingStateChanged(state="hovering", _timeout=5)
        ).wait()

        # start piloting
        print('starting piloting')
        olympe.Drone.start_piloting(self.drone)

    def camera_down(self):
        # move camera downward
        print('moving camera')
        cameraAction = self.drone(gimbal.set_target(
                        gimbal_id=0,
                        control_mode="position",
                        yaw_frame_of_reference="none",
                        yaw=0.0,
                        pitch_frame_of_reference="relative",
                        pitch=-90.0,
                        roll_frame_of_reference="none",
                        roll=0.0,
                )).wait()

        if not cameraAction.success():
            raise RuntimeError("Cannot set gimbal velocity target")

    # main stopping procedure for drone
    def stop(self):
        # land drone
        self.land()
        # disconnect drone
        self.disconnect()

    # land drone 
    def land(self):
        # stop piloting interface
        print('stop piloting')
        olympe.Drone.stop_piloting(self.drone)

        # land drone and disconnect
        print('drone landing')
        self.drone(
            Landing() >> FlyingStateChanged(state="landed", _timeout=5)
        ).wait()

    # stop stream and disconnect
    def disconnect(self):  
        # stop streaming
        self.drone.stop_video_streaming()

        # disconnect
        self.drone.disconnection()
        self.js.polling = False

    # main step function
    def step(self, action=None, render=False):     

        # unpack action
        roll, pitch, yaw, gaz = action

        # send action to drone
        olympe.Drone.piloting_pcmd(self.drone, roll, pitch, yaw, gaz, 1.0)

        # build state/observation (currently it consists of just the x,y position of landing pad)

        # get landing pad location
        april_tags = self.find_landingpad()

        # get drone imu data (pos,vel,acc)
        imu_state = self.get_imu()
        
        # build obs
        obs = np.concatenate((april_tags, imu_state))

        return obs

    # get internal state information from drone (IMU)
    def get_imu(self):

        # drone velocity
        vel = self.drone.get_state(SpeedChanged)

        # drone attitude
        att = self.drone.get_state(AttitudeChanged)

        # drone altitude
        alt = self.drone.get_state(AltitudeChanged)

        # build imu vector
        imu_state = np.array([vel["speedX"], vel["speedY"], vel["speedZ"],
                att["roll"], att["pitch"], att["yaw"], alt["altitude"]])

        # for debugging
        #if self.debug:
            #print(imu_state)

        return imu_state

    # get xy locations of the april-tags on the landing pad
    def find_landingpad(self):

        # we are currently working with a landing pad with 6 april tags
        tag_centers = np.zeros((2*self.num_tags))
        
        # grab april-tag location from current image
        if self.cv2frame is not None:
            # convert to grayscale
            gray = cv2.cvtColor(self.cv2frame, cv2.COLOR_BGR2GRAY)

            # extract april-tag info
            result = self.detector.detect(gray)

            # process april-tag info
            if len(result) > 0:
                for tag in range(len(result)):
                    # sort april tag centers by tag id:
                    tag_id = int(result[tag][1])
                    tag_cxy = result[tag][6]
                    tag_centers[(2*tag_id)] = tag_cxy[0]
                    tag_centers[(2*tag_id+1)] = tag_cxy[1]

                    # for debugging
                    if self.debug:
                        print("tag_center: ", result[tag][6],
                            "tag_id: ", result[tag][1])

        return tag_centers

    # Callback for the recieving images from drone camera
    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.
            :type yuv_frame: olympe.VideoFrame
        """
        # the VideoFrame.info() dictionary contains some useful informations
        # such as the video resolution
        info = yuv_frame.info()
        height, width = info["yuv"]["height"], info["yuv"]["width"]

        # convert pdraw YUV flag to OpenCV YUV flag
        cv2_cvt_color_flag = {
            olympe.PDRAW_YUV_FORMAT_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.PDRAW_YUV_FORMAT_NV12: cv2.COLOR_YUV2BGR_NV12,
        }[info["yuv"]["format"]]

        # yuv_frame.as_ndarray() is a 2D numpy array with the proper "shape"
        # i.e (3 * height / 2, width) because it's a YUV I420 or NV12 frame

        # Use OpenCV to convert the yuv frame to RGB
        self.cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)

        # Use OpenCV to show this frame
        if self.render:
            cv2.imshow("Olympe Streaming Example", self.cv2frame)
            cv2.waitKey(1)  # please OpenCV for 1 ms...

    # Callback for the recieving video from drone camera
    def h264_frame_cb(self, h264_frame):
        a = []

    # read from xbox controller
    def get_human_action(self):
        # get roll, pitch, yaw, gaz values from xbox controller
        roll = int(self.js.axis_states['rx']*self.speed) 
        pitch = int(self.js.axis_states['ry']*-self.speed)
        yaw = int(self.js.axis_states['x']*self.speed)
        gaz = int(self.js.axis_states['y']*-self.speed)
        
        # compensate for deadzone
        if abs(roll) <= self.speed*0.10:
            roll = 0
        if abs(pitch) <= self.speed*0.10:
            pitch = 0
        if abs(yaw) <= self.speed*0.10:
            yaw = 0
        if abs(gaz) <= self.speed*0.10:
            gaz = 0

        return [roll, pitch, yaw, gaz]

    # self contained function for collecting demonstrations
    # TODO: complete demonstration collection
    def run_demo(self):

        # Initialize demonstration file
        date_time = strftime("%H_%M_%m_%d_%Y", localtime())
        file = open(date_time + ".csv", 'w')
        columns = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    "tag_0_X", "tag_0_Y", "tag_1_X", "tag_1_Y", "tag_2_X", "tag_2_Y", "tag_3_X", "tag_3_Y",
                     "tag_4_X", "tag_4_Y", "tag_5_X", "tag_6_Y", "speedX", "speedY", "speedZ", "roll", "pitch",
                     "yaw", "altitude", "human_roll", "human_pitch", "human_yaw", "human_gaz")
        # write header
        file.write("{}".format(columns))

        # start drone
        self.connect()
        while not self.js.button_states['a']:
            pass
        self.takeoff()

        # main control loop
        self.continue_flying = True
        while self.continue_flying:
            # poll from the xbox controller
            if self.js.button_states['b']:
                self.continue_flying = False

            # playground area for messing with certain drone stuff for now
            # change drone camera angle
            if self.js.axis_states['hat0y']:
                cameraAction = self.drone(gimbal.set_target(
                        gimbal_id=0,
                        control_mode="velocity",
                        yaw_frame_of_reference="none",
                        yaw=0.0,
                        pitch_frame_of_reference="relative",
                        pitch=-1*self.js.axis_states['hat0y'],
                        roll_frame_of_reference="none",
                        roll=0.0,
                )).wait()

            # take as step and receive obs
            action = self.get_human_action()
            obs = self.step(action, render=True)

            # log file
            obs_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], obs[9],
                obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], obs[16], obs[17], obs[18], action[0], action[1],
                action[2], action[3])
            file.write(obs_str)


        # close out
        self.land()
        while not self.js.button_states['b']:
            pass
        file.close()
        self.disconnect()

    # test connecting>>disconnecting from drone
    def FlightTest1(self):
        drone.connect()
        while not self.js.button_states['b']:
            pass
        drone.disconnect()

    # test takeoff>>land
    def FlightTest2(self):
        drone.connect()
        while not self.js.button_states['a']:
            pass
        drone.takeoff()
        while not self.js.button_states['b']:
            pass
        drone.land()
        while not self.js.button_states['b']:
            pass
        drone.disconnect()


# main entry point
if __name__ == '__main__':

    # initialize parrotgym
    drone = parrotgym(render=True, debug=True)

    # run flight test 1
    #drone.FlightTest1()

    # run flight test 2
    #drone.FlightTest2()

    # run demo
    drone.run_demo()

    