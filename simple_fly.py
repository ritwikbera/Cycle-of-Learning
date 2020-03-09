
# Description: A simple flight controller to fly the parrot drone using and xbox remote
#
# Author: Nick Waytowich

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, moveBy, Landing, PCMD
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

import SimpleJoystick, threading

def main():

    # start polling thread for xbox controller
    js = SimpleJoystick.SimpleJoystick()
    js_thread = threading.Thread(target = js.poll)
    js_thread.start()
    print('controller connected')

    # connect to drone
    drone = olympe.Drone("10.202.0.1")
    drone.connection()
    print('drone connected')
    
    # Take off and then wait for a maximum of 5 seconds for the drone to start hovering
    print('taking off')
    drone(
        TakeOff()
        >> FlyingStateChanged(state="hovering", _timeout=5)
    ).wait()


    # start piloting interface
    print('starting piloting')
    olympe.Drone.start_piloting(drone)


    #olympe.Drone.piloting_pcmd(drone,50,0,0,0,5.0)

    continue_flying = True
    while continue_flying:

        # poll from the xbox controller
        if js.button_states['b']:
            continue_flying = False

        # get roll,pitch,yaw,gaz values from controller
        roll = int(js.axis_states['rx']*100.0) 
        pitch = int(js.axis_states['ry']*-100.0)
        yaw = int(js.axis_states['x']*100.0)
        gaz = int(js.axis_states['y']*-100.0)

        if abs(roll) <=5:
            roll = 0
        if abs(pitch) <=5:
            pitch = 0
        if abs(yaw) <=5:
            yaw = 0
        if abs(gaz) <=5:
            gaz = 0

        # send commands to drone
        olympe.Drone.piloting_pcmd(drone,roll,pitch,yaw,gaz,1.0)
        

    # stop piloting interface
    print('stop piloting')
    #drone(stop_piloting()).wait()
    olympe.Drone.stop_piloting(drone)

    # land drone and disconnect
    print('drone landing')
    drone(Landing()).wait()
    drone.disconnection()


# starting point
if __name__ == '__main__':
    main()