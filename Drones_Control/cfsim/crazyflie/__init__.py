import datetime
import logging
import time
import threading
from collections import namedtuple
from threading import Lock
from threading import Thread
from threading import Timer

from .callbacks import Caller
from .log import Log

class State:
    """Stat of the connection procedure"""
    DISCONNECTED = 0
    INITIALIZED = 1
    CONNECTED = 2
    SETUP_FINISHED = 3


class Param():
    def __init__(self, crazyflie):
        self.cf = crazyflie
        self.is_updated = True

    def set_value(self, complete_name, value):

        if complete_name == 'kalman.initialX':
            self.cf.x = value
        elif complete_name == 'kalman.initialY':
            self.cf.y = value
        elif complete_name == 'kalman.initialZ':
            self.cf.z = value

    def get_value(self, complete_name):

        if complete_name == 'kalman.initialX':
            return self.x
        elif complete_name == 'kalman.initialY':
            return self.y
        elif complete_name == 'kalman.initialZ':
            return self.z
        else:
            print("Warning: {} not implemented in sim as Param.".format(complete_name))
            return 0.0

class Commander():
    def __init__(self, crazyflie):
        self.cf = crazyflie

    def send_velocity_world_setpoint(self, vx, vy, vz, yawrate):
        """
        Send Velocity in the world frame of reference setpoint.
        vx, vy, vz are in m/s
        yawrate is in degrees/s
        """
        self.cf.vx = vx
        self.cf.vy = vy
        self.cf.vz = vz
        self.cf.yawrate = yawrate

    def send_stop_setpoint(self):
        self.cf.vx = 0
        self.cf.vy = 0
        self.cf.vz = 0
        self.cf.yawrate = 0
        self.cf.z = 0


class Crazyflie():
    """The Crazyflie class"""

    def __init__(self, link=None, ro_cache=None, rw_cache=None):
        # Called on disconnect, no matter the reason
        self.disconnected = Caller()
        # Called on unintentional disconnect only
        self.connection_lost = Caller()
        # Called when the first packet in a new link is received
        self.link_established = Caller()
        # Called when the user requests a connection
        self.connection_requested = Caller()
        # Called when the link is established and the TOCs (that are not
        # cached) have been downloaded
        self.connected = Caller()
        # Called if establishing of the link fails (i.e times out)
        self.connection_failed = Caller()
        # Called for every packet received
        self.packet_received = Caller()
        # Called for every packet sent
        self.packet_sent = Caller()
        # Called when the link driver updates the link quality measurement
        self.link_quality_updated = Caller()
        
        self.param = Param(self)
        self.log = Log(self)
        self.commander = Commander(self)

        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.yawrate = 0

        self.link = link
        self.link_uri = ''

        def sim(freq=100.0):
            dt = 1.0/freq
            while True:
                # Euler step
                self.x = self.x + dt*self.vx
                self.y = self.y + dt*self.vy
                self.z = self.z + dt*self.vz
                self.yaw = self.yaw + dt*self.yawrate

                time.sleep(dt)


        t = threading.Thread(target=sim, args = ())
        t.daemon = True
        t.start()

    def open_link(self, link_uri):
        self.link_uri = link_uri

    def get_value(self, complete_name):

        if complete_name == 'kalman.stateX':
            return self.x
        elif complete_name == 'kalman.stateY':
            return self.y
        elif complete_name == 'kalman.stateZ':
            return self.z
        elif complete_name == 'kalman.varPX':
            return 0.0
        elif complete_name == 'kalman.varPY':
            return 0.0
        elif complete_name == 'kalman.varPZ':
            return 0.0
        else:
            print("Warning: {} not implemented in sim.".format(complete_name))
            return 0.0