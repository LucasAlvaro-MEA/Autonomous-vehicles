import time
import threading

from .callbacks import Caller


class LogVariable():
    def __init__(self, name=''):
        self.name = name

class LogConfig(object):
    def __init__(self, name, period_in_ms):
        self.data_received_cb = Caller()

        self.cf = None

        self.period = int(period_in_ms / 10)
        self.period_in_ms = period_in_ms
        self.variables = []
        self.name = name


    def add_variable(self, name, fetch_as=None):
        self.variables.append(LogVariable(name))


    def start(self):
        def go():
            self.stopped = False
            while not self.stopped:
                ret_data = {}
                for var in self.variables:
                    ret_data[var.name] = self.cf.get_value(var.name)
                timestamp = time.time()
                self.data_received_cb.call(timestamp, ret_data, self)
                time.sleep(self.period_in_ms/1000)
        t = threading.Thread(target=go, args = ())
        t.daemon = True
        t.start()
        
    def stop(self):
        self.stopped = True

class Log():
    def __init__(self, crazyflie=None):
        self.cf = crazyflie

    def add_config(self, logconf):
        logconf.cf = self.cf