from threading import Event

from cfsim.crazyflie import Crazyflie

class SyncCrazyflie:

    def __init__(self, link_uri, cf=None):
        """ Create a synchronous Crazyflie instance with the specified
        link_uri """

        if cf:
            self.cf = cf
        else:
            self.cf = Crazyflie()

        self._link_uri = link_uri
        self._connect_event = None
        self._disconnect_event = None
        self._is_link_open = False
        self._error_message = None

    def open_link(self):
        if (self.is_link_open()):
            raise Exception('Link already open')

        print('Connecting to %s' % self._link_uri)

        self.cf.open_link(self._link_uri)
        
        self._connected(self._link_uri)

    def __enter__(self):
        self.open_link()
        return self

    def close_link(self):
        if (self.is_link_open()):
            self._disconnect_event = Event()
            self.cf.close_link()
            self._disconnect_event.wait()
            self._disconnect_event = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_link()

    def is_link_open(self):
        return self._is_link_open

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

    def _connection_failed(self, link_uri, msg):
        """Callback when initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self._is_link_open = False
        self._error_message = msg
        if self._connect_event:
            self._connect_event.set()

    def _disconnected(self, link_uri):
        self._remove_callbacks()
        self._is_link_open = False
        if self._disconnect_event:
            self._disconnect_event.set()