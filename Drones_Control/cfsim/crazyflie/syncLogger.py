from queue import Queue
from cfsim.crazyflie.syncCrazyflie import SyncCrazyflie

class SyncLogger:
    DISCONNECT_EVENT = 'DISCONNECT_EVENT'

    def __init__(self, crazyflie, log_config):
        """
        Construct an instance of a SyncLogger
        Takes an Crazyflie or SyncCrazyflie instance and one log configuration
        or an array of log configurations
        """
        if isinstance(crazyflie, SyncCrazyflie):
            self._cf = crazyflie.cf
        else:
            self._cf = crazyflie

        if isinstance(log_config, list):
            self._log_config = log_config
        else:
            self._log_config = [log_config]

        self._queue = Queue()

        self._is_connected = False

    def connect(self):
        if self._is_connected:
            raise Exception('Already connected')

        for config in self._log_config:
            self._cf.log.add_config(config)
            config.data_received_cb.add_callback(self._log_callback)
            config.start()

        self._is_connected = True

    def disconnect(self):
        if self._is_connected:
            for config in self._log_config:
                config.stop()

                config.data_received_cb.remove_callback(
                    self._log_callback)

            self._is_connected = False

    def is_connected(self):
        return self._is_connected

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if not self._is_connected:
            raise StopIteration

        data = self._queue.get()

        if data == self.DISCONNECT_EVENT:
            self._queue.empty()
            raise StopIteration

        return data

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        self._queue.empty()

    def _log_callback(self, ts, data, logblock):
        self._queue.put((ts, data, logblock))

    def _disconnected(self, link_uri):
        self.disconnect()
        self._queue.put(self.DISCONNECT_EVENT)