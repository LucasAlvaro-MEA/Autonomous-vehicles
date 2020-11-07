__author__ = 'Bitcraze AB'
__all__ = ['Caller']


class Caller():
    """ An object were callbacks can be registered and called """

    def __init__(self):
        """ Create the object """
        self.callbacks = []

    def add_callback(self, cb):
        """ Register cb as a new callback. Will not register duplicates. """
        if ((cb in self.callbacks) is False):
            self.callbacks.append(cb)

    def remove_callback(self, cb):
        """ Un-register cb from the callbacks """
        self.callbacks.remove(cb)

    def call(self, *args):
        """ Call the callbacks registered with the arguments args """
        copy_of_callbacks = list(self.callbacks)
        for cb in copy_of_callbacks:
            cb(*args)