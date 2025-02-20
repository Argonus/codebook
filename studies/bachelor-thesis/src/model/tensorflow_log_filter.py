import logging
import tensorflow as tf

class LogFilter(logging.Filter):
    def filter(self, record):
        """
        Filter out specific TensorFlow PNG warnings while keeping all other logs.
        """
        msg = str(record.getMessage())
        if "PNG warning: iCCP: profile" in msg:
            return False
        return True

