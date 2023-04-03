#!/usr/bin/env python

"""Logger parameters for the entire process.
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

import sys


import logging

logger = logging.getLogger('basic')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

# formatter = logging.Formatter('%(asctime)s - %(levelno)s - %(filename)s - '
#                               '%(funcName)s - %(message)s')
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def path_logger(filename):
    global logger
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(logging.DEBUG)

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
