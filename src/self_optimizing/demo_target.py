#!/usr/bin/env python3
"""
自优化演示文件 - 包含明显需要优化的问题供系统自修正
"""

import os,sys,re
import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

#TODO fix this
def very_long_function_that_should_be_split():
    x=1
    if x>0:
        logger.info("test")
    if x<10:
        logger.info("test2")    
    if x>0:
        logger.info("test")
    if x<10:
        logger.info("test2")    
    if x>0:
        logger.info("test")
    if x<10:
        logger.info("test2")    
    return x

#
#
#
def another_function():
    pass

def duplicate_function():
    x=1
    if x>0:
        logger.info("test")
    if x<10:
        logger.info("test2")    
    return x

def duplicate_function_copy():
    x=1
    if x>0:
        logger.info("test")
    if x<10:
        logger.info("test2")    
    return x