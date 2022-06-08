#coding=utf-8
import math

def mod2pi(angle):
    res = (angle + math.pi) % (2 * math.pi) - math.pi
    return  res if (res > -math.pi) else (res + 2 * math.pi)
    
