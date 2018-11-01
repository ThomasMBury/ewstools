#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:31:02 2018

@author: tb460
"""


def my_function():
  print("Hello from a function")

def add_numbers(a,b):
    return a+b

def multi_numbers(a,b):
    return a*b

def myfun(country="Angleterre"):
    print("I am from",country)
    return 'hello'
    
def myfun2(a ,b ,c=3):
    return a*b*c

def myfun3(a,b):
    return a-b

def myfun4(*varvallist):
    print("The output is")
    for varval in varvallist:
        print(varval)
    return