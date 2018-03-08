# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:35:13 2018

@author: bryan.nonni
"""

class Computation:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def addition(self): 
        print(self.x + self.y)
        
    def multiply(self):
        print(self.x * self.y)
        
#one class with multiple functions to take in one data set and execute it in different ways 
    
first_comp = Computation(3, 5)
first_comp.addition()
first_comp.multiply()
