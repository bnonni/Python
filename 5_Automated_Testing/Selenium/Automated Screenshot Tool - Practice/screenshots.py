# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:34:01 2018

@author: bryan.nonni
"""

import os
from selenium import webdriver


Driver = os.path.join(".", r"chromedriver.exe")

##options = webdriver.ChromeOptions()
##options.add_experimental

driver = webdriver.Chrome(Driver)
driver.get("https://www.linkedin.com")