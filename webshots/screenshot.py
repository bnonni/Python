# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:55:10 2018

@author: bryan.nonni
"""

import os
from selenium import webdriver

# Chrome Location, combines webshots folder to the chrome driver
chrome_location = os.path.join(r'.', 'chromedriver.exe')
print(chrome_location)

options = webdriver.ChromeOptions()
#options.add_experimental_option("", ['--ignore-certificate-errors'])

options.add_argument("--ignore-certificate-errors")
options.add_argument("--test-type")
options.binarylocation = "usr/bin/chromium
driver = webdriver.Chrome(chrome_options=options)

driver.get('https://www.google.com')
driver.save_screenshot("screenshot.png")
 
driver.close()

# Driver
#Driver = webdriver.Chrome(chrome_location, chrome_options=options)

#if __name__ == "__main__":
#   print("Code is live!")