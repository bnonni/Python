# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:20:59 2018

@author: bryan.nonni
"""

import os
import time
from datetime import datetime
from selenium import webdriver

#

# Chrome Location, combines webshots folder to the chrome driver
firefox_path = os.path.join('.', 'geckodriver.exe')
#print(firefox_path)


def google_search_screenshots(screen_folder):
    #not passing driver in function bc we're only looking at one Driver: FF
    """
    """
    driver = webdriver.Firefox()
    driver.get('https://www.ebates.com', 'https://www.retailmenot.com', 'https://www.couponcabin.com', 'https://www.groupon.com')
    driver.implicitly_wait(5)
    driver.save_screenshot(os.path.join(screen_folder, '{}.png'.format(screen_folder)))
    driver.sleep(5)
    driver.close()
    
## Checking
data_path = datetime.strftime(datetime.today(), '%Y-%m-%d %H-%M')

if data_path in os.listdir(os.getcwd()):
    pass
else:
    print("File not found, Creating Now . . .")
    os.mkdir(data_path)

    
    google_search_screenshots(data_path)









