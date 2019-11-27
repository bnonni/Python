# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:20:59 2018

@author: bryan.nonni
"""

import os
#import time
from datetime import datetime
from selenium import webdriver

#

# Chrome Location, combines webshots folder to the chrome driver
firefox_path = os.path.join('.', 'geckodriver.exe')
#print(firefox_path)


def ray_flan_get_txt(screen_folder):
    #not passing driver in function bc we're only looking at one Driver: FF
    """
    """
    driver = webdriver.Firefox()
    driver.get("https://webapi.raymourflanigan.com/api/product/feed?type=google&delimiter=%7C&encoding=UTF-8")
    driver.implicitly_wait(10)
    #time.sleep(10)
    driver.find_element_by_css_selector('pre').get(text)
            
    driver.close()
    
## Checking
data_path = datetime.strftime(datetime.today(), '%Y-%m-%d %H-%M')

if data_path in os.listdir(os.getcwd()):
    pass
else:
    print("File not found, Creating Now . . .")
    os.mkdir(data_path)

    
    ray_flan_get_txt(data_path)









