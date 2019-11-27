# -*- coding: utf-8 -*-
"""
Created on Thurs Sept 20 2018 12:52:10

@author: bryan.nonni
"""

import os
import time
from datetime import datetime
from selenium import webdriver
#from selenium.webdriver.common.proxy import FirefoxProfile
from selenium.webdriver.firefox.options import Options
#import pandas as pd
#Adding Headless/No browser Screnshots - aka the browser doesnt pop up!!!
options = Options()
options.add_argument("--headless")
#options.add_argument("chrome")

#Creating the webdriver
driver = webdriver.Firefox(firefox_options=options)

# Chrome Location, combines webshots folder to the chrome driver
firefox_path = os.path.join('.', 'geckodriver.exe')
#print(firefox_path)


def google_search_screenshots(screen_folder):
    #not passing driver in function bc we're only looking at one Driver: FF
    """
    """
    print (driver.get_window_size())
    
    driver.set_window_size(1366, 1000)
    print (driver.get_window_size())    
    count = 0
    while count <= 2:
        print(count)
        url = ["https://www.pulte.com/?at_preview_token=kCjs3STik78SqeQoRESw3sIO%2Ft7shUFvqsnsF53Dvx8%3D&at_preview_index=1_1&at_preview_listed_activities_only=true&at_preview_evaluate_as_true_audience_ids=3085739&QA=9","https://www.pulte.com/?at_preview_token=kCjs3STik78SqeQoRESw3sIO%2Ft7shUFvqsnsF53Dvx8%3D&at_preview_index=1_2&at_preview_listed_activities_only=true&at_preview_evaluate_as_true_audience_ids=3085739&QA=9", "https://www.pulte.com/?at_preview_token=kCjs3STik78SqeQoRESw3sIO%2Ft7shUFvqsnsF53Dvx8%3D&at_preview_index=1_3&at_preview_listed_activities_only=true&at_preview_evaluate_as_true_audience_ids=3085739&QA=9"]
        url = url[count]
        print(url)
        driver.get(url)
        print(time.sleep(15))
        file_name =  'Experience ' + str(count)
        driver.save_screenshot(os.path.join(screen_folder, '{}.png'.format(file_name)))
        #driver.close()
        print("Driver complete. Screenshot captured")
        count += 1
    
## Checking
data_path = datetime.strftime(datetime.today(), 'CP QA - %Y-%m-%d %H-%M')

if data_path in os.listdir(os.getcwd()):
    print("File found . . .")
    pass
else:
    print("File not found, Creating Now . . .")
    os.mkdir(data_path)

    
    google_search_screenshots(data_path)









