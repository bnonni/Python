# -*- coding: utf-8 -*-
"""
Created on Thurs Sept 20 2018 12:52:10

@author: bryan.nonni
"""

import os
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as firefox_options
from selenium.webdriver.chrome.options import Options as chrome_options

#Adding Headless/No browser Screnshots - aka the browser doesnt pop up!!!
print ('Passing options')
ffoptions = firefox_options()
ffoptions.add_argument("--headless")
choptions = chrome_options()
choptions.add_argument("--headless")
print ('Options passed')

# Firefox Location, combines webshots folder to the gecko driver
print ('Joining path + webdriver')
firefoxPath = os.path.join('.', 'geckodriver.exe')
chromePath = os.path.join('.', 'chromedriver.exe')
print ('Joining complete')

def firefox_screenshots(screen_folder):
    """
    firefox_screenshots function executes geckodriver.exe to visit a QA link and screenshot the experience 
    """         
    print ('firefox_screenshots function:: Started.')
    #Creating the webdriver
    print('Creating geckodriver')
    firefoxDriver = webdriver.Firefox(firefox_options=ffoptions)
    print ('Geckodriver created')
    print (firefoxDriver.get_window_size())
    firefoxDriver.set_window_size(1366, 1000)
    print (firefoxDriver.get_window_size())    
    version = 1
    while version <= 3:
        print(version)
        strVersion = str(version)
        url = "https://www.pulte.com/?at_preview_token=kCjs3STik78SqeQoRESw3sIO%2Ft7shUFvqsnsF53Dvx8%3D&at_preview_index=1_"+strVersion+"&at_preview_listed_activities_only=true&at_preview_evaluate_as_true_audience_ids=3085739&QA=9"
        print(url)
        firefoxDriver.get(url)
        sleep = time.sleep(15)
        print(sleep)
        file_name =  'Experience ' + str(version) +  datetime.strftime(datetime.today(), '%Y-%m-%d %H-%M-%S')
        firefoxDriver.save_screenshot(os.path.join(screen_folder, '{}.png'.format(file_name)))
        print("Experience A complete. Screenshot captured. Restarting.")
        version += 1
    chrome_screenshots(screen_folder)

ffDataPath = datetime.strftime(datetime.today(), 'Pulte - FF - CTA Optimization QA - %Y-%m-%d')
if ffDataPath in os.listdir(os.getcwd()):
    pass
else:
    print("Firefox file not found, Creating Now . . .")
    os.mkdir(ffDataPath)
    firefox_screenshots(ffDataPath)

def chrome_screenshots(screen_folder):
    """
    chrome_screenshots function executes chromedriver.exe to visit a QA link and screenshot the experience
    """
    chromeDriver = webdriver.Chrome(chrome_options=choptions)
    chDataPath = datetime.strftime(datetime.today(), 'Pulte - CH - CTA Optimization QA - %Y-%m-%d')
    screen_folder = chDataPath
    if chDataPath in os.listdir(os.getcwd()):
        print("Chrome file found . . .")
        pass
    else:
        print("Chrome file not found, Creating Now . . .")
        os.mkdir(chDataPath)
    
    chromeDriver.set_window_size(1366, 1000)
    print (chromeDriver.get_window_size())    
    version = 1;
    while version <= 3:
        print(version)
        strVersion = str(version)
        url = "https://www.pulte.com/?at_preview_token=kCjs3STik78SqeQoRESw3sIO%2Ft7shUFvqsnsF53Dvx8%3D&at_preview_index=1_"+strVersion+"&at_preview_listed_activities_only=true&at_preview_evaluate_as_true_audience_ids=3085739&QA=9"
        print(url)
        chromeDriver.get(url)
        sleep = time.sleep(15)
        print(sleep)
        file_name =  'Experience ' + str(version) +  datetime.strftime(datetime.today(), '%Y-%m-%d %H-%M-%S')
        chromeDriver.save_screenshot(os.path.join(screen_folder, '{}.png'.format(file_name)))
        print("Experience A complete. Screenshot captured. Restarting.")
        version += 1

 








