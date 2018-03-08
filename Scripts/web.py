# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time
import webbrowser

websites = ['www.youtube.com', 'www.facebook.com', 'www.google.com']


for website in websites:
    webbrowser.open(website)
    time.sleep(5)