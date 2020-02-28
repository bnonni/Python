#!/usr/bin/env python3

# ----------3a-3b----------- #
    
    # this is a 4 space indent


# Yes! :)
def long_function_name(
        var_one, var_two, var_three, 
        var_four):
    return var_one

foo = long_function_name("one", "two",
                         "three", "four")

# NO! Bad Programmer! :(
foo = long_function_name("one", "two",
    "three", "four")

def another_long_function_name(
    var_one, var_two, var_three,
    var_four):
    print(var_one)    

# ----------3c----------- #

# No extra indentation.
if ("this_is_one_thing" and
    "that_is_another_thing"):
    do_something()

# Add a comment
if ("this_is_one_thing" and
    "that_is_another_thing"):
    # Use this comment to distinguish
    do_something()

# Add some extra indentation on the conditional continuation line.
if ("this_is_one_thing"
        and "that_is_another_thing"):
    do_something()

# Multi-Line Constructs
my_list = [
    1, 2, 3,
    4, 5, 6,
    ]

result = some_function_that_takes_arguments(
    'a', 'b', 'c',
    'd', 'e', 'f',
    )

# ----------3d----------- #

    # this is a tab
    # this is 4 spaces
    # they are the same length
    # but spaces are preferred

# ----------3e----------- #


def this_is_a_line_of_code_that_is_79_characters_which_is_actually_really_long():
    return "STOP!"

"""This is a docstring that is 72 characters long and is not too bad"""

# ----------3f----------- #

with open('/path/to/some/file/you/want/to/read') as file_1, \
     open('/path/to/some/file/being/written', 'w') as file_2:
    file_2.write(file_1.read())

def some_really_long_function_name_that_would_never_exist(
        arg1, arg2, arg3,
        arg4, arg5, arg6):
    return "Such line. Wow."

my_dict = {
    "one": 1,
    "two": 2
}

# ----------3g----------- #
gross_wages = 1
taxable_interest = 2
dividends = 4
qualified_dividends = 5
ira_deduction = 6
student_loan_interest = 7

# Yes! Good programmer! :)
income = (gross_wages 
          + taxable_interest
          + (dividends - qualified_dividends)
          - ira_deduction
          - student_loan_interest)

# No! Bad programmer! :(
income = (gross_wages +
          taxable_interest +
          (dividends - qualified_dividends) -
          ira_deduction -
          student_loan_interest)

# ----------3h----------- #

class ClassOne:
    pass


class TwoSpacesAbove:
    pass


def Another_Two_Spaces_Above():
    return "Ok!"


class ClassTwo:
    def Method(self):
        return True

    def OneSpaceAbove(self):
        return True


def calculate_var(number_list):
    sum_list = 0
    for number in number_list:
        sum_list = sum_list + number
    mean = sum_list / len(number_list)

    sum_squares = 0
    for number in number_list:
        sum_squares = sum_squares + number**2
    mean_squares = sum_squares / len(number_list)

    return mean_squares - mean**2


# ----------3j----------- #
# Yes!
import os
import sys
from subprocess import Popen, PIPE

# No!
import os, sys

# Yes!
import mypkg.sibling
from mypkg import sibling
from mypkg.sibling import example

from . import sibling
from .sibling import example

# Avoid wildcard imports 
from os import *

# ---------_4_---------- #

from __future__ import barry_as_FLUFL

__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Cardinal Biggles'

import os
import sys

# ---------_5_----------- #
# Recommend single quotes by default because single quotes dominate all
# special characters in Bash (e.g. echo '$USER' prints $USER)

'Strings in single quotes'
'Makes it easier to use "double quotes" inside the single quotes'


def spam(a, b):
    return 
ham = []
eggs = {}

# Yes
spam(ham[1], {eggs: 2})

# No
spam( ham[ 1 ], { eggs: 2 } )

# Yes
foo = (0,)

# No
bar = (0, )

# Yes
if x == 4: print(x, y); x, y = y, x

# No
if x == 4 : print(x, y) ; x , y = y , x




# Yes
ham[1:9], ham[1:9:3], ham[:9:3], ham[1::3], ham[1:9:]
ham[lower:upper], ham[lower:upper:], ham[lower::step]
ham[lower+offset : upper+offset]
ham[: upper_fn(x) : step_fn(x)], ham[:: step_fn(x)]
ham[lower + offset : upper + offset] 

# # No
ham[lower + offset:upper + offset]
ham[1: 9], ham[1 :9], ham[1:9 :3]
ham[lower : : upper]
ham[ : upper]  

# Yes
spam(1)

# No
spam (1)

# Yes
dct['key'] = lst[index]

# No
dct ['key'] = lst [index]




# Yes
x = 1
y = 2
long_variable = 3

# No
x             = 1
y             = 2
long_variable = 3

# Yes
i = i + 1
submitted += 1
x = x*2 - 1
hypot2 = x*x + y*y
c = (a+b) * (a-b)
a <= b

# No
i=i+1
submitted +=1
x = x * 2 - 1
hypot2 = x * x + y * y
c = (a + b) * (a - b)
a<=b

# Yes
def munge(input: AnyStr): ...
def munge() -> PosInt: ...

# No
def munge(input:AnyStr): ...
def munge()->PosInt: ...

# Yes
def complex(real, imag=0.0):
    return magic(r=real, i=imag)

def munge(sep: AnyStr = None):
    return "Yes"

def munge(input: AnyStr, sep: AnyStr = None, limit=1000):
    return "Yes"

# No
def complex(real, imag = 0.0):
    return magic(r = real, i = imag)

def munge(input: AnyStr=None):
    return "No"

def munge(input: AnyStr, limit = 1000):
    return "No" 

# Yes
if foo == 'blah':
    do_blah_thing()

do_one()
do_two()
do_three()

# No
if foo == 'blah': do_blah_thing()
do_one(); do_two(); do_three()

# Prob naw
if foo == 'blah': do_blah_thing()
for x in lst: total += x
while t < 10: t = delay()

# Definitely naw!
if foo == 'blah': do_blah_thing()
else: do_non_blah_thing()

try: something()
finally: cleanup()

do_one(); do_two(); do_three(long, argument,
                             list, like, this)

if foo == 'blah': one(); two(); three()

# Yes
FILES = ('setup.cfg',)

for i in range(0, 10):
    # Loop over i ten times and print out the value od i, followed by a
    # new line character
    print(i, '\n')

def quadratic(a, b, c, x):
    # Calculate the solution to a quadratic dquation using the quadratic
    # formula.  Use 2 spaces after the period.
    # 
    # There are always two solutions to a quadratic equation, x_1 and x_2
    x_1 = (- b+(b**2-4*a*c)**(1/2)) / (2*a)
    x_2 = (- b-(b**2-4*a*c)**(1/2)) / (2*a)
    return x_1, x_2

# No
x = x + 1                 # Increment x

# Yes
x = x + 1                 # Compensate for border


"""This is a docstring"""
'''Also a docstring'''



"""This docstring is 72 characters long and when it gets too long go to

the next line. Put the closing quotes on its own line.
"""

"""Return a foobang

Optional plotz says to frobnicate the bizbaz first.
"""

# Function
def function_snake_case(): ...

# Package and Module
package.py
module.py

# Class
class ClassCamelCase:
    # Method
    def method_snake_case():
        # Constant
        CONSTANT_VARIABLE = True
        return CONSTANT_VARIABLE

class ClassName:
    def _private_method(): ...

class A:
    def _single_method(self):
        pass    def __double_method(self): # for mangling
        pass class B(A):
    def __double_method(self): # for mangling
        pass

Tkinter.Toplevel(master, class_='ClassName')

try:
    if item[1][2]=='1':
    qtytype='0'
    qty=str(item[1][0])
    items=item[1][0]
    amt='0.0'
    tot_amt=str(float(item[1][1])/100)
#              tot_amt=str(float(int(item[1][0])*int(item[1][1]))/100)
    else:
    qtytype='1'
    items=item[1][0]
    amt='0.0'
    tot_amt=str(float(item[1][1])/100)
    qty=0
    for entry in item[1][4]:
        qty+=entry[0]
except TypeError:
    eType, eValue, eTraceback = sys.exc_info()
    print >> sys.stderr, time.strftime("%Y-%m-%d %H:%M:%S"), str(traceback.format_exception(eType,eValue,eTraceback))

# Instead of this ...
f = lambda x: 2*x 

# Use this ...
def f(x): return 2*x


# Yes
try:
    value = collection[key]
except KeyError:
    return key_not_found(key)
else:
    return handle_value(value)

# No
try:
    # Too broad!
    return handle_value(collection[key])
except KeyError:
    # Will also catch KeyError raised by handle_value()
    return key_not_found(key)


# Yes
with conn.begin_transaction():
    do_stuff_in_transaction(conn)

# No
with conn:
    do_stuff_in_transaction(conn)


# Yes
def foo(x):
    if x >= 0:
        return math.sqrt(x)
    else:
        return None

def bar(x):
    if x < 0:
        return None
    return math.sqrt(x)

# No
def foo(x):
    if x >= 0:
        return math.sqrt(x)

def bar(x):
    if x < 0:
        return
    return math.sqrt(x)

# Yes
if foo.startswith('bar'):

# No
if foo[:3] == 'bar':

# Yes
if isinstance(obj, int):

# No
if type(obj) is type(1):

# Yes
if not seq:
if seq:

# No
if len(seq):
if not len(seq):


# Yes, Good!
if greeting:

# No, Bad!
if greeting == True:

# Nope, Worse!
if greeting is True:

def foo():
    try:
        1 / 0
    finally:
        return 42



# Semantics

