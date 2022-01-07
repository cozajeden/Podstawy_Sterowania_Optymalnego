# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 08:04:53 2021

@author: student
"""
import numpy as np
from numpy import array as arr
from numpy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def zad2_1():
    x = 3**12-5
    print(x)
    
    x = arr([[2, 0.5]])@\
        arr([[1, 4],[-1, 3]])@\
        arr([[-1],[-3]])
    print(x)
    
    x = arr([
        [1, -2, 0],
        [-2, 4, 0],
        [2, -1, 7]])
        
    x = linalg.matrix_rank(x)
    print(x)
    
    b = arr([[-1], [2]])
    a = arr([[1, 2], [-1, 0]])
    x = linalg.solve(a, b)
    print(x)
    
def zad2_2():
    a = arr([1, 1, -129, 171, 1620])
    p = np.poly1d(a)
    print('dla x=-46, y={0}'.format(p(-46)))
    print('dla x=14, y={0}'.format(p(14)))
    
    
def zad3_1():
    a = arr([1, 1, -129, 171, 1620])
    p = np.poly1d(a)
    space = np.arange(-46, 15)
    y_min = float('inf')
    y_max = float('-inf')
    for x in space:
        y = p(x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    print('y min={0}'.format(y_min))
    print('y max={0}'.format(y_max))
    
def zad4_1(a, b, c, d, e, bounds, accuracy=100):
    p = np.poly1d(arr([a, b, c, d, e]))
    space = np.linspace(*bounds, accuracy)
    y_min = float('inf')
    y_max = float('-inf')
    for x in space:
        y = p(x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    print('y min={0}'.format(y_min))
    print('y max={0}'.format(y_max))
    
def zad4_2(*coefficients, bounds, accuracy=100):
    p = np.poly1d(arr(coefficients))
    space = np.linspace(*bounds, accuracy)
    y_min = float('inf')
    y_max = float('-inf')
    for x in space:
        y = p(x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    print('y min={0}'.format(y_min))
    print('y max={0}'.format(y_max))
    
def zad5_1_2(*coefficients, bounds, accuracy=100):
    p = np.poly1d(arr(coefficients))
    space = np.linspace(*bounds, accuracy)
    y_min = float('inf')
    x_min = float('inf')
    y_max = float('-inf')
    x_max = float('-inf')
    y = arr([])
    for x in space:
        temp = p(x)
        y = np.append(y, [temp])
        if temp < y_min:
            y_min = temp
            x_min = x
        if temp > y_max:
            y_max = temp
            x_max = x
    plt.plot(space, y)
    plt.plot([x_min], [y_min], 'o')
    plt.plot([x_max], [y_max], 'o')
    plt.legend(['polynomial', 'minimum', 'maximum'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(''.join(['{0}{1}x'.format('+' if c >= 0 else '',c) for c in coefficients]))
    plt.tight_layout()
    plt.show()
    
    
zad2_1()
zad2_2()
zad3_1()
zad4_1(1, 1, -129, 171, 1620, (-46, 14))
zad4_2(1, 1, -129, 171, 1620, bounds=(-46, 14), accuracy=1000)
zad5_1_2(1, 1, -129, 171, 1620, bounds=(-46, 14), accuracy=1000)
