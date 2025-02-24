'''
Jonah Winchell
Code as a Liberal Art, Spring 2025
Utility functions for Python Image Generator
'''

import os
import math
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFilter


def pair(arg):
    ''' Custom argument type '''
    return [x for x in arg.split(',')]


def load_glove(model_file):
    ''' Load the GloVe NLP model into memory '''
    model = {}
    
    # Throw error if GloVe not installed
    if not os.path.isfile(model_file):
        print('GloVe model is not installed! Please follow instructions in README.md')
        exit(1)
    
    length = 1917494        # Model size for glove.42B.300d
    start = datetime.now()
    with open(model_file, 'r') as f:
        i = 0
        # Load data into usable schema
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            model[word] = embedding
            i += 1
            __stats(length, start, i)
    
    time = (datetime.now()-start).seconds
    print(f'\n\n\tGloVe loaded in {time} seconds         ')
    return model


def __stats(length, start, i):
    ''' Display loading messages '''

    ave_time = (datetime.now() - start).seconds / i
    seconds_left = round(int(((length - i) * ave_time) / 5)) * 5    # Rounded to nearest 5 seconds
    if seconds_left > 60:
        message = f'About a minute remaining'
    elif seconds_left > 5:
        message = f'About {seconds_left} seconds remaining'
    else:
        message = f'Almost finished laoding'

    print(f'\tLoading GloVe language model...          \n'
          f'\t{round((i / length) * 100, 3)}% done     \n'
          f'\t{message}                                  ',
          end='\r\033[A\r\033[A\r')
    

def generate_gradient(width, height):
    """ Generate a horizontal gradient """

    def sigmoid(x, width):
        ''' Gradient value is a logistic function of x '''
        l = 255
        k = 0.02
        y = width/2
        return l / (1 + math.exp(-k * (x - y)))

    # Assign pixel value based on horizontal position
    mask = Image.new('RGBA', (width, height))
    mask_data = []
    for y in range(height):
        for x in range(width):
            val = int(sigmoid(x, width))
            pixel = (val, val, val)
            mask_data.append(pixel)
    mask.putdata(mask_data)
    mask.save(f'{os.getcwd()}/temp/gradient.png')
    return mask


def generate_keyhole(width, height):
    ''' Generate a keyhole mask '''

    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    c1 = (width / 3, height / 4)
    c2 = ((width * 2) / 3, (height * 3) / 4)
    draw.ellipse((c1, c2), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(10))
    mask.save(f'{os.getcwd()}/temp/keyhole.png')
    return mask
