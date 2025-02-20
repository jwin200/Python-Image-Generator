'''
Jonah Winchell
Code as a Liberal Art, Spring 2025
Utility methods for Python Image Generator
'''

import os
import datetime
import numpy as np


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