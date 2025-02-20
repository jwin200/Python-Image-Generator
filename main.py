'''
Jonah Winchell
Code as a Liberal Art, Spring 2025
Project 1: Image Generator
'''

try:
    import os
    import torch
    import random
    import requests
    import argparse
    import numpy as np
    from utils import *
    from io import BytesIO
    from bs4 import BeautifulSoup as bs
    from perlin_noise import PerlinNoise
    from PIL import Image, ImageFilter, ImageEnhance
except ImportError:
    print("Missing necessary dependencies! Install with: ")
    print("pip3 install -r requirements.txt")
    print("Please try again after installation")
    exit(1)

args = {}
glove = {}
model_file = 'glove.42B.300d.txt'
default_adjectives = ['happy', 'sad', 'confused', 'angry', 'funny']
default_nouns = ['beach', 'mountain', 'snail', 'tower', 'horse']


def __parse():
    ''' Parse command line arguments '''
    parser = argparse.ArgumentParser(description = 'Get object pairs',
                                    formatter_class = argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('-p', '--pairs',
                        metavar='D',
                        dest='pairs',
                        type=pair,
                        nargs='+',
                        help='List of adjective-noun pairs. In form of "-p happy,watermelon sad,geriatric"')
    
    parser.add_argument('-f', '--folder',
                        metavar='F',
                        dest='folder',
                        nargs='?',
                        default='downloads',
                        help='Optional name of desired images folder, defaults to "downloads"')
    
    return vars(parser.parse_args())


def __main():
    ''' Initialize global variables and run scripts '''
    global args, glove
    args = __parse()
    loaded = False
    inputs = {}

    # Throw error if not enough pairs are given
    if len(args['pairs']) < 2:
        print("Please enter two or more adjective-noun pairs!")
        exit(1)

    # Retrieve base images and evaluate adjectives
    pairs = args['pairs']
    for a, n in pairs:
        if a not in default_adjectives and not loaded:
            print(f'Custom adjectives detected!')
            glove = load_glove(model_file)
            loaded = True
        rel = evaluate_adjective(a)
        img = download_images(n)
        inputs[n] = [rel, img]

    # Filter and combine images
    filtered_images = []
    for description in inputs.values():
        filtered_images.append(filter_image(description))
    final_img = combine(filtered_images)

    # Save final image
    path = f'{os.getcwd()}/final'
    if not os.path.exists(path):
        os.makedirs(path)
    final_img = final_img.convert('RGB')
    final_img.save(f'{path}/{pairs[0][1]}.jpg')

    
def combine(images):
    ''' Given two or more images, combine them with one of three methods '''
    methods = ['blend', 'keyhole', 'gradient']

    def blend(img1, img2):
        # Blend two images 50/50
        img1 = img1.convert(mode='RGBA')
        img2 = img2.convert(mode='RGBA')
        img1 = img1.resize(img2.size)

        img = Image.blend(img1, img2, .5)
        return img
    
    def keyhole(img1, img2):
        # View one image through a circular mask of the other
        img1 = img1.resize(img2.size)
        width, height = img2.size
        mask = generate_keyhole(width, height)

        img = Image.composite(img1, img2, mask)
        return img
    
    def gradient(img1, img2):
        # Kind of a 'sweep' filter, horizontal gradient mask
        img1 = img1.resize(img2.size)
        width, height = img1.size
        mask = generate_gradient(width, height).convert(mode='L')

        img = Image.composite(img1, img2, mask)
        return img
        
    # Run all images through a random method
    img = None
    for i in images:
        if img is None:
            img = i
        else:
            i = i.resize(img.size)
            method = random.choice(methods)
            img = eval(f'{method}(img, i)')

    return img


def filter_image(desc):
    ''' Take complete image descriptions and apply custom filters '''
    adjective, emotions = list(desc[0].items())[0]
    image_path = desc[1].split('.')[0]
    img = Image.open(f'{image_path}.jpg').convert(mode='RGB')

    def happy(img, x):
        # Add sun
        width, height = img.size
        sun = Image.open('default_images/sun.png')
        sun.thumbnail((int(height/2), int(height/2)))       # Reduce sun's size to match image
        img.paste(sun, (int(width - sun.size[0]), 0), sun)  # Overlay sun onto top right corner

        # Shift hue to yellow
        const = int(x * 50)
        data = img.getdata()
        new_data = []
        for p in data:
            pixel = (p[0]+const, p[1]+const, p[2])          # Add constant to R and G values
            new_data.append(pixel)
        img.putdata(new_data)

        # Darken to compensate for hue change
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1-(x/10))
        return img
    
    def sad(img, x):
        # Shift hue to blue
        const = int(x * 50)
        data = img.getdata()
        new_data = []
        for p in data:
            pixel = (p[0]-const, p[1]-const, p[2]+const)
            new_data.append(pixel)
        img.putdata(new_data)
        return img
    
    def confused(img, x):
        # Blur the image for confused
        img = img.filter(ImageFilter.GaussianBlur(int(x * 3)))
        return img
    
    def angry(img, x):
        # Make some Perlin Noise
        noise = PerlinNoise(octaves=10, seed=1)
        xpix, ypix = img.size
        pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
        noise = Image.fromarray(np.array(pic) * 255, 'L').convert(mode='RGBA')

        # Blend noise into base image
        img = img.convert(mode='RGBA')
        img = Image.blend(img, noise, (x / 4))

        # Shift hue to red
        const = int(x * 50)
        data = img.getdata()
        new_data = []
        for p in data:
            pixel = (p[0]+const, p[1], p[2])
            new_data.append(pixel)
        img.putdata(new_data)
        return img
    
    def funny(img, x):
        # Add some random colors
        const = int(x * 25)
        data = img.getdata()
        new_data = []
        for p in data:
            r = ((random.randrange(3)) - 1) * const
            pixel = (p[0]+r, p[1]-r, p[2])
            new_data.append(pixel)
        img.putdata(new_data)

        # Reduce color gamut to between 8 and 128
        quantize_factor = int(128 - (x * 120))
        img = img.quantize(quantize_factor)
        return img
    
    # For every default emotion, apply the corresponding filter with varying intensities
    for e in emotions.keys():
        intensity = emotions[e][1]
        if intensity != 0.0:
            img = eval(f'{e}(img, intensity)')

    # Save final image
    img = img.convert('RGB')
    if not os.path.exists('temp'):
        os.makedirs('temp')
    img.save(f'temp/temp-new.jpg')
    
    return img


def download_images(query):
    ''' Download requested images to local folder, return path to image '''
    params = {
        "q": query,         # Search query
        "tbm": "isch",      # Image results
        "safe": True        # Safe search
    }

    # Check for default images
    if query in default_nouns:
        final_path = f'default_images/{query}.jpg'
        return final_path

    # Google image search
    html = requests.get("https://www.google.com/search", params=params, timeout=30)
    soup = bs(html.content, features="html.parser")
    soup.prettify()

    # Select random image
    search_results = soup.select('div img')
    i = random.randint(1, len(search_results)-1)
    image_url = search_results[i]['src']

    # Save image
    image = requests.get(image_url).content
    path = f'{os.getcwd()}/{args["folder"]}'
    final_path = f'{path}/{query}.jpg'
    if not os.path.exists(path):
        os.makedirs(path)

    img = Image.open(BytesIO(image)).convert(mode='RGB')
    img.save(final_path)

    return final_path


def evaluate_adjective(word):
    ''' Find the relatedness between default and custom adjectives '''
    params = {
        word: {
            'happy': [0, 0],        # 'happy': [Euclidian Distance, Cosine Similarity]
            'sad': [0, 0],
            'confused': [0, 0],
            'angry': [0, 0],
            'funny': [0, 0],
        }
    }

    # Catch and return default adjectives
    if word in params[word].keys():
        params[word][word] = [1.0, 1.0]
        return params

    # Throw error if adjective is not in corpus
    try:
        word_vec = torch.from_numpy(glove[word])
    except Exception:
        print(f'I\'m sorry, I don\'t recognize {word}!')
        exit(1)
    
    # Evaluate adjective relation
    descriptors = params[word]
    for desc in descriptors:
        sample_vec = torch.from_numpy(glove[desc])
        euc_dist = torch.linalg.vector_norm(word_vec - sample_vec).item()                           # Euclidian distance
        cos_sim = torch.cosine_similarity(word_vec.unsqueeze(0), sample_vec.unsqueeze(0)).item()    # Cosine Similarity
        descriptors[desc] = [euc_dist, cos_sim]

    # Normalize values --> [0, 1], higher is more related
    # Round all numbers to three decimal places
    new_vals = {}
    for key, value in descriptors.items():
        x = 1 - (value[0] / 10)
        new_vals[key] = [round(x, 3), round(descriptors[key][1], 3)]
    params[word] = new_vals

    # Make happy and sad mutually exclusive
    if params[word]['happy'][1] > params[word]['sad'][1]:
        params[word]['sad'] = [0, 0]
    else:
        params[word]['happy'] = [0, 0]

    return params

 
if __name__ == "__main__":
    __main()
