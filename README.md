# Python Image Generator

**Jonah Winchell**
**Code as a Liberal Art**
**Spring 2025**

### Overview

This application takes two or more adjective-noun pairs and generates a corresponding image using nouns as base images and the adjectives to create filters over these pictures. The filtered images are then blended into one final picture.

### Usage

After downloading this package, the user must first install all necessary Python dependencies. Remember to use a Python virtual environment! Final installation can be done with the following command:

`pip3 install -r requirements.txt`

If you want to utilize non-default adjectives (see: **Adjectives**) with the GloVe language model[^1], make sure it is properly installed:

1. Go to https://nlp.stanford.edu/projects/glove/ and download the `glove.42B.300d.zip` file.
2. Unzip the file, which should create `glove.42B.300d.txt`.
3. Copy the unzipped text file to the same folder as `main.py`. There should now be 5 local files:
    - `glove.42B.300d.txt`
    - `main.py`
    - `utils.py`
    - `requirements.txt`
    - `README.md`

As well as a folder named `default_images/`. From here, the user may run the program with the following:

`python3 main.py -p adj1,noun1 adj2,noun2`

and so on with any number of adjective-noun pairs.
To specify a folder to download images into, use the `-f` flag:

`python3 main.py -p adj1,noun1 adj2,noun2 -f ~/images_here`

The program defaults to creating a `downloads/` folder for this and a `temp/` folder for intermediate image generation. The final image can be found in the `final/` folder.

### Adjectives

Any adjective can be used in this generator, though there are five built-in adjective filters with which this program can run much more quickly. These adjectives are:

- Happy
- Sad
- Confused
- Angry
- Funny

If the user inputs any other adjective, the program will automatically load the GloVe language model (this will take about a minute) to determine which filters to use and in what proportion. For example, if the user inputs `forlorn`, the program may assign the following values[^2][^3]:

```
{
    forlorn: {
        happy: 0.0,
        sad: .89,
        confused: .70,
        angry: .45,
        funny: .05
    }
}
```

Note that `happy` and `sad` are mutually exclusive. The image will then be run through all filters at their corresponding intensities. If a built-in adjective is chosen this step is skipped and the program assigns an automatic value of `1.0` for the corresponding filter.

### Nouns

Any noun (any word at all, even) can be used as a base image, though using any of the following built-in nouns allows for higher quality images:

- Mountain
- Horse
- Snail
- Tower
- Beach

If the user inputs a noun not in this list, the program automatically downloads a random image from a Google Search query to the user's disk. Please note that the program does not filter noun inputs whatsoever, any input is directly queried to Google Images and a random image will be selected.


[^1]: Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
[^2]: Values are `[0.0, 1.0]`, higher is more related
[^3]: These values may not be representative of what would actually be assigned to `forlorn`
