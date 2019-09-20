from PIL import Image
import glob
import random

# This program takes in a line of text, finds it's height and width, then crops a square out,
# whose sides are the height of the line, and is randomly chosen of locations in the line.

# This requires you to have the program in the outer directory, next to the folders of the authors which contain
# The folders of each document

num = 0
for filepath in glob.iglob('**/*.png', recursive=True):
    num +=1
    im = Image.open(filepath)
    width, height = im.size
    rand = random.randint(0, width-height)
    box = (rand, 0, rand+height, height)
    region = im.crop(box)

    # This if statement is temporary: only for testing purposes
    if num <= 10:
        # Instead of .show(), switch it to saving
        region.show()


