import imageio
import os
import re
import numpy as np

os.chdir('images')
filenames = [file for file in os.listdir('.') if file.startswith('Berlin_hour')]
hours = [int(re.findall('Berlin_hour_(\d+).png', file)[0]) for file in filenames]
sortIdx = np.argsort(hours)
filenames = np.asarray(filenames)[sortIdx]
images = []
for filename in filenames:
    print('reading', filename)
    images.append(imageio.imread(filename))
imageio.mimwrite('Rad-Puls-Berlin.gif', images, **{'fps': 10})

print('done')