import os
from PIL import Image

for filename in os.listdir('Flicker8k_Dataset/'):
    img = Image.open(f'Flicker8k_Dataset/{filename}').convert('LA').convert('RGB')
    img.save(f'Flicker8k_Dataset_grey/grey_{filename}')
