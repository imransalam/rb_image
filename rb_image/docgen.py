
import os
import cv2
import string
import random
import numpy as np
from skimage import io
from PIL import Image, ImageDraw, ImageFont

class DataGenerator():
    def __init__(self, image_shape=(2750, 2125, 3), num_samples=1, color_codes={}, output_dir=''):
        self.output_dir = output_dir
        self.color_codes = color_codes
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-L.ttf",20)
        self.title_font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf",60)
        self.heading_font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf",40)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, 'labels')):
            os.makedirs(os.path.join(output_dir, 'labels'))
        if not os.path.exists(os.path.join(output_dir, 'images')):
            os.makedirs(os.path.join(output_dir, 'images'))


    def get_random_word(self):
        return "".join([random.choice(string.ascii_letters) for i in range(np.random.randint(3, 8))])
    
    def get_random_line(self):
        return " ".join([self.get_random_word() for i in range(np.random.randint(60, 70))])

    def generate_text(self, num_lines):
        return "\n".join([self.get_random_line() for i in range(num_lines)])

    def draw_text(self, img, font, num_level):
        approx_num_lines = int(img.shape[0] / 12) # 1 Line has approximately 12 vertical pixels
        txt = self.generate_text(approx_num_lines) # '\t'.join(['' for i in range(num_level)])
        img = Image.fromarray(img.astype(np.uint8), 'RGB')
        draw = ImageDraw.Draw(img)
        draw.text((0,0), txt ,(0, 0,0),font=font)
        draw = ImageDraw.Draw(img)
        return np.array(img)
    
    def get_title(self, label, image): # IMPLEMENT AGAIN... POSSIBLE BUGS
        jump = np.random.randint(low=2, high=15)
        title_gap_above = np.random.randint(low=10, high=int(0.05*self.image_shape[0]))
        title_gap_below = np.random.randint(low=title_gap_above + jump, high=int(0.07*self.image_shape[0]))
        title_gap_left = np.random.randint(low=int(0.05 * self.image_shape[1]), high=int(0.85 * self.image_shape[1]))
        title_gap_right = np.random.randint(low=int(title_gap_left + 100), high=int(title_gap_left + 150))
        if title_gap_right <= title_gap_left:
            return label, image, [title_gap_above, title_gap_below, title_gap_left, title_gap_right]
        label[title_gap_above : title_gap_below, title_gap_left : title_gap_right] = self.color_codes['title']
        image[title_gap_above : title_gap_below, title_gap_left : title_gap_right] = self.draw_text(image[title_gap_above : title_gap_below, title_gap_left : title_gap_right], self.title_font, 0)
        return label, image, [title_gap_above, title_gap_below, title_gap_left, title_gap_right]

    def get_levels(self, label, image, side_gaps, level, num_levels=1, limit=(0,0)):
        if not level in self.color_codes:
            return label, image
        for i in range(num_levels):
            jump = np.random.randint(low=45, high=65)
            level_gap_above = np.random.randint(low=limit[0] + 40 , high=limit[0] + jump)
            level_gap_below = np.random.randint(low=level_gap_above + jump * (2 / int(level[-1])), high=level_gap_above + jump * (10 / int(level[-1])) )
            if level_gap_below >= limit[1]:
                continue
            label[level_gap_above : level_gap_below, side_gaps[0] : side_gaps[1]] = label[level_gap_above : level_gap_below, side_gaps[0] : side_gaps[1]] + self.color_codes[level]
            image[level_gap_above : level_gap_below, :] = [1,1,1]
            image[level_gap_above : level_gap_below, side_gaps[0] : side_gaps[1]] = self.draw_text(image[level_gap_above : level_gap_below, side_gaps[0] : side_gaps[1]], self.font, int(level[-1]))
            temp_new_level = 'level' + str(int(level[-1]) + 1)
            label, image = self.get_levels(label, image, [side_gaps[0] + np.random.randint(low=30, high=75), side_gaps[1] - np.random.randint(low=30, high=75)], temp_new_level, num_levels=np.random.randint(low=1, high=10), limit=(level_gap_above + np.random.randint(low=1, high=10), level_gap_below - np.random.randint(low=1, high=10)))
            limit = (level_gap_below, limit[1])
        return label, image

    def save(self, label, image, name):
        io.imsave(os.path.join(self.output_dir, 'labels', name + '.jpg'), label)
        io.imsave(os.path.join(self.output_dir, 'images', name + '.jpg'), image)

    def generator(self):
        for i in range(self.num_samples):
            try:
                if i % 10 == 0:
                    print('Samples Done: ', i)
                label = np.zeros(self.image_shape, dtype=np.uint8)
                image = np.ones(self.image_shape, dtype=np.uint8)
                label, image, title_gaps = self.get_title(label, image)

                level1_gap_left = np.random.randint(low=int(0.05 * self.image_shape[1]), high=int(0.15 * self.image_shape[1]))
                level1_gap_right = np.random.randint(low=int(0.85 * self.image_shape[1]), high=int(0.95 * self.image_shape[1]))
                label, image = self.get_levels(label, image, [level1_gap_left, level1_gap_right], 'level1', num_levels=np.random.randint(low=1, high=10), limit=(title_gaps[1],self.image_shape[0]))
                
                self.save(label, image * 255, str(i))
            except Exception as e:
                print(e)

            






