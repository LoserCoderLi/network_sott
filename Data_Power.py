import os
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm
import random

class Augmenter:
    def __init__(self, rotation=0, translation=0, scale=(1, 1), contrast=(1, 1), saturation=(1, 1), color=(1, 1), target_size=(28, 28)):
        self.rotation = rotation
        self.translation = translation
        self.scale = scale
        self.contrast = contrast
        self.saturation = saturation
        self.color = color
        self.target_size = target_size

    def augment(self, image):
        if self.rotation != 0:
            image = self.random_rotate(image)
        if self.translation != 0:
            image = self.random_translate(image)
        if self.scale != (1, 1):
            image = self.random_scale(image)
        if self.contrast != (1, 1):
            image = self.random_contrast(image)
        if self.saturation != (1, 1):
            image = self.random_saturation(image)
        if self.color != (1, 1):
            image = self.random_color(image)

        image = self.normalize(image)
        image = self.resize_image(image)
        return image

    def resize_image(self, image):
        if image.size[0] == 0 or image.size[1] == 0:
            image = image.resize(self.target_size, Image.BICUBIC)
        return image

    def random_rotate(self, image):
        angle = random.uniform(-self.rotation, self.rotation)
        return image.rotate(angle)

    def random_translate(self, image):
        while True:
            x_translation = random.uniform(-self.translation, self.translation)
            y_translation = random.uniform(-self.translation, self.translation)
            image_transformed = image.transform(image.size, Image.AFFINE, (1, 0, x_translation, 0, 1, y_translation))
            if image_transformed.size[0] > 0 and image_transformed.size[1] > 0:
                break
        return image_transformed

    def random_scale(self, image):
        while True:
            scale_factor = random.uniform(self.scale[0], self.scale[1])
            new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
            if new_size[0] > 0 and new_size[1] > 0:
                break
        return image.resize(new_size, Image.BICUBIC)

    def random_contrast(self, image):
        enhancer = ImageEnhance.Contrast(image)
        contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
        return enhancer.enhance(contrast_factor)

    def random_saturation(self, image):
        enhancer = ImageEnhance.Color(image)
        saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
        return enhancer.enhance(saturation_factor)

    def random_color(self, image):
        enhancer = ImageEnhance.Color(image)
        color_factor = random.uniform(self.color[0], self.color[1])
        return enhancer.enhance(color_factor)

    def normalize(self, image):
        image = np.array(image).astype(np.float32) / 255.0
        return Image.fromarray((image * 255).astype(np.uint8))


def load_images_from_folder(folder, label, target_size=(224, 224)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            if img is not None:
                img = img.resize(target_size, Image.BICUBIC)  # 调整图片大小
                images.append(np.array(img))
                labels.append(label)
    return images, labels