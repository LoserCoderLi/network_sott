import os
from PIL import Image, ImageEnhance  # 导入用于图像处理的库
import numpy as np  # 导入用于处理数值计算的库
from tqdm import tqdm  # 导入用于显示进度条的库
import random  # 导入随机数生成的库

class Augmenter:
    def __init__(self, rotation=0, translation=0, scale=(1, 1), contrast=(1, 1), saturation=(1, 1), color=(1, 1), target_size=(28, 28)):
        """
        初始化图像增强器类。

        参数:
            rotation (int): 旋转角度范围。
            translation (int): 平移范围。
            scale (tuple): 缩放比例范围。
            contrast (tuple): 对比度增强范围。
            saturation (tuple): 饱和度增强范围。
            color (tuple): 色彩增强范围。
            target_size (tuple): 目标图像大小。
        """
        self.rotation = rotation
        self.translation = translation
        self.scale = scale
        self.contrast = contrast
        self.saturation = saturation
        self.color = color
        self.target_size = target_size

    def augment(self, image):
        """
        对图像应用增强操作。

        参数:
            image (PIL.Image): 输入的图像对象。

        返回:
            PIL.Image: 增强后的图像。
        """
        if self.rotation != 0:
            image = self.random_rotate(image)  # 随机旋转图像
        if self.translation != 0:
            image = self.random_translate(image)  # 随机平移图像
        if self.scale != (1, 1):
            image = self.random_scale(image)  # 随机缩放图像
        if self.contrast != (1, 1):
            image = self.random_contrast(image)  # 随机调整对比度
        if self.saturation != (1, 1):
            image = self.random_saturation(image)  # 随机调整饱和度
        if self.color != (1, 1):
            image = self.random_color(image)  # 随机调整色彩

        image = self.normalize(image)  # 归一化图像
        image = self.resize_image(image)  # 调整图像大小
        return image

    def resize_image(self, image):
        """
        调整图像大小。

        参数:
            image (PIL.Image): 输入的图像对象。

        返回:
            PIL.Image: 调整大小后的图像。
        """
        if image.size[0] == 0 or image.size[1] == 0:
            image = image.resize(self.target_size, Image.BICUBIC)  # 使用双三次插值调整图像大小
        return image

    def random_rotate(self, image):
        """
        随机旋转图像。

        参数:
            image (PIL.Image): 输入的图像对象。

        返回:
            PIL.Image: 旋转后的图像。
        """
        angle = random.uniform(-self.rotation, self.rotation)  # 随机选择旋转角度
        return image.rotate(angle)

    def random_translate(self, image):
        """
        随机平移图像。

        参数:
            image (PIL.Image): 输入的图像对象。

        返回:
            PIL.Image: 平移后的图像。
        """
        while True:
            x_translation = random.uniform(-self.translation, self.translation)  # 随机选择X轴平移距离
            y_translation = random.uniform(-self.translation, self.translation)  # 随机选择Y轴平移距离
            image_transformed = image.transform(image.size, Image.AFFINE, (1, 0, x_translation, 0, 1, y_translation))  # 应用仿射变换
            if image_transformed.size[0] > 0 and image_transformed.size[1] > 0:  # 确保平移后图像尺寸有效
                break
        return image_transformed

    def random_scale(self, image):
        """
        随机缩放图像。

        参数:
            image (PIL.Image): 输入的图像对象。

        返回:
            PIL.Image: 缩放后的图像。
        """
        while True:
            scale_factor = random.uniform(self.scale[0], self.scale[1])  # 随机选择缩放比例
            new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))  # 计算缩放后的尺寸
            if new_size[0] > 0 and new_size[1] > 0:  # 确保缩放后图像尺寸有效
                break
        return image.resize(new_size, Image.BICUBIC)

    def random_contrast(self, image):
        """
        随机调整图像对比度。

        参数:
            image (PIL.Image): 输入的图像对象。

        返回:
            PIL.Image: 对比度调整后的图像。
        """
        enhancer = ImageEnhance.Contrast(image)  # 创建对比度增强器
        contrast_factor = random.uniform(self.contrast[0], self.contrast[1])  # 随机选择对比度因子
        return enhancer.enhance(contrast_factor)

    def random_saturation(self, image):
        """
        随机调整图像饱和度。

        参数:
            image (PIL.Image): 输入的图像对象。

        返回:
            PIL.Image: 饱和度调整后的图像。
        """
        enhancer = ImageEnhance.Color(image)  # 创建饱和度增强器
        saturation_factor = random.uniform(self.saturation[0], self.saturation[1])  # 随机选择饱和度因子
        return enhancer.enhance(saturation_factor)

    def random_color(self, image):
        """
        随机调整图像色彩。

        参数:
            image (PIL.Image): 输入的图像对象。

        返回:
            PIL.Image: 色彩调整后的图像。
        """
        enhancer = ImageEnhance.Color(image)  # 创建色彩增强器
        color_factor = random.uniform(self.color[0], self.color[1])  # 随机选择色彩因子
        return enhancer.enhance(color_factor)

    def normalize(self, image):
        """
        将图像归一化到[0, 1]范围。

        参数:
            image (PIL.Image): 输入的图像对象。

        返回:
            PIL.Image: 归一化后的图像。
        """
        image = np.array(image).astype(np.float32) / 255.0  # 将图像像素值归一化到[0, 1]范围
        return Image.fromarray((image * 255).astype(np.uint8))  # 恢复到原始像素值范围，并转换为PIL图像对象


def load_images_from_folder(folder, label, target_size=(224, 224)):
    """
    从指定文件夹加载图像，并调整大小。

    参数:
        folder (str): 图像文件夹路径。
        label (int): 图像对应的标签。
        target_size (tuple): 调整后的图像尺寸。

    返回:
        tuple: 包含图像数组和对应标签的列表。
    """
    images = []
    labels = []
    for filename in os.listdir(folder):  # 遍历文件夹中的所有文件
        if filename.endswith((".png", ".jpg", ".jpeg")):  # 过滤出图像文件
            img = Image.open(os.path.join(folder, filename)).convert('RGB')  # 打开图像并转换为RGB格式
            if img is not None:
                img = img.resize(target_size, Image.BICUBIC)  # 调整图像大小
                images.append(np.array(img))  # 将图像添加到列表中
                labels.append(label)  # 添加对应的标签
    return images, labels  # 返回图像和标签列表
