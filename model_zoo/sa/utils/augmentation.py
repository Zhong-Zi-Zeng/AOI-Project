from __future__ import annotations
from torchvision.transforms import *
import torch


class RandomFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        random_h_flip = RandomHorizontalFlip(p=self.prob)
        random_v_flip = RandomVerticalFlip(p=self.prob)

        input = torch.concat([image, mask], dim=0)
        input = random_h_flip(input)
        input = random_v_flip(input)
        image, mask = torch.split(input, input.shape[0] // 2, dim=0)

        return image, mask


class ColorEnhance:
    def __init__(self,
                 brightness_gain: float = 0.5,
                 contrast_gain: float = 0.5,
                 saturation_gain: float = 0.5,
                 hue_gain: float = 0.5):
        self.brightness_gain = brightness_gain
        self.contrast_gain = contrast_gain
        self.saturation_gain = saturation_gain
        self.hue_gain = hue_gain

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        return ColorJitter(brightness=self.brightness_gain,
                           contrast=self.contrast_gain,
                           saturation=self.saturation_gain,
                           hue=self.hue_gain)(image), mask


class RandomPerspective:
    def __init__(self,
                 degrees: float = 30.,
                 translate: float = 0.1,
                 scale: float = 0.9,
                 shear: float = 0.):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        random_perspective = RandomAffine(degrees=self.degrees,
                                          translate=(self.translate, self.translate),
                                          scale=(self.scale, self.scale),
                                          shear=self.shear)

        input = torch.concat([image, mask], dim=0)
        input = random_perspective(input)
        image, mask = torch.split(input, input.shape[0] // 2, dim=0)

        return image, mask


class GaussianNoise:
    def __init__(self,
                 variance: float = 0.05):
        self.variance = variance

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        noise = torch.randn(image.size()) * self.variance
        image = image.to(torch.float32) + noise

        return image, mask


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt

    # aug = Augmentation()
    image = cv2.imread(r"D:\AOI\ControllerDataset\white\patch-train\50.jpg")
    mask = cv2.imread(r"D:\AOI\ControllerDataset\white\patch-train-gt\50.jpg")
    # trans = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    rp = RandomFlip()
    input, mask = rp(transforms.ToTensor()(image), transforms.ToTensor()(mask))

    input = input.permute(1, 2, 0).cpu().numpy()
    mask = mask.permute(1, 2, 0).cpu().numpy()
    # input = input.cpu().numpy()
    # mask = mask.cpu().numpy()

    plt.subplot(121)
    plt.imshow(input)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()
