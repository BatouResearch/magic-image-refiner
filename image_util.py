from PIL import Image
import cv2
import numpy as np


def resize_to_multiple_of_64(img):
    # Calculate new dimensions, rounding up to the nearest multiple of 64
    new_width = (img.width + 63) // 64 * 64
    new_height = (img.height + 63) // 64 * 64
    new_img = img.resize((new_width, new_height), resample=Image.LANCZOS)
    return new_img


def resize_for_condition_image(input_image, resolution):
    if resolution == "original":
        img = resize_to_multiple_of_64(input_image)
        return img

    img = input_image.convert("RGB")
    width, height = input_image.size
    scale_factor = float(int(resolution)) / min(height, width)
    new_height, new_width = int(round(height * scale_factor / 64)) * 64, int(round(width * scale_factor / 64)) * 64
    img = img.resize((new_width, new_height), resample=Image.LANCZOS)
    return img


def create_hdr_effect(original_image, hdr):
    cv_original = pil_to_cv(original_image)
    brightness_factors = calculate_brightness_factors(hdr)
    images = [adjust_brightness(cv_original, factor) for factor in brightness_factors]
    merge_mertens = cv2.createMergeMertens()
    hdr_image = merge_mertens.process(images)
    hdr_image_8bit = np.clip(hdr_image*255, 0, 255).astype('uint8')
    hdr_image_pil = Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))
    return hdr_image_pil


def calculate_brightness_factors(hdr_intensity):
    factors = [1.0] * 9
    if hdr_intensity > 0:
        factors = [1.0 - 0.9 * hdr_intensity, 1.0 - 0.7 * hdr_intensity, 1.0 - 0.45 * hdr_intensity,
                   1.0 - 0.25 * hdr_intensity, 1.0, 1.0 + 0.2 * hdr_intensity,
                   1.0 + 0.4 * hdr_intensity, 1.0 + 0.6 * hdr_intensity, 1.0 + 0.8 * hdr_intensity]
    return factors


def pil_to_cv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def adjust_brightness(cv_image, factor):
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v = np.clip(v * factor, 0, 255).astype('uint8')
    adjusted_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)