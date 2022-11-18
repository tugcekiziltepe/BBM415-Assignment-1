import numpy as np

def FloydSteinberd(image, q):
    """
    :param image: numpy array in grayscale
    :param q: the parameter for the quantization
    :return: Dithering applied image
    """
    new_image = image.copy()
    new_image = np.array(new_image, dtype=np.float32)
    # Top to bottom
    for y in range(image.shape[0]-1):
        # Left to right
        for x in range(1, image.shape[1] - 1):
            old_pixel = new_image[y, x].copy()
            new_pixel = find_quantized_value(old_pixel, q)
            new_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel # quantization error

            new_image[y , x + 1] = new_image[y , x + 1] + quant_error * 7 / 16
            # Border check, if value is greater than 255 the image will have wide range of intensity values
            # This situation leads to more white color in the image.
            if new_image[y, x + 1] > 255:
                new_image[y, x + 1] = 255
            if new_image[y, x + 1] < 0:
                new_image[y, x + 1] = 0

            new_image[y + 1 , x - 1] = new_image[y + 1 , x - 1]  + quant_error * 3 / 16
            # Border check, if value is greater than 255 the image will have wide range of intensity values
            # This situation leads to more white color in the image.
            if  new_image[y + 1 , x - 1] > 255:
                 new_image[y + 1 , x - 1] = 255
            if  new_image[y + 1 , x - 1] < 0:
                 new_image[y + 1 , x - 1] = 0

            new_image[y + 1, x] = new_image[y + 1, x] + quant_error * 5 / 16
            # Border check, if value is greater than 255 the image will have wide range of intensity values
            # This situation leads to more white color in the image.
            if new_image[y + 1, x]> 255:
                new_image[y + 1, x] = 255
            if new_image[y + 1, x] < 0:
                new_image[y + 1, x] = 0

            # Border check, if value is greater than 255 the image will have wide range of intensity values
            # This situation leads to more white color in the image.
            new_image[y + 1, x + 1] = new_image[y + 1, x + 1] + quant_error * 1 / 16
            if new_image[y + 1, x + 1] > 255:
                new_image[y + 1, x + 1]= 255
            if new_image[y + 1, x + 1] < 0:
                new_image[y + 1, x + 1] = 0

    return new_image

def find_quantized_value(old_pixel, q):
    """
    :param old_pixel: int value
    :param q: quantization parameter
    :return new_pixel: float value
    """
    new_pixel = np.round((q * old_pixel) / 255) * (255 / q)
    return new_pixel

