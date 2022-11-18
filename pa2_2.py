import numpy as np
import math

def take_log(image):
    """
    It takes logarithm of given image
    Parameters
    ----------
    image : numpy_darray
    """
    for dim in range(image.shape[2]):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                number = image[y, x, dim]
                if number != 0:
                    image[y, x, dim] = math.log10(number)
                else: 
                    image[y, x, dim] = math.log10(1)

def colorTransfer(source, target):
    """
    It transfers color characteristics of an image to another.
    Parameters
    ----------
    source : numpy_darray
        source image
    target : numpy_darray
        target image
    Return
    ----------
    RGB_source : numpy_darray
        source image which is converted
    """

    # 1. Apply the given transformation to convert RGB source and target images to LMS cone space
    RGB_to_LMS_filter = np.array([
    [0.3811, 0.5783, 0.0402],
    [0.1967, 0.7244, 0.0782],
    [0.0241, 0.1288, 0.8444]])

    LMS_source = np.dot(source, RGB_to_LMS_filter.T)
    LMS_target = np.dot(target, RGB_to_LMS_filter.T)  

    # 2. Convert data to logarithmic space for both source and target images:
    take_log(LMS_source)
    take_log(LMS_target)

    # 3. Apply the given transformation to convert to lαβ space

    LMS_to_lαβ_filter = np.dot(np.array([
    [1/math.sqrt(3), 0, 0],
    [0, 1/math.sqrt(6), 0],
    [0, 0, 1/math.sqrt(2)]
    ]),
        np.array([[1, 1,1],
    [1, 1, -2],
    [1, -1, 0]]))

    lαβ_source = np.dot(LMS_source, LMS_to_lαβ_filter.T)
    lαβ_target = np.dot(LMS_target, LMS_to_lαβ_filter.T)

    # 4. Compute the mean and variance of the images for l,α,β channels.

    lαβ_source_means = np.mean(lαβ_source[:, :, 0]), np.mean(lαβ_source[:, :, 1]), np.mean(lαβ_source[:, :, 2])
    lαβ_target_means = np.mean(lαβ_target[:, :, 0]), np.mean(lαβ_target[:, :, 1]), np.mean(lαβ_target[:, :, 2])

    lαβ_source_vars = np.var(lαβ_source[:, :, 0]), np.var(lαβ_source[:, :, 1]), np.var(lαβ_source[:, :, 2])
    lαβ_target_vars = np.var(lαβ_target[:, :, 0]), np.var(lαβ_target[:, :, 1]), np.var(lαβ_target[:, :, 2])

   # 5. Subtract the mean of source image from the source image

    l_source = lαβ_source[:, :, 0] - lαβ_source_means[0]
    α_source = lαβ_source[:, :, 1] - lαβ_source_means[1]
    β_source = lαβ_source[:, :, 2] - lαβ_source_means[2]

    # 6. Scale the data points by the respective variance

    l_source = (lαβ_target_vars[0] / lαβ_source_vars[0]) * l_source
    α_source = (lαβ_target_vars[1] / lαβ_source_vars[1]) * α_source
    β_source = (lαβ_target_vars[2] / lαβ_source_vars[2]) * β_source

    # 7. Add the target’s mean to the scaled data points

    l_source = l_source + lαβ_target_means[0]
    α_source = α_source + lαβ_target_means[1]
    β_source = β_source + lαβ_target_means[2]

    # 8. Apply transform matrix to convert lαβ to LMS
    lαβ_source = np.dstack((l_source, α_source, β_source))
    
    lαβ_to_LMS_filter = np.dot(
        np.array(
            [[1, 1, 1],
            [1, 1, -1],
            [1, -2, 0]]
            ),
        np.array(
            [[1/math.sqrt(3), 0,0],
            [0, 1/math.sqrt(6), 0],
            [0, 0, 1/math.sqrt(2)]]) 
        )

    lαβ_source = np.dot(lαβ_source, lαβ_to_LMS_filter.T)

    # 9. Go back to linear space

    lαβ_source = 10 ** lαβ_source
    lαβ_source[lαβ_source == 1] = 0

    # 10. Apply transform matrix to convert LMS to RGB:
    LMS_to_RGB_filter = np.array([
                                [4.4679, -3.5873, 0.1193],
                                [-1.2186, 2.3809, -0.1624],
                                [0.0497, -0.2439, 1.2045]])

    RGB_source = np.dot(lαβ_source, LMS_to_RGB_filter.T)
    
    return RGB_source
