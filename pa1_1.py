from pa1_2 import FloydSteinberd
import cv2

if __name__ == '__main__':
    img_path = '/home/tugcekizilepe/Desktop/BBM415/Assignment1/example images/dithering/1.png'
    # Read the image, then convert it to numpy array
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    q = 255 # quantization parameter
    new_image = FloydSteinberd(img, q)

    cv2.imwrite('dithering-result.jpg',  new_image)
    cv2.waitKey()


