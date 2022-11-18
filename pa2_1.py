from pa2_2 import colorTransfer, colorTransfer
import cv2

if __name__ == '__main__':

    # Read source image
    source_path = '/home/tugcekizilepe/Desktop/BBM415/Assignment1/example images/colortransfer/girl-and-fantasy-sky-scenery-1-free-photo.jpg'
    source = cv2.imread(source_path)
    
    # Read target image
    target_path = '/home/tugcekizilepe/Desktop/BBM415/Assignment1/example images/colortransfer/red.jpg'
    target = cv2.imread(target_path)

    # Call colorTransfer function
    new_image = colorTransfer(source, target)

    # Save color transfered image
    cv2.imwrite("color-transfer-result.jpg", new_image)



