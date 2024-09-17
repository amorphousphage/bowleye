import cv2
import numpy as np

class PinScorer:
    def __init__(self, reference_image, pin_coordinates):
        self.reference_image = reference_image
        self.pin_coordinates = pin_coordinates

    def PinsStillStanding(self, current_image, threshold=100):
        standing_pins = ['1','2','3','4','5','6','7','8','9','10']
        for pin, coord in self.pin_coordinates.items():
            ref_pixel = self.reference_image[coord[1], coord[0]]
            current_pixel = current_image[coord[1], coord[0]]
            ref_pixel = ref_pixel.astype(int)
            current_pixel = current_pixel.astype(int)

            # Calculate the color difference
            diff = np.linalg.norm(ref_pixel - current_pixel)

            if diff > threshold:
                standing_pins.remove(pin)

        return standing_pins
