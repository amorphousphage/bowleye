import cv2
import numpy as np

class PinScorer:
    def __init__(self, reference_image, pin_coordinates):
        self.reference_image = reference_image
        self.pin_coordinates = pin_coordinates

    def PinsStillStanding(self, read_image, threshold=100):
        standing_pins = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        for pin, coord in self.pin_coordinates.items():
            x, y = coord[0], coord[1]

            # Define a region around the pin
            region_size = 20

            cv2.imwrite("videos/01read.png", read_image)
            
            ref_region = self.reference_image[y-region_size:y+region_size, x-region_size:x+region_size]
            read_region = read_image[y-region_size:y+region_size, x-region_size:x+region_size]
            
            # Calculate the mean pixel values of the region
            ref_pixel_mean = np.mean(ref_region, axis=(0, 1))
            read_pixel_mean = np.mean(read_region, axis=(0, 1))
            
            ref_pixel_mean = ref_pixel_mean.astype(int)
            read_pixel_mean = read_pixel_mean.astype(int)

            # Calculate color difference
            diff = np.linalg.norm(ref_pixel_mean - read_pixel_mean)

            if diff > threshold:
                standing_pins.remove(pin)

        return standing_pins
