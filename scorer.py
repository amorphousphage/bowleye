import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

class PinScorer:
    def __init__(self, pin_coordinates):
        self.pin_coordinates = pin_coordinates

    def PinScoreReading(self, ref_image, read_image, kernel, sigma, threshold, maximum, region_size):
        # Convert frame and reference_frame to grayscale
        gray = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        # Apply a blur to the grayscale frame to reduce noise
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), sigma)
        # Add Blur to the Reference ROI image to reduce noise
        blurred_reference = cv2.GaussianBlur(reference_gray, (kernel, kernel), sigma)

        # Compute SSIM and diff map
        score, diff = compare_ssim(gray, reference_gray, full=True)
        diff = (1 - diff) * 255  # Invert: higher value = more different
        diff = diff.astype("uint8")

        # Apply binary thresholding to segment the differences
        _, binary = cv2.threshold(diff, threshold, maximum, cv2.THRESH_BINARY)

        # Define a list of still standing pins
        fallen_pins = []

        # Half the region_size to make it a radius
        region_size = int(region_size // 2)

        for pin, coord in self.pin_coordinates.items():
            x, y = coord[0], coord[1]

            # Define a region around the pin
            read_region = binary[y-region_size:y+region_size, x-region_size:x+region_size]
            
            # Count the number of white pixels (not black pixels with value 0)
            white_pixel_count = np.count_nonzero(read_region)

            # Total number of pixels in the region
            total_pixels = read_region.size

            # If the majority of pixels are white, remove the pin from standing_pins
            if white_pixel_count / total_pixels > 0.5:  # More than 50% white pixels
                fallen_pins.append(pin)

            # Make sure the ref_image is contiguous array
            ref_image = np.ascontiguousarray(ref_image)

            # Show the region in the reference_frame and save the ref, read and binary image
            cv2.rectangle(ref_image, (x-region_size, y-region_size), (x+region_size, y+region_size), (0, 255, 0), 2)

        cv2.imwrite("test/03binary.png", binary)
        cv2.imwrite("test/02read.png", read_image)
        cv2.imwrite("test/01ref.png", ref_image)

        return fallen_pins
