import os
import sys

import numpy as np
import cv2
import time
import pytest

sys.path.insert(0, os.getcwd())
from src.medianBlur.medianBlur import medianBlur


@pytest.mark.parametrize("window_size", [3, 5])
def test_medianBlur(window_size):
    # Generate a random input image
    input_image = np.random.randint(0, 255, size=(4196, 4196), dtype=np.uint8)

    # Using cv2.medianBlur
    cv2_result = cv2.medianBlur(input_image, window_size)

    # Using CUDA medianBlur
    cuda_result = medianBlur(input_image, window_size)

    # Comparing results
    assert np.array_equal(cv2_result, cuda_result), "Results are not the same!"


# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
