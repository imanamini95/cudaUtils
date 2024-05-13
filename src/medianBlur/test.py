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
    input_image = np.random.randint(0, 255, size=(1024, 1024), dtype=np.uint8)

    # Using cv2.medianBlur
    start_time = time.time()
    cv2_result = cv2.medianBlur(input_image, window_size)
    cv2_time = time.time() - start_time

    # Using CUDA medianBlur
    start_time = time.time()
    cuda_result = medianBlur(input_image, window_size)
    cuda_time = time.time() - start_time

    print(f"Results are the same for window_size={window_size}!")
    print("Time taken by cv2.medianBlur:", cv2_time)
    print("Time taken by CUDA medianBlur:", cuda_time)

    # Comparing results
    assert np.array_equal(cv2_result, cuda_result), "Results are not the same!"


# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
