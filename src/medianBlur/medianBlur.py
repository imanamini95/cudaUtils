import numpy as np
import ctypes

# Load CUDA module
_medianBlur = ctypes.CDLL(
    r"C:\Projects\ProfessionalDev\cudaUtils\src\medianBlur\medianBlur.dll"
)

# Define function arguments types
_medianBlur.cuda_medianBlur.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32),  # input
    np.ctypeslib.ndpointer(dtype=np.int32),  # output
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int,  # window_size
]

# Define function return type
_medianBlur.cuda_medianBlur.restype = None


def medianBlur(input_image, window_size):
    input_image = np.array(input_image, dtype=np.int32)
    output_image = np.empty_like(input_image)

    _medianBlur.cuda_medianBlur(
        input_image,
        output_image,
        input_image.shape[1],
        input_image.shape[0],
        window_size,
    )
    return output_image


# Example usage
if __name__ == "__main__":
    import time
    import cv2

    input_image = np.random.randint(0, 255, size=(9999, 9999), dtype=np.uint8)
    window_size = 3

    start_time = time.time()
    cuda_result = medianBlur(input_image, window_size)
    cuda_time = time.time() - start_time

    start_time = time.time()
    cv2_result = cv2.medianBlur(input_image, window_size)
    cv2_time = time.time() - start_time

    print("Time taken by cv2.medianBlur:", cv2_time)
    print("Time taken by CUDA medianBlur:", cuda_time)
