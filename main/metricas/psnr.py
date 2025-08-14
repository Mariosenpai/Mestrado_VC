
from math import log10, sqrt
import cv2
import numpy as np

def _PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def main():
     original = cv2.imread(r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\image (7).png")
     compressed = cv2.imread(r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\mixed.png")
     compressed = cv2.resize(compressed, original.shape[0:2])

     value = _PSNR(original, compressed)
     print(f"PSNR value is {value/100} dB")

if __name__ == "__main__":
    main()