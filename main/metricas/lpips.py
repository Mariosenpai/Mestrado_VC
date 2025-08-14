import cv2
import numpy as np
import torch
from torch import rand
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def carregar_e_normalizar(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 127.5 - 1.0  # normaliza para [-1, 1]
    img = np.clip(img, -1.0, 1.0)  # garante que todos os valores estão dentro do intervalo
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # [N, C, H, W]
    return img_tensor

def _lpips(img1, img2):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    img1_normalizada = carregar_e_normalizar(img1)
    img2_normalizada = carregar_e_normalizar(img2)

    return lpips(img1_normalizada, img2_normalizada)



if __name__ == '__main__':
    original = cv2.imread(r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\image (7).png").astype(np.float32)
    compressed = cv2.imread(r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\mixed.png").astype(np.float32)
    compressed = cv2.resize(compressed, original.shape[0:2])

    print(_lpips(original, compressed))