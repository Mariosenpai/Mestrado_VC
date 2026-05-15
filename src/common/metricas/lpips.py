
import numpy as np
import torch
from torch import rand
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity



def _lpips(img1, img2):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')



    return None



