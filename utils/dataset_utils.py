import paddle
import numpy as np

class Augment_RGB_paddle:
    def __init__(self):
        pass
    def transform0(self, img):
        return img   
    def transform1(self, img):
        img = np.rot90(img, k=1, axes=[-1,-2])
        return img
    def transform2(self, img):
        img = np.rot90(img, k=2, axes=[-1,-2])
        return img
    def transform3(self, img):
        img = np.rot90(img, k=3, axes=[-1,-2])
        return img
    def transform4(self, img):
        img = np.flip(img,-2)
        return img
    def transform5(self, img):
        img = np.rot90(img, k=1, axes=[-1,-2])
        img = np.flip(img,-2)
        return img
    def transform6(self, img):
        img = np.rot90(img, k=2, axes=[-1,-2])
        img = np.flip(img,-2)
        return img
    def transform7(self, img):
        img = np.rot90(img, k=3, axes=[-1,-2])
        img = np.flip(img,-2)
        return img


def MixUp(rgb_gt, rgb_noisy, beta=1.2):
    bs = rgb_gt.shape[0]
    indices = paddle.randperm(bs)
    rgb_gt2 = paddle.index_select(rgb_gt, indices)
    rgb_noisy2 = paddle.index_select(rgb_noisy, indices)

    lam = paddle.Tensor(np.random.beta(beta, beta, (bs,1)).reshape([-1,1,1,1]).astype(np.float32))

    rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
    rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

    return rgb_gt, rgb_noisy
