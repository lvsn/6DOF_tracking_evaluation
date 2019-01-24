import scipy.signal
import scipy.stats
from scipy.misc import imresize
import torch
from skimage.measure import block_reduce

from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
import random
import numpy as np

from deep_6dof_tracking.data.rgbd_dataset import RGBDDataset
from deep_6dof_tracking.data.utils import add_hsv_noise, depth_blend, gaussian_noise, color_blend, show_frames
from deep_6dof_tracking.utils.transform import Transform


class Occluder(object):
    def __init__(self, path, proba=0.75):
        self.loader = DeepTrackLoader(path)
        self.proba = proba
    """
    Remove wrong bounding box ( width or height of 0)
    input : bounding box [x, y, w, h, ...]
    """
    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        prior_T = Transform.from_parameters(*prior)
        if random.uniform(0, 1) < self.proba:
            rand_id = random.randint(0, self.loader.size() - 1)
            occluder_rgb, occluder_depth, occ_pose = self.loader.load_image(rand_id)
            if random.randint(0, 1):
                occluder_rgb, occluder_depth, _ = self.loader.load_pair(rand_id, 0)
            occluder_depth = occluder_depth.astype(np.float32)
            # Z offset of occluder to be closer to the occluded object
            offset = occ_pose.matrix[2, 3] - prior_T.matrix[2, 3]
            occluder_depth[occluder_depth != 0] += offset*1000
            occluder_depth[occluder_depth != 0] -= random.randint(0, 500)
            occluder_rgb = add_hsv_noise(occluder_rgb, 1, 0.1, 0.1)
            occluder_rgb = imresize(occluder_rgb, depthB.shape, interp='nearest')
            occluder_depth = imresize(occluder_depth, depthB.shape, interp='nearest', mode="F").astype(np.int16)
            rgbB, depthB, occluder_mask = depth_blend(rgbB, depthB, occluder_rgb, occluder_depth)
        else:
            occluder_mask = None
        return rgbA, depthA, rgbB, depthB, prior, occluder_mask


class HSVNoise(object):
    def __init__(self, h_noise, s_noise, v_noise, proba=0.5):
        self.proba = proba
        self.h_noise = h_noise
        self.s_noise = s_noise
        self.v_noise = v_noise

    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        rgbB = add_hsv_noise(rgbB, self.h_noise, self.s_noise, self.v_noise, proba=self.proba)
        return rgbA, depthA, rgbB, depthB, prior


class GaussianNoise(object):
    def __init__(self, rgb_noise, depth_noise, proba=0.95):
        self.rgb_noise = rgb_noise
        self.depth_noise = depth_noise
        self.proba = proba

    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        if random.uniform(0, 1) < self.proba:
            noise = random.uniform(0, self.rgb_noise)
            rgbB = gaussian_noise(rgbB, noise)
        if random.uniform(0, 1) < self.proba:
            noise = random.uniform(0, self.depth_noise)
            depthB = gaussian_noise(depthB, noise)

        return rgbA, depthA, rgbB, depthB, prior


class GaussianBlur(object):
    def __init__(self, max_kernel_size, min_kernel_size=3, proba=0.4):
        self.proba = proba
        self.max_kernel_size = max_kernel_size
        self.min_kernel_size = min_kernel_size

    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        if random.uniform(0, 1) < self.proba:
            kernel_size = random.randint(self.min_kernel_size, self.max_kernel_size)
            kernel = self.gkern(kernel_size)
            rgbB[:, :, 0] = scipy.signal.convolve2d(rgbB[:, :, 0], kernel, mode='same')
            rgbB[:, :, 1] = scipy.signal.convolve2d(rgbB[:, :, 1], kernel, mode='same')
            rgbB[:, :, 2] = scipy.signal.convolve2d(rgbB[:, :, 2], kernel, mode='same')
        if random.uniform(0, 1) < self.proba:
            kernel_size = random.randint(self.min_kernel_size, self.max_kernel_size)
            kernel = self.gkern(kernel_size)
            depthB[:, :] = scipy.signal.convolve2d(depthB[:, :], kernel, mode='same')
        return rgbA, depthA, rgbB, depthB, prior

    @staticmethod
    def gkern(kernlen=21, nsig=2):
        """Returns a 2D Gaussian kernel array."""
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(scipy.stats.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel


class Background(object):
    def __init__(self, path):
        self.background = RGBDDataset(path)

    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        color_background, depth_background = self.background.load_random_image(rgbB.shape[1])
        depth_background = depth_background.astype(np.int32)
        rgbB, depthB = color_blend(rgbB, depthB, color_background, depth_background)
        return rgbA, depthA, rgbB, depthB, prior


class OffsetDepth(object):
    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        prior_T = Transform.from_parameters(*prior)
        depthA = self.normalize_depth(depthA, prior_T)
        depthB = self.normalize_depth(depthB, prior_T)
        #show_frames(rgbA, depthA, rgbB, depthB)
        return rgbA.astype(np.float32), depthA, rgbB.astype(np.float32), depthB, prior

    @staticmethod
    def normalize_depth(depth, pose):
        depth = depth.astype(np.float32)
        zero_mask = depth == 0
        depth += pose.matrix[2, 3] * 1000
        depth[zero_mask] = 5000
        return depth


class KinectOffset(object):
    """
    Simulate a bad Kinect measurment
    """
    def __init__(self, std=7):
        self.std = std

    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        noise = int(random.normalvariate(0, self.std))
        depthB[depthB != 0] -= noise
        return rgbA, depthA, rgbB, depthB, prior


class ChannelHide(object):
    def __init__(self, disable_proba=0.25):
        self.disable_proba = disable_proba

    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        if random.uniform(0, 1) < self.disable_proba:
            if random.randint(0, 1):
                rgbB[:, :, :] = 0
                #noise = random.uniform(0, 30)
                #rgbB = gaussian_noise(rgbB, noise)
            else:
                depthB[:, :] = 0
                #noise = random.uniform(0, 30)
                #depthB = gaussian_noise(depthB, noise)
        return rgbA, depthA, rgbB, depthB, prior


class NormalizeChannels(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        rgbA, depthA = self.normalize_channels(rgbA, depthA, self.mean[:4], self.std[:4])
        rgbB, depthB = self.normalize_channels(rgbB, depthB, self.mean[4:], self.std[4:])
        return rgbA, depthA, rgbB, depthB, prior

    @staticmethod
    def normalize_channels(rgb, depth, mean, std):
        """
        Normalize image by negating mean and dividing by std (precomputed)
        :param self:
        :param rgb:
        :param depth:
        :param type:
        :return:
        """
        rgb = rgb.T
        depth = depth.T
        rgb -= mean[:3, np.newaxis, np.newaxis]
        rgb /= std[:3, np.newaxis, np.newaxis]
        depth -= mean[3, np.newaxis, np.newaxis]
        depth /= std[3, np.newaxis, np.newaxis]
        return rgb, depth


class Transpose(object):
    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        rgbA = rgbA.T
        rgbB = rgbB.T
        depthA = depthA.T
        depthB = depthB.T
        return rgbA, depthA, rgbB, depthB, prior


class ToTensor(object):
    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        bufferA = np.zeros((4, rgbA.shape[1], rgbA.shape[2]), dtype=np.float32)
        bufferB = np.zeros((4, rgbA.shape[1], rgbA.shape[2]), dtype=np.float32)
        bufferA[0:3, :, :] = rgbA
        bufferA[3, :, :] = depthA
        bufferB[0:3, :, :] = rgbB
        bufferB[3, :, :] = depthB
        bufferA = torch.from_numpy(bufferA)
        bufferB = torch.from_numpy(bufferB)
        return [bufferA, bufferB]


class DepthDownsample(object):
    def __init__(self, proba=0.7):
        self.proba = proba

    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        w = depthB.shape[0]
        h = depthB.shape[1]
        # small hack.. downsample should be at image generation time...
        if w > 150:
            if random.uniform(0, 1) < self.proba:
                new_img = block_reduce(depthB[:, :], block_size=(2, 2), func=np.mean)[1:-1, 1:-1]
                new_img = scipy.misc.imresize(new_img, (w, h), interp="nearest", mode="F")
                depthB[:, :] = new_img
        return rgbA, depthA, rgbB, depthB, prior
