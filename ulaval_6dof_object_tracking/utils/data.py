from scipy import ndimage
import numpy as np
import random
from skimage.color import rgb2hsv, hsv2rgb


def add_hsv_noise(rgb, hue_offset, saturation_offset, value_offset, proba=0.5):
    mask = np.all(rgb != 0, axis=2)
    hsv = rgb2hsv(rgb/255)
    if random.uniform(0, 1) > proba:
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue_offset, hue_offset)) % 1
    if random.uniform(0, 1) > proba:
        hsv[:, :, 1] = (hsv[:, :, 1] + random.uniform(-saturation_offset, saturation_offset)) % 1
    if random.uniform(0, 1) > proba:
        hsv[:, :, 2] = (hsv[:, :, 2] + random.uniform(-value_offset, value_offset)) % 1
    rgb = hsv2rgb(hsv) * 255
    return rgb.astype(np.uint8) * mask[:, :, np.newaxis]


def depth_blend(rgb1, depth1, rgb2, depth2):
    new_depth2 = depth2.copy()
    new_depth1 = depth1.copy()

    rgb1_mask = np.all(rgb1 == 0, axis=2)
    rgb2_mask = np.all(rgb2 == 0, axis=2)

    rgb1_mask = ndimage.binary_dilation(rgb1_mask)

    new_depth2[rgb2_mask] = -100000
    new_depth1[rgb1_mask] = -100000

    mask = (new_depth1 < new_depth2)
    pos_mask = mask.astype(np.uint8)
    neg_mask = (mask == False).astype(np.uint8)

    masked_rgb_occluder = rgb1 * pos_mask[:, :, np.newaxis]
    masked_rgb_object = rgb2 * neg_mask[:, :, np.newaxis]

    masked_depth_occluder = depth1 * pos_mask
    masked_depth_object = depth2 * neg_mask

    blend_rgb = masked_rgb_occluder + masked_rgb_object
    blend_depth = masked_depth_occluder + masked_depth_object

    return blend_rgb, blend_depth, pos_mask


def gaussian_noise(img, gaussian_std):
    type = img.dtype
    copy = img.astype(np.float)
    gaussian_noise = np.random.normal(0, gaussian_std, img.shape)
    copy = (gaussian_noise + copy)
    if type == np.uint8:
        copy[copy < 0] = 0
        copy[copy > 255] = 255
    return copy.astype(type)


def color_blend(rgb1, depth1, rgb2, depth2):
    mask = np.all(rgb1 == 0, axis=2)
    mask = ndimage.binary_dilation(mask).astype(mask.dtype)
    depth1[mask] = 0
    rgb1[mask, :] = 0
    mask = mask.astype(np.uint8)
    new_depth = depth2 * mask + depth1
    new_color = rgb2 * mask[:, :, np.newaxis] + rgb1
    return new_color.astype(np.uint8), new_depth


def show_frames(rgbA, depthA, rgbB, depthB):
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(2, 3)
    ax1, ax2, ax5 = axis[0, :]
    ax3, ax4, ax6 = axis[1, :]
    ax1.imshow(rgbA.astype(np.uint8))
    ax2.imshow(rgbB.astype(np.uint8))
    ax3.imshow(depthA)
    ax4.imshow(depthB)
    ax5.imshow((rgbA - rgbB).sum(axis=2))
    ax6.imshow(depthA - depthB)
    plt.show()


def compute_2Dboundingbox(pose, camera, scale_size=230, scale=(1, 1, 1)):
    obj_x = pose.matrix[0, 3] * scale[0]
    obj_y = pose.matrix[1, 3] * scale[1]
    obj_z = pose.matrix[2, 3] * scale[2]
    offset = scale_size / 2
    points = np.ndarray((4, 3), dtype=np.float)
    points[0] = [obj_x - offset, obj_y - offset, obj_z]     # top left
    points[1] = [obj_x - offset, obj_y + offset, obj_z]     # top right
    points[2] = [obj_x + offset, obj_y - offset, obj_z]     # bottom left
    points[3] = [obj_x + offset, obj_y + offset, obj_z]     # bottom right
    return camera.project_points(points).astype(np.int32)


def project_center(pose, camera, scale=(1, 1, 1)):
    obj_x = pose.matrix[0, 3] * scale[0]
    obj_y = pose.matrix[1, 3] * scale[1]
    obj_z = pose.matrix[2, 3] * scale[2]
    points = np.ndarray((1, 3), dtype=np.float)
    points[0] = [obj_x, obj_y, obj_z]
    return camera.project_points(points).astype(np.int32)


def normalize_scale(color, depth, boundingbox, output_size=(100, 100)):
    import cv2
    left = np.min(boundingbox[:, 1])
    right = np.max(boundingbox[:, 1])
    top = np.min(boundingbox[:, 0])
    bottom = np.max(boundingbox[:, 0])

    # Compute offset if bounding box goes out of the frame (0 padding)
    h, w, c = color.shape
    crop_w = right - left
    crop_h = bottom - top
    color_crop = np.zeros((crop_h, crop_w, 3), dtype=color.dtype)
    depth_crop = np.zeros((crop_h, crop_w), dtype=np.float)
    top_offset = abs(min(top, 0))
    bottom_offset = min(crop_h - (bottom - h), crop_h)
    right_offset = min(crop_w - (right - w), crop_w)
    left_offset = abs(min(left, 0))

    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, h)
    right = min(right, w)
    color_crop[top_offset:bottom_offset, left_offset:right_offset, :] = color[top:bottom, left:right, :]
    depth_crop[top_offset:bottom_offset, left_offset:right_offset] = depth[top:bottom, left:right]

    resized_rgb = cv2.resize(color_crop, output_size, interpolation=cv2.INTER_NEAREST)
    resized_depth = cv2.resize(depth_crop, output_size, interpolation=cv2.INTER_NEAREST)

    mask_rgb = resized_rgb != 0
    mask_depth = resized_depth != 0
    resized_depth = resized_depth.astype(np.uint16)
    final_rgb = resized_rgb * mask_rgb
    final_depth = resized_depth * mask_depth
    return final_rgb, final_depth


def combine_view_transform(vp, view_transform):
    """
    combines a camera space transform with a camera axis dependent transform.
    Whats important here is that view transform's translation represent the displacement from
    each axis, and rotation from each axis. The rotation is applied around the translation point of view_transform.
    :param vp:
    :param view_transform:
    :return:
    """
    camera_pose = vp.copy()
    R = camera_pose.rotation
    T = camera_pose.translation
    rand_R = view_transform.rotation
    rand_T = view_transform.translation

    rand_R.combine(R)
    T.combine(rand_R)
    rand_T.combine(T)
    return rand_T


def image_blend(foreground, background):
    """
    Uses pixel 0 to compute blending mask
    :param foreground:
    :param background:
    :return:
    """
    if len(foreground.shape) == 2:
        mask = foreground[:, :] == 0
    else:
        mask = foreground[:, :, :] == 0
        mask = np.all(mask, axis=2)[:, :, np.newaxis]
    return background * mask + foreground


def compute_axis(pose, camera):
    points = np.ndarray((4, 3), dtype=np.float)
    points[0] = [0, 0, 0]
    points[1] = [1, 0, 0]
    points[2] = [0, 1, 0]
    points[3] = [0, 0, 1]
    points *= 0.1
    camera_points = pose.dot(points)
    camera_points[:, 0] *= -1
    return camera.project_points(camera_points).astype(np.int32)


def center_pixel(pose, camera):
    obj_x = pose.matrix[0, 3] * 1000
    obj_y = pose.matrix[1, 3] * 1000
    obj_z = pose.matrix[2, 3] * 1000
    point = [obj_x, -obj_y, -obj_z]
    return camera.project_points(np.array([point])).astype(np.uint32)
