"""
    Example script to load a sequence and a 3D model
"""
import argparse

import cv2
import numpy as np
import os

from ulaval_6dof_object_tracking.utils.model_renderer import ModelRenderer
from ulaval_6dof_object_tracking.utils.sequence_loader import SequenceLoader

GREEN_HUE = 120 / 2
RED_HUE = 230 / 2
BLUE_HUE = 0
LIGH_BLUE_HUE = 20
PURPLE_HUE = 300 / 2

BRIGHTNESS = 100


def set_hue(rgb_input, hue_value):
    hsv = cv2.cvtColor(rgb_input, cv2.COLOR_RGB2HSV).astype(np.int)
    hsv[:, :, 0] = hue_value
    hsv[:, :, 1] += 150
    hsv[:, :, 2] += 100
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    rgb_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb_output[rgb_input == 0] = 0
    return rgb_output


def image_blend_gray(foreground, background):
    """
    Uses pixel 0 to compute blending mask
    will set the background gray
    :param foreground:
    :param background:
    :return:
    """
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    background_gray[:, :] = (np.clip(background_gray.astype(np.int) - BRIGHTNESS, 0, 255)).astype(np.uint8)
    if len(foreground.shape) == 2:
        mask = foreground[:, :] == 0
    else:
        mask = foreground[:, :, :] == 0
        mask = np.all(mask, axis=2)[:, :, np.newaxis]
    return background_gray[:, :, np.newaxis] * mask + foreground


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show sequence')
    parser.add_argument('-r', '--root', help="root path", action="store", default="../sample")
    parser.add_argument('-s', '--sequence',
                        help="Available sequences : \n"
                             "stability_near_[1-4], stability_far_[1-4], stability_occluded_[1-4]\n"
                             "occlusion_0, occlusion_[h/v]_[15/30/45/60/75]\n"
                             "interaction_translation, interaction_rotation, interaction_full, interaction_hard",
                        action="store", required=True)
    parser.add_argument('-o', '--object',
                        help="Available objects :"
                             " clock, cookiejar, dog, dragon, kinect, lego, shoe, skull, turtle, walkman, wateringcan",
                        action="store", required=True)

    arguments = parser.parse_args()
    root_path = arguments.root
    sequence_name = arguments.sequence
    object_name = arguments.object

    model_path = os.path.join(root_path, "{}/geometry.ply".format(object_name))
    sequence_path = os.path.join(root_path, "{}_{}".format(object_name, sequence_name))

    sequence = SequenceLoader(sequence_path)
    renderer = ModelRenderer(model_path, "utils/shader", sequence.camera,
                             [(sequence.camera.width, sequence.camera.height)])

    for pose, rgb, depth in sequence:
        rgb_render, depth_render = renderer.render_image(pose)
        rgb_render = set_hue(rgb_render, PURPLE_HUE)

        rgb = cv2.pyrDown(rgb)
        rgb_render = cv2.pyrDown(rgb_render)

        blend = image_blend_gray(rgb_render[:, :, ::-1], rgb)

        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.imshow("blending", blend)
        cv2.waitKey(10)
