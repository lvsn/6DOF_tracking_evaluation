"""
    Example script to load a sequence and a 3D model
"""
import cv2
import numpy as np

from ulaval_6dof_object_tracking.utils.model_renderer import ModelRenderer
from ulaval_6dof_object_tracking.utils.sequence_loader import SequenceLoader

GREEN_HUE = 120/2
RED_HUE = 230/2
BLUE_HUE = 0
LIGH_BLUE_HUE = 20
PURPLE_HUE = 300/2

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
    object_id = "clock"

    model_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models/{}/geometry.ply".format(object_id)
    sequence_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/eccv_backup/Sequences/processed/{}_fix_occluded_2".format(object_id)

    sequence = SequenceLoader(sequence_path)
    renderer = ModelRenderer(model_path, "utils/shader", sequence.camera, [(sequence.camera.width, sequence.camera.height)])

    for pose, rgb, depth in sequence:

        rgb_render, depth_render = renderer.render_image(pose)
        rgb_render = set_hue(rgb_render, PURPLE_HUE)

        rgb = cv2.pyrDown(rgb)
        rgb_render = cv2.pyrDown(rgb_render)

        blend = image_blend_gray(rgb_render[:, :, ::-1], rgb)

        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.imshow("blending", blend)
        cv2.waitKey(10)