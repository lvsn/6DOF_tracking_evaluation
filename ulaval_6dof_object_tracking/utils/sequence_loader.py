import os
import json
import numpy as np
from PIL import Image

from ulaval_6dof_object_tracking.utils.camera_configs import Camera
from ulaval_6dof_object_tracking.utils.transform import Transform


class SequenceLoader:
    def __init__(self, root):
        self.root = root
        self.data_pose = None
        self.current = 0

        with open(os.path.join(self.root, "meta_data.json")) as data_file:
            data = json.load(data_file)
        self.camera = Camera.load_from_json(self.root)
        self.metadata = data["metaData"]
        self.poses = np.load(os.path.join(self.root, "poses.npy"))

    def get_frame(self, i):
        pose = Transform.from_matrix(self.poses[i].reshape(4, 4))
        rgb = np.array(Image.open(os.path.join(self.root, "{}.png".format(i))))
        depth = np.array(Image.open(os.path.join(self.root, "{}d.png".format(i)))).astype(np.uint16)
        return pose, rgb, depth

    def size(self):
        return len(self.poses)

    def __iter__(self):
        self.current = 0
        return self

    def __len__(self):
        return self.size()

    def __getitem__(self, i):
        return self.get_frame(i)

    def __next__(self):
        if self.current > self.size():
            raise StopIteration
        else:
            self.current += 1
            return self.get_frame(self.current - 1)