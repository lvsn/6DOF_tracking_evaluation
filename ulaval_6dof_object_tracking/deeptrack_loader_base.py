import os
import json

from tqdm import tqdm

from ulaval_6dof_object_tracking.utils.frame import FrameNumpy, Frame, FrameHdf5
from ulaval_6dof_object_tracking.utils.camera import Camera
from ulaval_6dof_object_tracking.utils.transform import Transform
from pytorch_toolbox.loader_base import LoaderBase


class DeepTrackLoaderBase(LoaderBase):
    def __init__(self, root, pretransforms=[], posttransforms=[], target_transform=[], read_data=True):
        self.data_pose = []
        self.data_pair = {}
        self.metadata = {}
        self.pretransforms = pretransforms
        self.posttransforms = posttransforms
        self.read_data = read_data
        super(DeepTrackLoaderBase, self).__init__(root, [], target_transform)

    def load(self, path):
        """
        Load a viewpoints.json to dataset's structure
        Todo: datastructure should be more similar to json structure...
        :return: return false if the dataset is empty.
        """
        # Load viewpoints file and camera file
        with open(os.path.join(path, "viewpoints.json")) as data_file:
            data = json.load(data_file)
        self.camera = Camera.load_from_json(path)
        self.metadata = data["metaData"]
        self.set_save_type(self.metadata["save_type"])
        count = 0
        # todo this is not clean!i
        while True:
            try:
                id = str(count)
                pose = Transform.from_parameters(*[float(data[id]["vector"][str(x)]) for x in range(6)])
                self.add_pose(None, None, pose)
                if "pairs" in data[id]:
                    for i in range(int(data[id]["pairs"])):
                        pair_id = "{}n{}".format(id, i)
                        pair_pose = Transform.from_parameters(*[float(data[pair_id]["vector"][str(x)]) for x in range(6)])
                        self.add_pair(None, None, pair_pose, count)
                count += 1

            except KeyError:
                break
        return self.data_pose

    def make_dataset(self, dir):
        data = []
        if self.read_data:
            try:
                data = self.load(dir)
            except FileNotFoundError:
                print("[Warning] no dataset saved at path {}".format(dir))
                print("Resuming...")
        return data

    def from_index(self, index):
        raise RuntimeError("Not Implemented")

    def load_image(self, index):
        frame, pose = self.data_pose[index]
        rgb, depth = frame.get_rgb_depth(self.root)
        return rgb, depth, pose

    def load_pair(self, index, pair_id):
        frame, pose = self.data_pair[int(index)][pair_id]
        rgb, depth = frame.get_rgb_depth(self.root)
        return rgb, depth, pose

    def size(self):
        return len(self.data_pose)

    def set_save_type(self, frame_class):
        if frame_class == "numpy":
            self.frame_class = FrameNumpy
        elif frame_class == "hdf5":
            self.frame_class = FrameHdf5
        else:
            self.frame_class = Frame

    def add_pose(self, rgb, depth, pose):
        index = self.size()
        frame = self.frame_class(rgb, depth, str(index))
        self.data_pose.append([frame, pose])
        return index

    def pair_size(self, id):
        id_int = int(id)
        if id_int not in self.data_pair:
            return 0
        else:
            return len(self.data_pair[id_int])

    def add_pair(self, rgb, depth, pose, id):
        id_int = int(id)
        if id_int >= len(self.data_pose):
            raise IndexError("impossible to add pair if pose does not exists")
        if id_int in self.data_pair:
            frame = self.frame_class(rgb, depth, "{}n{}".format(id_int, len(self.data_pair[id_int]) - 1))
            self.data_pair[id_int].append((frame, pose))
        else:
            frame = self.frame_class(rgb, depth, "{}n0".format(id_int))
            self.data_pair[id_int] = [(frame, pose)]

    def dump_images_on_disk(self, verbose=False):
        """
        Unload all images data from ram and save them to the dataset's path ( can be reloaded with load_from_disk())
        :return:
        """
        print("[INFO]: Dump image on disk")
        for frame, pose in tqdm(self.data_pose):
            if verbose:
                print("Save frame {}".format(frame.id))
            if int(frame.id) in self.data_pair:
                for pair_frame, pair_pose in self.data_pair[int(frame.id)]:
                    pair_frame.dump(self.root)
            frame.dump(self.root)

    def save_json_files(self, metadata):
        viewpoints_data = {}
        for frame, pose in self.data_pose:
            self.insert_pose_in_dict(viewpoints_data, frame.id, pose)
            if int(frame.id) in self.data_pair:
                viewpoints_data[frame.id]["pairs"] = len(self.data_pair[int(frame.id)])
                for pair_frame, pair_pose in self.data_pair[int(frame.id)]:
                    self.insert_pose_in_dict(viewpoints_data, pair_frame.id, pair_pose)
            else:
                viewpoints_data[frame.id]["pairs"] = 0
        viewpoints_data["metaData"] = metadata
        with open(os.path.join(self.root, "viewpoints.json"), 'w') as outfile:
            json.dump(viewpoints_data, outfile)
        if self.camera is None:
            raise Exception("Camera is not defined for dataset...")
        self.camera.save(self.root)

    @staticmethod
    def insert_pose_in_dict(dict, key, item):
        params = {}
        for i, param in enumerate(item.to_parameters()):
            params[str(i)] = str(param)
        dict[key] = {"vector": params}
