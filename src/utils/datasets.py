import glob
import os
import cv2
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_erosion
from src.utils.semantic_utils import SemanticDecoder
from src.utils.transform_utils import SWAP_AND_FLIP_WORLD_AXES, FLIP_CAM_AXES


class Atlas(Dataset):
    def __init__(self, cfg, args, scale):
        super(Atlas, self).__init__()
        self.name = cfg['dataset']

        self.scale = scale

        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder

        # Load cameras and image and mask paths
        self.poses, self.color_paths, self.mask_paths, self.registration_matrices = self.load_cams_and_filepaths(
            os.path.join(self.input_folder, 'transforms.json'))

        # TODO: what about semantic decoder? ignore for now
        # self.semantic_decoder = SemanticDecoder()

        return


    def load_cams_and_filepaths(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)

        frames = data['frames']

        color_paths = [os.path.join(self.input_folder, frame['file_path']) for frame in frames]
        mask_paths = [os.path.join(self.input_folder, frame['mask_path']) for frame in frames]
        poses = np.array([frame['transform_matrix'] for frame in frames], dtype=float)
        poses = torch.from_numpy(poses).float()
        registration_matrices = np.array([frame['mesh_matrix'] for frame in frames], dtype=float)
        registration_matrices = torch.from_numpy(registration_matrices).float()

        # Load intrinsics and wrap them in a dict
        fx = np.array([frame['fl_x'] for frame in frames], dtype=float)
        fy = np.array([frame['fl_y'] for frame in frames], dtype=float)
        cx = np.array([frame['cx'] for frame in frames], dtype=float)
        cy = np.array([frame['cy'] for frame in frames], dtype=float)
        self.intrinsics = np.array([[fx[i], fy[i], cx[i], cy[i]] for i in range(len(frames))], dtype=float)
        self.intrinsics = torch.from_numpy(self.intrinsics).float()

        ## TODO: Make all poses relative to the first one; not sure what this is good for
        #self.poses = torch.linalg.inv(self.poses[0])[None, ...] @ self.poses

        # Scale the translations; apparently necessary to be in correct metric
        poses[:, :3, 3] *= self.scale

        # See Nerfstudio conventions here https://docs.nerf.studio/quickstart/data_conventions.html

        # Reorient camera in camera space
        poses = poses @ FLIP_CAM_AXES

        # Reorient world space
        poses = SWAP_AND_FLIP_WORLD_AXES @ poses

        return poses, color_paths, mask_paths, registration_matrices


    def __getitem__(self, index):
        color_data = cv2.cvtColor(cv2.imread(self.color_paths[index]), cv2.COLOR_BGR2RGB)
        color_data = torch.from_numpy(color_data)
        color_data = color_data / 255.

        # TODO: will later need tool masking as well
        mask_path = self.mask_paths[index]
        if os.path.isfile(mask_path):
            mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_data = cv2.resize(mask_data, (color_data.shape[1], color_data.shape[0]), interpolation=cv2.INTER_NEAREST)
            # mask_data = torch.from_numpy(binary_erosion(mask_data, iterations=7, border_value=1) > 0).bool()
            mask_data = torch.from_numpy(mask_data).bool()
        else:
            print(f"No mask found for path {mask_path}")

            mask_data = None

        # TODO: can add some semantic segmentation here later
        semantics = None

        pose = self.poses[index]
        registration_matrix = self.registration_matrices[index]

        intrinsics = self.intrinsics[index]

        # Returning none for color_data_r for now, as we don't have stereo images
        return index, color_data, None, pose, registration_matrix, mask_data, semantics, intrinsics


    def __len__(self):
        return len(self.poses)


class StereoMIS(Dataset):
    def __init__(self, cfg, args, scale):
        super(StereoMIS, self).__init__()
        self.name = cfg['dataset']
        self.scale = scale
        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, 'video_frames*', '*l.png')))
        self.load_poses(os.path.join(self.input_folder, 'groundtruth.txt'), slice(cfg['data']['start'],cfg['data']['stop'],cfg['data']['step']))
        self.color_paths = self.color_paths[slice(cfg['data']['start'],cfg['data']['stop'],cfg['data']['step'])]
        self.n_img = len(self.color_paths)
        self.semantic_decoder = SemanticDecoder()

    def load_poses(self, path: str, sclice):
        with open(path, 'r') as f:
            data = f.read()
            lines = data.replace(",", " ").replace("\t", " ").split("\n")
            list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                    len(line) > 0 and line[0] != "#"]

        trans = np.asarray([l[1:4] for l in list if len(l) > 0], dtype=float)
        quat = np.asarray([l[4:] for l in list if len(l) > 0], dtype=float)
        self.poses = np.eye(4)[None, ...].repeat(quat.shape[0], axis=0)
        self.poses[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
        self.poses[:, :3, 3] = trans
        self.poses = torch.from_numpy(self.poses[sclice]).float()
        self.poses = torch.linalg.inv(self.poses[0])[None, ...] @ self.poses
        self.poses[:, :3, 3] *= self.scale

    def __getitem__(self, index):
        color_data = cv2.cvtColor(cv2.imread(self.color_paths[index]), cv2.COLOR_BGR2RGB)
        color_data = torch.from_numpy(color_data)
        color_data = color_data / 255.

        color_data_r = cv2.cvtColor(cv2.imread(self.color_paths[index].replace('l.png', 'r.png')), cv2.COLOR_BGR2RGB)
        color_data_r = torch.from_numpy(color_data_r)
        color_data_r = color_data_r / 255.

        # add tool mask if existing
        mask_path = self.color_paths[index].replace("video_frames", "masks")
        if os.path.isfile(mask_path):
            mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_data = cv2.resize(mask_data, (color_data.shape[1], color_data.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_data = torch.from_numpy(binary_erosion(mask_data, iterations=7, border_value=1) > 0).bool()
        else:
            mask_data = None

        # add semantic segmentation if existing
        sm_path = self.color_paths[index].replace("video_frames", "semantic_predictions")
        if os.path.isfile(sm_path):
            semantics = cv2.cvtColor(cv2.imread(sm_path), cv2.COLOR_BGR2RGB)
            semantics = cv2.resize(semantics, (color_data.shape[1], color_data.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            semantics = self.semantic_decoder(semantics)
        else:
            semantics = None
        pose = self.poses[index]
        # No frame-specific intrinsics provided in this dataset
        intrinsics = None
        return index, color_data, color_data_r, pose, mask_data, semantics, intrinsics

    def get_name(self, index):
        return os.path.basename(self.color_paths[index])

    def __len__(self):
        return self.n_img


# extend default collate for None elements
def collate_none_fn(batch, *, collate_fn_map = None):
    return None
from torch.utils.data._utils.collate import default_collate_fn_map
default_collate_fn_map.update({type(None):collate_none_fn})
