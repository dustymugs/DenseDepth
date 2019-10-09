import click
import csv
import glob
import itertools
import math
import numpy as np
import os
import os.path as osp
from PIL import Image, ImageOps

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.2

from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config))

from keras.models import load_model

from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images

class DenseDepthService(object):

    # specs from XBox Kinect v1
    # reference: http://wiki.ros.org/kinect_calibration/technical

    input_size = (640, 480)

    # in degrees
    hFov = 57
    vFov = 43

    scaling_factor = 10. # for depth data from DenseDepth trained on NYUv2

    rhFov = math.radians(hFov)
    rvFov = math.radians(vFov)

    # TODO add support for KITTI

    def __init__(
        self,
        model_path='nyu.h5',
        keypoints_csv=None,
        no_tif=False,
        no_ply=False,
    ):

        self.model_path = model_path
        self.keypoints_csv = keypoints_csv
        self.no_tif = no_tif
        self.no_ply = no_ply

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, new_value):

        assert osp.isfile(new_value)
        self._model_path = new_value

        self._model = None
        self.model # to proactively load model

    @property
    def keypoints_csv(self):
        return self._keypoints_csv

    @keypoints_csv.setter
    def keypoints_csv(self, new_value):

        if not new_value:
            self.keypoints_map = {}
            return

        assert osp.isfile(new_value)
        self._keypoints_csv = new_value

        self._build_keypoints_map()

    @property
    def keypoints_map(self):
        return self._keypoints_map

    @keypoints_map.setter
    def keypoints_map(self, new_value):
        self._keypoints_map = new_value

    def _build_keypoints_map(self):

        keypoints_map = {}
        with open(self._keypoints_csv) as fh:
            reader = csv.DictReader(fh)

            for row in reader:
                by_image = self.keypoints_map.setdefault(row['image'], [])
                by_image.append((
                    row['keypoint'],
                    int(row['x']),
                    int(row['y'])
                ))

        self.keypoints_map = keypoints_map

    @property
    def model(self):

        if getattr(self, '_model', None) is None:

            # Custom object needed for inference and training
            custom_objects = {
                'BilinearUpSampling2D': BilinearUpSampling2D,
                'depth_loss_function': None
            }

            # Load model into GPU / CPU
            self._model = load_model(
                self.model_path,
                custom_objects=custom_objects,
                compile=False
            )

        return self._model

    def _predict(self, images_data):

        return predict(self.model, images_data)

    def predict(self, image_paths, images_data=None):

        # Input images
        if images_data is None:
            images_data = load_images(image_paths, self.input_size) 

        print('Loaded ({}) images'.format(images_data.shape[0]))

        # Compute results
        outputs = self._predict(images_data)

        return self._process_outputs(outputs, image_paths)

    @staticmethod
    def resize_image(im, size, return_info=False):

        '''
        Resize Image object to width and height while maintaining aspect ratio
        '''

        width, height = size
        im_width, im_height = im.size

        if im_width > im_height:
            scale_factor = width / im_width
        else:
            scale_factor = height / im_height

        im = im.resize(
            (int(im_width * scale_factor), int(im_height * scale_factor)),
            resample=Image.LANCZOS
        )

        delta_w = width - im.size[0]
        delta_h = height - im.size[1]
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - delta_w // 2,
            delta_h - delta_h // 2,
        )
        if delta_w != 0 or delta_h != 0:
            im = ImageOps.expand(im, padding)

        if not return_info:
            return im
        else:
            return im, scale_factor, padding[:2]

    def _get_path_base(self, path):

        if not hasattr(self, '_path_base'):
            self._path_base = {}

        if path in self._path_base:
            return self._path_base[path]

        base, ext = osp.splitext(path)
        self._path_base[path] = base

        return base

    def _make_depth_tif(self, im, image_path):

        if self.no_tif:
            return None

        print('{} => {}'.format(image_path, depth_path))
        depth_path = '{}.depth.tif'.format(self._get_path_base(image_path))
        im.save(depth_path, 'TIFF')

        return depth_path

    def _measure_keypoints(self, pos, image_path):

        image_keypoints = self.keypoints_map.get(image_path, [])
        if not image_keypoints:
            return None

        dist_path = '{}.kpdist.csv'.format(self._get_path_base(image_path))

        print('{} => {}'.format(image_path, dist_path))
        with open(dist_path, 'w') as fh:

            field_names = [
                'from_keypoint',
                'to_keypoint',
                'from_to_distance',
                'from_row',
                'from_x',
                'from_y',
                'from_z',
                'to_row',
                'to_x',
                'to_y',
                'to_z',
            ]
            writer = csv.DictWriter(fh, fieldnames=field_names)
            writer.writeheader()

            for a_idx, b_idx in itertools.combinations(
                range(len(image_keypoints)),
                2
            ):

                a_kp, a_x, a_y = image_keypoints[a_idx]
                b_kp, b_x, b_y = image_keypoints[b_idx]

                a_pixel = int(a_y * width + a_x)
                assert a_pixel < length
                a_voxel = pos[a_pixel]

                b_pixel = int(b_y * width + b_x)
                assert b_pixel < length
                b_voxel = pos[b_pixel]

                ab_dist = math.sqrt(
                    np.sum(
                        np.power(b_voxel - a_voxel, 2)
                    )
                )

                writer.writerow({
                    'from_row': a_idx,
                    'from_keypoint': a_kp,
                    'from_x': a_voxel[0],
                    'from_y': a_voxel[1],
                    'from_z': a_voxel[2],
                    'to_row': b_idx,
                    'to_keypoint': b_kp,
                    'to_x': b_voxel[0],
                    'to_y': b_voxel[1],
                    'to_z': b_voxel[2],
                    'from_to_distance': ab_dist
                })

        return dist_path

    def _make_ply(self, pos, color, image_path):

        if self.no_ply:
            return None

        ply_path = '{}.ply'.format(self._get_path_base(image_path))
        print('{} => {}'.format(image_path, ply_path))
        with open(ply_path,"w") as fh:
            fh.write('''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
'''.format(
                len(pos)
            ))

            for idx, xyz in enumerate(pos):

                x, y, z = xyz
                r, g, b = color[idx]
                fh.write(f'{x} {y} {z} {r} {g} {b} 0\n')

        return ply_path

    def _make_point_cloud(self, depth_data, image_path, src):

        width, height = src.size
        length = width * height

        # convert to point cloud
        fx = width / (2 * math.tan(self.rhFov / 2))
        fy = height / (2 * math.tan(self.rvFov / 2))
        cx = width / 2
        cy = height / 2

        xx = np.tile(range(width), height)
        yy = np.repeat(range(height), width)
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy

        z = depth_data.reshape(length)
        pos = np.dstack((
            xx * z,
            yy * z,
            z
        )).reshape((length, 3)) * self.scaling_factor
        color = np.array(src).reshape((length, 3))

        dist_path = self._measure_keypoints(pos, image_path)
        ply_path = self._make_ply(pos, color, image_path)

        return dist_path, ply_path

    def _process_output(self, idx, raw_depth_data, image_path):

        src = Image.open(image_path)

        raw_depth = Image.fromarray(
            np.moveaxis(raw_depth_data, -1, 0)[0],
            mode='F'
        )
        im = DenseDepthService.resize_image(raw_depth, src.size)
        depth_data = np.array(im)

        tif_path = self._make_depth_tif(im, image_path)
        dist_path, ply_path = self._make_point_cloud(
            depth_data,
            image_path,
            src
        )

        return tif_path, dist_path, ply_path

    def _process_outputs(self, outputs, image_paths):

        # output results as depth tif and point cloud ply
        paths = []
        for idx, raw_depth_data in enumerate(outputs):

            image_path = osp.abspath(image_paths[idx])
            tif_path, dist_path, ply_path = self._process_output(
                idx,
                raw_depth_data,
                image_path
            )

            paths.append(
                (image_path, tif_path, dist_path, ply_path)
            )

        return paths

@click.command()
@click.option('--model', default='nyu.h5', type=click.Path(), help='Trained Keras model file.')
@click.option('--keypoints', type=click.Path(), help='Keypoints in CSV format')
@click.option('--no-tif', is_flag=True, default=False, help='If provided, depth image will not be created')
@click.option('--no-ply', is_flag=True, default=False, help='If provided, point cloud will not be created')
@click.argument('image', nargs=-1, required=True)
def do_it(model, keypoints, no_tif, no_ply, image):
    '''
    High Quality Monocular Depth Estimation via Transfer Learning

    Predicts a depth image and outputs that prediction as a TIFF image file
    and a pointcloud as a PLY file

    IMAGE is an image file or a glob pattern

    --keypoints expects a CSV file with the following columns

        image - path to image file. should be same as how IMAGE is provided
        keypoint - keypoint name
        x - X coordinate of keypoint
        y - Y coordinate of keypoint

        coordinates are in units of pixels with (0, 0) at the upper-left corner of image
    '''

    image_paths = []
    for i in image:
        image_paths.extend(glob.glob(i))

    svc = DenseDepthService(
        model_path=model,
        keypoints_csv=keypoints,
        no_tif=no_tif,
        no_ply=no_ply,
    )

    return svc.predict(image_paths=image_paths)

if __name__ == '__main__':
    do_it()
