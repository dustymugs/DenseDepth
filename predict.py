import click
import csv
import glob
import itertools
import math
import numpy as np
import os
import os.path as osp
from PIL import Image

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model

from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images

# specs from XBox Kinect v1
# reference: http://wiki.ros.org/kinect_calibration/technical

# in degrees
hFov = 57
vFov = 43

scalingFactor = 10. # for depth data from DenseDepth trained on NYUv2

rhFov = math.radians(hFov)
rvFov = math.radians(vFov)

# TODO add support for KITTI

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

    image_globs = image

    keypoints_map = {}
    if keypoints is not None:

        with open(keypoints) as fh:
            reader = csv.DictReader(fh)

            for row in reader:
                by_image = keypoints_map.setdefault(row['image'], [])
                by_image.append((keypoint, x, y))

    # Input images
    images_data = None 
    image_paths = []
    for image_glob in image_globs:

        image_path = glob.glob(image_glob)
        if len(image_path) < 1:
            continue

        image_paths.extend(image_path)

        # returns (image idx, height, width, num channels)
        image_data = load_images(image_path)
        if images_data is None:
            images_data = image_data
        else:
            images_data = np.append(images_data, image_data, axis=0)

    print('\nLoaded ({}) images'.format(images_data.shape[0], images_data.shape[1:]))

    # Custom object needed for inference and training
    custom_objects = {
        'BilinearUpSampling2D': BilinearUpSampling2D,
        'depth_loss_function': None
    }

    print('Loading model...')

    # Load model into GPU / CPU
    model = load_model(model, custom_objects=custom_objects, compile=False)

    print('Model loaded')

    # Compute results
    outputs = predict(model, images_data) # returns (image idx, height, width, num channels)

    # output results as depth tif and point cloud ply
    for idx, depth_data in enumerate(outputs):

        image_path = osp.abspath(image_paths[idx])
        image_base, image_ext = osp.splitext(image_path)
        depth_path = f'{image_base}.depth.tif'

        src = Image.open(image_path)

        # save depth image to tiff
        if not no_tif:

            print('{} => {}'.format(image_path, depth_path))
            im = Image.fromarray(np.moveaxis(depth_data, -1, 0)[0], mode='F')
            # resize depth image to original image
            im = im.resize((src.width, src.height), resample=Image.LANCZOS)
            im.save(depth_path, 'TIFF')

        # convert to point cloud
        fx = src.width / (2 * math.tan(rhFov / 2))
        fy = src.height / (2 * math.tan(rvFov / 2))

        width, height = src.size
        cx = width / 2
        cy = height / 2

        xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy

        length = width * height
        z = depth_data.reshape(length)
        pos = np.dstack((xx * z, yy * z, z)).reshape((length, 3)) * scalingFactor
        color = np.array(src).reshape((length, 3))

        image_keypoints = keypoints_map.get(image_paths[idx], [])
        if image_keypoints:

            with open(..., 'w') as fh:
                field_names = [
                    'from_row',
                    'from_keypoint',
                    'from_x',
                    'from_y',
                    'from_z',
                    'to_row',
                    'to_keypoint',
                    'to_x',
                    'to_y',
                    'to_z',
                    'from_to_distance',
                ]
                writer = csv.DictWriter(fh, fieldnames=field_names)

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
                        math.pow(b_voxel[0] - a_voxel[0], 2) + \
                        math.pow(b_voxel[1] - a_voxel[1], 2) + \
                        math.pow(b_voxel[2] - a_voxel[2], 2)
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

        # save point cloud to ply
        if not no_ply:

            ply_path = f'{image_base}.ply'
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

if __name__ == '__main__':
    do_it()
