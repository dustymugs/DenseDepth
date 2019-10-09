from flask import Flask, render_template, request, jsonify
from threading import Lock
from werkzeug.utils import secure_filename

from predict import DenseDepthService

app = Flask(__name__)

svc = None
svc_lock = Lock()

@app.route('/', methods=['POST'])
def predict():
    '''
    request: {
        'keypoints': {
            PATH_TO_IMAGE_1: {
                KEYPOINT: {
                    'x': 123456,
                    'y': 123456,
                    'present': true
                },
                KEYPOINT: {
                    'x': 123456,
                    'y': 123456,
                    'present': false
                },
                ...
            },
            ...
        },
        'image_paths': [
            PATH_TO_IMAGE_1,
            PATH_TO_IMAGE_2,
            ...
        ]
    }

    response: {
        PATH_TO_IMAGE_1: {
            'depth_image': PATH_TO_DEPTH_IMAGE,
            'kp_distance': PATH_TO_KEYPOINT_DISTANCES_FILE,
            'pc_ply': PATH_TO_POINTCLOUD_PLY,
        },
        ...
    }
    '''

    req = request.get_json()
    keypoints = req['keypoints']
    image_paths = req['image_paths']

    # convert keypoints json to tuples
    keypoints_map = {
        image_path: [
            (kp_name, int(kp_data['x']), int(kp_data['y']))
            for kp_name, kp_data in kps_data.items()
        ]
        for image_path, kps_data in keypoints.items()
    }

    with svc_lock:
        svc.no_tif = True
        svc.no_ply = True
        svc.keypoints_map = keypoints_map
        paths = svc.predict(image_paths)

    return jsonify({
        image_path: {
            'depth_image': tif_path,
            'kp_distance': dist_path,
            'pc_ply': ply_path
        }
        for image_path, tif_path, dist_path, ply_path in paths
    })

if __name__ == '__main__':

    svc = DenseDepthService()
    app.run(debug=False, host='0.0.0.0', port=5051)
