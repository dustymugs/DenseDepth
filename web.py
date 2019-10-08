from flask import Flask, render_template, request, jsonify
from threading import Lock
from werkzeug.utils import secure_filename

from predict import DenseDepthService

app = Flask(__name__)

svc = None
svc_lock = Lock()

@app.route('/', methods=['POST'])
def predict():

    req = request.get_json()
    keypoints_csv = req['keypoints_csv']
    image_paths = req['image_paths']

    with svc_lock:
        svc.no_tif = True
        svc.no_ply = True
        svc.keypoints_csv = keypoints_csv
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
