#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# the resulting .ply file can be viewed for example with meshlab
# sudo apt-get install meshlab

"""
This script reads a registered pair of color and depth images and generates a
colored 3D point cloud in the PLY format.
"""

import argparse
import math
import numpy as np
import os
from PIL import Image
import sys

# specs from XBox Kinect v1
# reference: http://wiki.ros.org/kinect_calibration/technical

# in degrees
hFov = 57
vFov = 43

scalingFactor = 10. # for depth data from DenseDepth trained on NYUv2

def generate_pointcloud(rgb_file,depth_file,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    """
    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file)

    assert rgb.size == (640, 480), \
        "Images must be 640 x 480"
    assert rgb.size == depth.size, \
        "Color and depth image do not have the same resolution."
    assert rgb.mode == "RGB", \
        "Color image is not in RGB format"
    assert depth.mode == "F", \
        "Depth image is not in float32 format"

    rhFov = math.radians(hFov)
    rvFov = math.radians(vFov)
    fx = rgb.width / (2 * math.tan(rhFov / 2))
    fy = rgb.height / (2 * math.tan(rvFov / 2))
    print('fx, fy', fx, fy)

    width, height = rgb.size
    cx = width / 2
    cy = height / 2

    xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy

    length = width * height
    z = np.array(depth).reshape(length)
    pos = np.dstack((xx * z, yy * z, z)).reshape((length, 3)) * scalingFactor
    color = np.array(rgb).reshape((length, 3))

    with open(ply_file,"w") as fh:
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
    parser = argparse.ArgumentParser(description='''
    This script reads a registered pair of color and depth images and generates a colored 3D point cloud in the
    PLY format. 
    ''')
    parser.add_argument('rgb_file', help='input color image (format: png)')
    parser.add_argument('depth_file', help='input depth image (format: png)')
    parser.add_argument('ply_file', help='output PLY file (format: ply)')
    args = parser.parse_args()

    generate_pointcloud(args.rgb_file,args.depth_file,args.ply_file)
