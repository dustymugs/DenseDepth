# distance between points
import click
from decimal import Decimal
from locale import localeconv
import math
import os
from PIL import Image
import re

def get_places(val):

    places = 0
    while val > 1.:

        val /= 10.
        places += 1

    return places

@click.command()
@click.argument('depth-file', type=click.Path())
@click.argument('x1', type=Decimal)
@click.argument('y1', type=Decimal)
@click.argument('x2', type=Decimal)
@click.argument('y2', type=Decimal)
@click.option('--scale-factor', default=1, type=Decimal, help="Value to add to internally derived scale")
def do_it(depth_file, x1, y1, x2, y2, scale_factor):
    '''
    Compute the distance between two points given a depth image and two points
    on said depth image.

    Point coordinates should be in image coordinates
    (e.g.upper-left corner is 0,0 and X, Y are positive values)
    '''

    assert os.path.isfile(depth_file)
    im = Image.open(depth_file)

    x1 = math.floor(x1)
    y1 = math.floor(y1)
    x2 = math.floor(x2)
    y2 = math.floor(y2)

    if x1 == x2 and y1 == y2:
        return 0.

    l1 = im.getpixel((x1, y1))
    l2 = im.getpixel((x2, y2))
    print(f'l1, l2: {l1}, {l2}')

    d1 = min(l1, l2)
    d2 = max(l1, l2)
    print(f'd1, d2: {d1}, {d2}')

    pixel_values = set()
    if x1 != x2:

        # get the line between (x1, y1) and (x2, y2)
        m = (y2 - y1) / (x2 - x1)
        b = -1 * m * x2 + y2

        print(f'y = {m}x + {b}')
        assert y1 == math.floor(m * x1 + b)
        assert y2 == math.floor(m * x2 + b)

        get_y = lambda x: m * x + b 

        # walk along x between x1 and x2
        x_step = 1
        _x = min(x1, x2)
        max_x = max(x1, x2)
        while _x < max_x:
            _y = math.floor(get_y(_x))
            coords = (_x, _y)

            pixel_values.add(
                (coords, im.getpixel(coords))
            )

            _x += 1 # by one pixel

    else:

        _y = math.floor(min(y1, y2))
        max_y = max(y1, y2)
        while _y < max_y:
            coords = (x1, _y)

            pixel_values.add(
                (coords, im.get_pixel(coords))
            )

            _y += 1 # by one pixel

    num_pixels = len(pixel_values)
    scale = get_places(num_pixels)
    print(f'num_pixels: {num_pixels}')
    print(f'scale: {scale}')

    # compute angle in pixel space

    # center of image
    center_x = im.width/ 2
    center_y = im.height / 2
    distance = math.sqrt(math.pow(center_x - x1, 2) + math.pow(center_y - y1, 2))
    print(distance)
    distance = math.sqrt(math.pow(center_x - x2, 2) + math.pow(center_y - y2, 2))
    print(distance)

    # greatest variability here given we arbitrarily choose the value of p1
    p1 = math.pow(10, scale + scale_factor)
    p2 = num_pixels

    # angles are in radians
    phi = math.atan2(p2, p1)
    theta = (math.pi - phi) / 2.
    print(f'phi: {phi}')
    print(f'theta: {theta}')

    a = d2 - d1
    b = math.sqrt(2. * math.pow(d1, 2.) * (1. - math.cos(phi)))
    gamma = math.pi - theta
    print(f'a: {a}')
    print(f'b: {b}')
    print(f'gamma: {gamma}')

    c = math.sqrt(math.pow(a, 2) + math.pow(b, 2) - (2 * a * b * math.cos(gamma)))
    print(f'c: {c}')

if __name__ == '__main__':
    do_it()
