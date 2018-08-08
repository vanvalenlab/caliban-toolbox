import numpy as np


""" Stitch images with Fiji results
Exported from Stitch_image_Grid_Sequence in Fiji

e.g.
points = [(0.0, 0.0), (-556.46246, -8.537842), (-565.82874, 533.8285), (-131.09012, 483.0899)]
"""

def relative_position(points):
    points_flat = [int(el) for inner in points for el in inner]
    xpoints = points_flat[1::2]
    ypoints = points_flat[0::2]
    max_xp = min(xpoints)
    max_yp = min(ypoints)
    rel_xpoints = map(lambda l: l - max_xp, xpoints)
    rel_ypoints = map(lambda l: l - max_yp, ypoints)
    rel_points = (rel_xpoints, rel_ypoints)
    return rel_points

def replace_peri(img, val=0):
    img[0, :] = val
    img[-1, :] = val
    img[:, 0] = val
    img[:, -1] = val
    return img

def stitching(img, rel_points):

    imShape=img.shape
    rel_xpoints=rel_points[0]
    rel_ypoints=rel_points[1]
    stitchImg = np.zeros((max(rel_xpoints)+imShape[0], max(rel_ypoints)+imShape[1], imShape[2]))

    for pos in range(imShape[2]):
        tmp = img[:, :, pos]
        stitchImg[rel_xpoints[pos]:rel_xpoints[pos]+imShape[0], rel_ypoints[pos]:rel_ypoints[pos]+imShape[1], pos] = tmp

    stitchImg[:,:,-1] = np.array(stitchImg.max(axis=2), dtype=np.float32)
    stitchImg = stitchImg[sorted(rel_xpoints)[1]:imShape[0]+sorted(rel_xpoints)[2], sorted(rel_ypoints)[1]:imShape[0]+sorted(rel_ypoints)[2]]
    return stitchImg

