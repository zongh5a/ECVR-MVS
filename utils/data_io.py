import re, sys, torch
import numpy as np
from PIL import Image


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()

def write_depth_img(filename, depth):
    min, max = depth.min(), depth.max()
    image = Image.fromarray((depth-min)/(max-min)*255).convert("L")
    image.save(filename)

    return 1

# read pair.txt
def read_pairfile(pair_path):
    num_viewpoint,pairs = 0, []
    with open(pair_path) as f:
        num_viewpoint = int(f.readline())
        # viewpoints (49)
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            pairs.append([ref_view, src_views])

    return num_viewpoint, pairs

def tocuda(data_batch, device, parallel=False):
    r = {}
    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor):
            if parallel:
                r[k] = v.cuda()
            else:
                r[k] = v.to(device)
        elif isinstance(v, dict):
            if parallel:
                r[k] = {k_: v_.cuda() for k_, v_ in v.items()}
            else:
                r[k] = {k_: v_.to(device) for k_, v_ in v.items()}
        else:
            continue
    return r

def save_camfile(intrinsic, extrinsic, depth_range, save_location):
    content = "extrinsic" + "\n"
    for r in extrinsic:
        for i in r:
            content += str(i) + " "
        content += "\n"
    content += "\n" + "intrinsic" + "\n"
    for r in intrinsic:
        for i in r:
            content += str(i) + " "
        content += "\n"
    content += "\n" + str(depth_range[0]) + " " + str(depth_range[1])


    with open(save_location, "w") as f:
        f.write(content)