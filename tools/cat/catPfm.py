import re,cv2,sys
import numpy as np
import matplotlib.pyplot as plt


def catPfm(path):
    file=open(path, 'rb')
    header = file.readline().decode('UTF-8').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    depth_image = cv2.flip(data, 0)

    ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
    min, max = ma.min(), ma.max()
    print('value range: ', min, max)

    plt.title(path)
    plt.imshow(depth_image, 'rainbow')
    # plt.colorbar(extend="both", ticks=[min, min+0.25*(max-min), min+0.5*(max-min), min+0.75*(max-min), max])
    plt.colorbar(ticks=[min, min+0.25*(max-min), min+0.5*(max-min), min+0.75*(max-min), max], shrink=0.8)
    # plt.colorbar(extend="both", ticks=[0, 0.25, 0.5, 0.75, 1])  #
    plt.show()


if __name__=="__main__":
    catPfm(sys.argv[1])
