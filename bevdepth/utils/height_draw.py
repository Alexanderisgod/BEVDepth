import numpy as np
import torch


def gaussian_2d(shape, sigma_x=1, sigma_y=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma_x * sigma_y))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def evenly_H_2d(shape, x, y, height):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    
    m, n = max(0.5, m), max(0.5, n)
    y = height/(2*m)*(m-y-0.5)
    h = y.repeat(2*n+1, axis=1)
    return h

def draw_heatmap(heatmap, center, radius_x, radius_y, k=1):
    """Get masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter_x, diameter_y = 2 * radius_x + 1, 2 * radius_y + 1
    # gaussian = gaussian_2d((diameter_y, diameter_x), sigma_x=(diameter_x/6), sigma_y=(diameter_y/6))
    x, y = int(center[0]), int(center[1])

    # 均匀分布
    gaussian = evenly_H_2d((diameter_y, diameter_x), x, y, 2)

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    # print(x, y, top, bottom, left, right)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius_y - top:radius_y + bottom,
                 radius_x - left:radius_x + right]).to(torch.float32)
    # print(gaussian.shape, masked_heatmap.shape, masked_gaussian.shape)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def get_coord(center, dim):
    '''
    Args:
        center: numpy of center point(x, y, z).
    '''
    l, w, h = dim
    coords = center[None,...].repeat(2, 0)
    if coords.shape[1]==0: return []
    coords[0, 2] += h/4
    coords[1, 2] -= h/4

    return coords

def img_pe(fH, fW):
    x_coords = np.linspace(0.5, fW - .5, fW, dtype=np.float32)\
        .reshape(1, fW).repeat(fH, 0)
    y_coords = np.linspace(0.5, fH - .5, fH, dtype=np.float32)\
        .reshape(fH, 1).repeat(fW, 1)

    pe = np.stack((x_coords, y_coords), axis=-1)
    return pe