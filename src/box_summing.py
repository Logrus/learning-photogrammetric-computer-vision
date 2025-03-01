import numpy


def box_summing(image, kernel_size):
    """Fast box summing convolution using integral images."""

    h, w = image.shape
    k = kernel_size // 2

    output = np.zeros_like(image, dtype=np.float64)

    integral = np.cumsum(np.cumsum(np.asarray(image, dtype=np.float64), axis=0), axis=1)

    for y in range(k, h - k):
        for x in range(k, w - k):
            # left x, top y
            lx, ty = x - k, y - k
            # right x, bottom y
            rx, by = x + k, y + k

            sum_value = float(integral[by, rx])
            if lx > 0:
                sum_value -= integral[by, lx - 1]
            if ty > 0:
                sum_value -= integral[ty - 1, rx]
            if lx > 0 and ty > 0:
                sum_value += integral[ty - 1, lx - 1]

            output[y, x] = sum_value

    return output
