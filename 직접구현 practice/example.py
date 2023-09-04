def conv(X, filters, stride=1, pad=0):
    n, c, h, w = X.shape
    n_f, _, filter_h, filter_w = filters.shape

    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    # add padding to height and width.
    in_X = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    out = np.zeros((n, n_f, out_h, out_w))

    for i in range(n): # for each image.
        for c in range(n_f): # for each channel.
            for h in range(out_h): # slide the filter vertically.
                h_start = h * stride
                h_end = h_start + filter_h
                for w in range(out_w): # slide the filter horizontally.
                    w_start = w * stride
                    w_end = w_start + filter_w
                    # Element-wise multiplication.
                    out[i, c, h, w] = np.sum(in_X[i, :, h_start:h_end, w_start:w_end] * filters[c])
