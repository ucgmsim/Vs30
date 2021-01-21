
def gidx2val(model, gidx):
    """
    
    """
    vals = np.empty((len(gidx), 2), dtype=np.float32)
    vals[...] = np.nan

    valid_idx = gidx != 255
    vals[valid_idx] = model[gidx[valid_idx]]

    return vals
