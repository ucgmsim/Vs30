

def weighted(geology_vs30, geology_stdv, terrain_vs30, terrain_stdv, outfile, stdv_weight=F, k=1):
    """
    Combine geology and terrain models (path to geotiff files).
    """

    if stdv_weight:
        m_g = (geology_stdv ** 2) ** -k
        m_t = (terrain_stdv ** 2) ** -k
        w_g = m_g / (m_g + m_t)
        w_t = m_t / (m_g + m_t)
        del m_g, m_t
    else:
        w_g = 0.5
        w_t = 0.5

    log_gt = log(geology_vs30) * w_g + log(terrain_vs30) * w_t

    outfile_vs30 = exp(log_gt)
    outfile_stdv = (w_g * ((log(geology_stdv) - log_gt) ** 2 + geology_stdv ** 2) +
                    w_t * ((log(terrain_stdv) - log_gt) ** 2 + terrain_stdv ** 2)) ** 0.5
