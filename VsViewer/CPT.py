import numpy as np
from pathlib import Path


class CPT:
    """
    Contains the data from a CPT file
    """

    def __init__(self, cpt_ffp: str):
        self.cpt_ffp = Path(cpt_ffp)
        depth, qc, fs, u, info = self.get_cpt_data()
        self.depth = depth
        self.Qc = qc
        self.Fs = fs
        self.u = u
        self.info = info

    def get_cpt_data(self):
        """Get CPT data and return [z; qc; fs; u2; info]"""
        data = np.loadtxt(self.cpt_ffp, dtype=float, delimiter=",", skiprows=1)

        # Get CPT info
        info = dict()
        info["z_min"] = np.round(data[0, 0], 2)
        info["z_max"] = np.round(data[-1, 0], 2)
        info["z_spread"] = np.round(data[-1, 0] - data[0, 0], 2)

        # Filtering
        data = data[(np.all(data[:, [0]] < 30, axis=1)).T]  # z is less then 30 m
        info["Removed rows containing 0 or below Fs or Qc values"] = not np.alltrue(
            data[:, [1, 2]] > 0
        )
        data = data[np.all(data[:, [1, 2]] > 0, axis=1)]  # delete rows with zero qc, fs

        if len(data) == 0:
            raise Exception("CPT File has no valid lines")

        z_raw = data[:, 0]  # m
        qc_raw = data[:, 1]  # MPa
        fs_raw = data[:, 2]  # MPa
        u_raw = data[:, 3]  # Mpa

        downsize = np.arange(z_raw[0], 30, 0.02)
        z = np.array([])
        qc = np.array([])
        fs = np.array([])
        u = np.array([])
        for j in range(len(downsize)):
            for i in range(len(z_raw)):
                if abs(z_raw[i] - downsize[j]) < 0.001:
                    z = np.append(z, z_raw[i])
                    qc = np.append(qc, qc_raw[i])
                    fs = np.append(fs, fs_raw[i])
                    u = np.append(u, u_raw[i])

        if len(u) > 50:
            while u[50] >= 10:
                u = u / 1000  # account for differing units

        # some units are off - so need to see if conversion is needed
        if len(fs) > 100:
            # Account for differing units
            if fs[100] > 1.0:
                fs = fs / 1000
        elif len(fs) > 5:
            if fs[5] > 1.0:
                fs = fs / 1000
        else:
            fs = fs

        return z, qc, fs, u, info
