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
        z_raw = data[:, 0]  # m
        qc_raw = data[:, 1]  # MPa
        fs_raw = data[:, 2]  # MPa
        u_raw = data[:, 3]  # Mpa

        if len(data) == 0:
            raise Exception("CPT File has no valid lines")

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

    def get_cpt_params(self):
        """Compute basic CPT parameters"""
        # compute pore pressure corrected tip resistance
        a = 0.8
        qt = self.Qc - self.u * (1 - a)
        # assume soil unit weight (MN/m3)
        gamma = 0.00981 * 1.9
        # atmospheric pressure (MPa)
        pa = 0.1
        # groundwater table depth(m)
        gwt = 1.0
        # compute vertical stress profile
        totalStress = np.zeros(len(self.depth))
        u0 = np.zeros(len(self.depth))
        for i in range(1, len(self.depth)):
            totalStress[i] = (
                gamma * (self.depth[i] - self.depth[i - 1]) + totalStress[i - 1]
            )
            if self.depth[i] >= gwt:
                u0[i] = 0.00981 * (self.depth[i] - self.depth[i - 1]) + u0[i - 1]
        effStress = totalStress - u0
        effStress[0] = effStress[1]  # fix error caused by dividing 0

        # compute non-normalised Ic based on the correlation by Robertson (2010).
        Rf = (self.Fs / self.Qc) * 100
        Ic = ((3.47 - np.log10(self.Qc / pa)) ** 2 + (np.log10(Rf) + 1.22) ** 2) ** 0.5
        n = 0.381 * Ic + 0.05 * (effStress / pa) - 0.15
        for i in range(0, len(n)):
            if n[i] > 1:
                n[i] = 1
        Qtn = ((qt - totalStress) / pa) * (pa / effStress) ** n

        # note in Chris's code, Qtn is used instead of qc1n or qt1n
        # does not make that much of difference
        qc1n = Qtn
        qt1n = Qtn
        return qt, Ic, Qtn, qc1n, qt1n, effStress
