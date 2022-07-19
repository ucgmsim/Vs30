from io import BytesIO
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd


class CPT:
    """
    Contains the data from a CPT file
    """

    def __init__(
        self,
        name: str,
        depth: np.ndarray,
        qc: np.ndarray,
        fs: np.ndarray,
        u: np.ndarray,
        info: Dict = None,
    ):
        self.name = name
        self.depth = depth
        self.Qc = qc
        self.Fs = fs
        self.u = u
        self.info = info

        # cpt parameter info init for lazy loading
        self._qt = None
        self._Ic = None
        self._Qtn = None
        self._effStress = None

    @property
    def qt(self):
        """
        Gets the qt value and computes the value if not set
        """
        if self._qt is None:
            # compute pore pressure corrected tip resistance
            a = 0.8
            self._qt = self.Qc - self.u * (1 - a)
        return self._qt

    @property
    def Ic(self):
        """
        Gets the Ic value and computes the value if not set
        """
        if self._Ic is None:
            # atmospheric pressure (MPa)
            pa = 0.1
            # compute non-normalised Ic based on the correlation by Robertson (2010).
            Rf = (self.Fs / self.Qc) * 100
            self._Ic = (
                (3.47 - np.log10(self.Qc / pa)) ** 2 + (np.log10(Rf) + 1.22) ** 2
            ) ** 0.5
        return self._Ic

    @property
    def Qtn(self):
        """
        Gets the Qtn value and computes the value if not set
        """
        if self._Qtn is None:
            self._Qtn, self._effStress = self.calc_cpt_params()
        return self._Qtn

    @property
    def effStress(self):
        """
        Gets the effStress value and computes the value if not set
        """
        if self._effStress is None:
            self._Qtn, self._effStress = self.calc_cpt_params()
        return self._effStress

    def calc_cpt_params(self):
        """Compute and save Qtn and effStress CPT parameters"""
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

        n = 0.381 * self.Ic + 0.05 * (effStress / pa) - 0.15
        for i in range(0, len(n)):
            if n[i] > 1:
                n[i] = 1
        Qtn = ((self.qt - totalStress) / pa) * (pa / effStress) ** n

        return Qtn, effStress

    def to_json(self):
        """
        Creates a json response dictionary from the CPT
        """
        return {
            "name": self.name,
            "depth": self.depth.tolist(),
            "Qc": self.Qc.tolist(),
            "Fs": self.Fs.tolist(),
            "u": self.u.tolist(),
            "info": self.info,
            "qt": self._qt,
            "Ic": self._Ic,
            "Qtn": self._Qtn,
            "effStress": self._effStress,
        }

    @staticmethod
    def from_json(json: Dict):
        """
        Creates a CPT from a json dictionary string
        """
        return CPT(
            json["name"],
            np.asarray(json["depth"]),
            np.asarray(json["Qc"]),
            np.asarray(json["Fs"]),
            np.asarray(json["u"]),
            json["info"],
        )

    @staticmethod
    def from_file(cpt_ffp: str):
        """
        Creates a CPT from a CPT file
        """
        cpt_ffp = Path(cpt_ffp)
        data = np.loadtxt(cpt_ffp, dtype=float, delimiter=",", skiprows=1)
        depth, qc, fs, u, info = CPT.process_cpt(data)
        return CPT(cpt_ffp.stem, depth, qc, fs, u, info)

    @staticmethod
    def from_byte_stream(file_name: str, stream: bytes):
        """
        Creates a CPT from a file stream
        """
        csv_data = pd.read_csv(BytesIO(stream))
        data = np.asarray(csv_data)
        depth, qc, fs, u, info = CPT.process_cpt(data)
        return CPT(Path(file_name).stem, depth, qc, fs, u, info)

    @staticmethod
    def process_cpt(data: np.ndarray):
        """Process CPT data and returns depth, Qc, Fs, u, info"""
        # Get CPT info
        info = dict()
        info["z_min"] = np.round(data[0, 0], 2)
        info["z_max"] = np.round(data[-1, 0], 2)
        info["z_spread"] = np.round(data[-1, 0] - data[0, 0], 2)

        # Filtering
        below_30_filter = np.all(data[:, [0]] <= 30, axis=1)
        info["Removed rows"] = np.where(below_30_filter == False)[0]
        data = data[below_30_filter.T]  # z is less then 30 m
        zero_filter = np.all(data[:, [1, 2]] > 0, axis=1)
        info["Removed rows"] = np.concatenate(
            (
                (np.where(zero_filter == False)[0]),
                info["Removed rows"],
            )
        ).tolist()
        data = data[zero_filter]  # delete rows with zero qc, fs

        if len(data) == 0:
            raise Exception("CPT File has no valid lines")

        z_raw = data[:, 0]  # m
        qc_raw = data[:, 1]  # MPa
        fs_raw = data[:, 2]  # MPa
        u_raw = data[:, 3]  # Mpa

        downsize = np.arange(z_raw[0], 30.02, 0.02)
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
