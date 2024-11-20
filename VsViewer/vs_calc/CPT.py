from io import BytesIO
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

class ExceededMaxIterations(Exception):
    pass

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
        is_kpa: bool = False,
        ground_water_level: float = 1,
        net_area_ratio: float = 0.8,
    ):
        self.name = name
        self.depth = depth
        self.Qc = qc
        self.Fs = fs
        self.u = u
        self.info = info
        self.is_kpa = is_kpa
        self.ground_water_level = ground_water_level
        self.net_area_ratio = net_area_ratio

        # cpt parameter info init for lazy loading
        self._qt = None
        self._Ic = None
        self._Qtn = None
        self._effStress = None
        self._n = None
        self._gamma = None
        self._totalStress = None

    @property
    def qt(self):
        """
        Gets the qt value and computes the value if not set
        """
        if self._qt is None:
            self._qt = self.Qc + self.u * (1 - self.net_area_ratio)
        return self._qt

    @property
    def Qtn(self):
        """
        Gets the Qtn value and computes the value if not set
        """
        if self._Qtn is None:
            (
                self._Qtn,
                self._effStress,
                self._Ic,
                self._n,
                self._totalStress,
            ) = self.calc_cpt_params()
        return self._Qtn

    @property
    def effStress(self):
        """
        Gets the effStress value and computes the value if not set
        """
        if self._effStress is None:
            (
                self._Qtn,
                self._effStress,
                self._Ic,
                self._n,
                self._totalStress,
            ) = self.calc_cpt_params()
        return self._effStress

    @property
    def Ic(self):
        """
        Gets the Ic value and computes the value if not set
        """
        if self._Ic is None:
            (
                self._Qtn,
                self._effStress,
                self._Ic,
                self._n,
                self._totalStress,
            ) = self.calc_cpt_params()
        return self._Ic

    @property
    def n(self):
        """
        Gets the n value and computes the value if not set
        """
        if self._n is None:
            (
                self._Qtn,
                self._effStress,
                self._Ic,
                self._n,
                self._totalStress,
            ) = self.calc_cpt_params()
        return self._n

    @property
    def totalStress(self):
        """
        Gets the totalStress value and computes the value if not set
        """
        if self._totalStress is None:
            (
                self._Qtn,
                self._effStress,
                self._Ic,
                self._n,
                self._totalStress,
            ) = self.calc_cpt_params()
        return self._totalStress

    @property
    def gamma(self):
        """
        It estimates the soil total unit weight - in (MN/m3)

        According to Robertson & Cabal (2010)

        References
        ----------
        Robertson P.K., Cabal K.L. (2010). Estimating soil unit weight from CPT. 2nd International
        Symposium on Cone Penetration Test, CPT'10, Huntington Beach, CA, USA.
        """
        default_gamma = 0.00981 * 1.9  # (MN/m3)
        # Unit weight of water (kN/m3)
        gamma_w = 9.80665
        # atmospheric pressure (MPa)
        pa = 0.1
        # Friction ratio
        Rf = self.Fs / self.qt * 100
        # Equation is in kN/m3, then converted back into MN/n3 (Rf and qt / pa are ratios so no need to convert)
        gamma = (
            (0.27 * np.log10(Rf) + 0.36 * np.log10(self.qt / pa) + 1.236) * gamma_w
        ) / 1000
        # If the values of qc or fs are zero, negative or non-existent, then enforce default gamma
        gamma = np.where((self.Qc <= 0) | (self.Fs <= 0), default_gamma, gamma)
        # If less than 14kN/m3, use 14kN/m3
        gamma = np.maximum(14.0 / 1000, gamma)
        return gamma

    def calc_cpt_params(self, max_iterations: int = 1e6):
        """
        Compute and save Qtn, Ic, n, effStress and totalStress CPT parameters

        Returns
        -------
        Qtn : np.ndarray
            Normalized cone tip resistance
        effStress : np.ndarray
            Effective stress
        Ic : np.ndarray
            Cone resistance factor
        n : np.ndarray
            Cone resistance factor exponent
        totalStress : np.ndarray
            Total stress
        max_iterations : int
            Maximum number of allowed iterations


        References
        ----------
        P.K. Robertson (1990, 2009, 2010)
        Robertson and Wride (1998)
        """
        # atmospheric pressure (MPa)
        pa = 0.1
        # compute vertical stress profile
        totalStress = np.zeros(len(self.depth))
        totalStress[0] = self.gamma[0] * self.depth[0]
        u0 = np.zeros(len(self.depth))
        for i in range(1, len(self.depth)):
            totalStress[i] = (
                self.gamma[i] * (self.depth[i] - self.depth[i - 1]) + totalStress[i - 1]
            )
            if self.depth[i] >= self.ground_water_level:
                u0[i] = 0.00981 * (self.depth[i] - self.ground_water_level)

        effStress = totalStress - u0

        Fr = (self.Fs / (self.qt - totalStress)) * 100
        Qtn = np.zeros(len(self.depth))
        Ic = np.zeros(len(self.depth))
        # assumed initial value of n
        n = 0.5 * np.ones(len(self.depth))

        # loop over each depth point
        for i in range(0, len(n)):
            deltan = 1
            # iterate Qtn, Ic and n until convergence
            iteration_counter = 0
            while deltan >= 0.01:
                iteration_counter += 1
                if iteration_counter > max_iterations:
                    raise ExceededMaxIterations(f"Exceeded the maximum number of iterations ({max_iterations:.1e}) "
                                                f"for {self.name} without converging to a numerical solution.")
                n0 = n[i]
                cN = (pa / effStress[i]) ** n[i]
                if cN > 1.7:
                    cN = 1.7
                Qtn[i] = ((self.qt[i] - totalStress[i]) / pa) * cN
                Ic[i] = (
                    (3.47 - np.log10(Qtn[i])) ** 2 + (np.log10(Fr[i]) + 1.22) ** 2
                ) ** 0.5
                n[i] = 0.381 * Ic[i] + 0.05 * (effStress[i] / pa) - 0.15
                if n[i] > 1:
                    n[i] = 1
                deltan = np.abs(n0 - n[i])

        return Qtn, effStress, Ic, n, totalStress

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
            "is_kpa": self.is_kpa,
            "gwl": self.ground_water_level,
            "nar": self.net_area_ratio,
            "qt": None if self._qt is None else self._qt.tolist(),
            "Ic": None if self._Ic is None else self._Ic.tolist(),
            "Qtn": None if self._Qtn is None else self._Qtn.tolist(),
            "effStress": None if self._effStress is None else self._effStress.tolist(),
            "n": None if self._n is None else self._n.tolist(),
            "gamma": None if self._gamma is None else self._gamma.tolist(),
            "totalStress": None if self._totalStress is None else self._totalStress,
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
            json["is_kpa"],
            json["gwl"],
            json["nar"],
        )

    @staticmethod
    def from_file(cpt_ffp: str, gwl: float = 1, nar: float = 0.8):
        """
        Creates a CPT from a CPT file

        Parameters
        ----------
        cpt_ffp : str
            The file path to the CPT file
        gwl : float, optional
            The groundwater level, by default 1
        nar : float, optional
            The net area ratio, by default 0.8
        """
        cpt_ffp = Path(cpt_ffp)
        data = np.loadtxt(cpt_ffp, dtype=float, delimiter=",", skiprows=1)
        depth, qc, fs, u, info = CPT.process_cpt(data)
        return CPT(
            cpt_ffp.stem,
            depth,
            qc,
            fs,
            u,
            info,
            ground_water_level=gwl,
            net_area_ratio=nar,
        )

    @staticmethod
    def from_byte_stream(file_name: str, stream: bytes, form: Dict):
        """
        Creates a CPT from a file stream
        """
        file_name = Path(file_name)
        file_data = (
            pd.read_csv(BytesIO(stream))
            if file_name.suffix == ".csv"
            else pd.read_excel(BytesIO(stream))
        )
        data = np.asarray(file_data)
        is_kpa = form.get("iskPa") == "True"
        depth, qc, fs, u, info = CPT.process_cpt(data, is_kpa)
        return CPT(
            form.get("cptName", file_name.stem),
            depth,
            qc,
            fs,
            u,
            info,
            is_kpa,
            float(form.get("gwl")),
            float(form.get("nar")),
        )

    @staticmethod
    def process_cpt(data: np.ndarray, is_kpa: bool = False):
        """Process CPT data and returns depth, Qc, Fs, u, info"""
        # Convert units to MPa if needed
        if is_kpa:
            data[:, [1, 2, 3]] = data[:, [1, 2, 3]] / 1000

        # Get CPT info
        info = dict()
        info["z_min"] = np.round(data[0, 0], 2)
        info["z_max"] = np.round(data[-1, 0], 2)
        info["z_spread"] = np.round(data[-1, 0] - data[0, 0], 2)

        # Filtering
        zero_filter = np.all(data[:, [1, 2]] > 0, axis=1)
        info["Removed rows"] = np.where(zero_filter == False)[0].tolist()
        data = data[zero_filter]  # delete rows with zero qc, fs

        if len(data) == 0:
            raise Exception("CPT File has no valid lines")

        z = data[:, 0]  # m
        qc = data[:, 1]  # MPa
        fs = data[:, 2]  # MPa
        u = data[:, 3]  # Mpa

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
