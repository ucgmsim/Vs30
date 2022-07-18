from io import BytesIO
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd


class SPT:
    """
    Contains the data from an SPT file
    """

    def __init__(
        self,
        name: str,
        depth: np.ndarray,
        n: np.ndarray,
        hammer_type: str,
        borehole_diameter: float,
        energy_ratio: float,
        soil_type: str,
    ):
        self.name = name
        self.depth = depth
        self.N = n
        self.hammer_type = hammer_type
        self.borehole_diameter = borehole_diameter
        self.energy_ratio = energy_ratio
        self.soil_type = soil_type

        # spt parameter info init for lazy loading
        self._n60 = None

    @property
    def N60(self):
        """
        Gets the N60 value and computes the value if not set
        """
        if self._n60 is None:
            N60_list = []
            NValues = spt_data_at_this_location["NValue"]
            hammer_type = spt_location["HammerType"]
            borehole_dia = spt_location["BoreholeDiameter"]
            energy_ratio = spt_location["EnergyRatio"]
            rod_length = spt_data_at_this_location["Depth"]
            for j in range(NValues.count()):
                N = NValues[j]
                Ce, Cb, Cr, Cs = variables.all_variables(energy_ratio, hammer_type, borehole_dia, rod_length[j])
                N60 = N * Ce * Cb * Cr * Cs
                N60_list.append(N60)
            self._n60 = N60_list
        return self._n60

    def to_json(self):
        """
        Creates a json response dictionary from the CPT
        """
        json_dict = {
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
        return json_dict

    @staticmethod
    def from_json(json: Dict):
        """
        Creates a CPT from a json dictionary string
        """
        name, depth, qc, fs, u, info = (
            json["name"],
            np.asarray(json["depth"]),
            np.asarray(json["Qc"]),
            np.asarray(json["Fs"]),
            np.asarray(json["u"]),
            json["info"],
        )
        return CPT(name, depth, qc, fs, u, info)

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
        info["Removed rows"] = np.where(np.all(data[:, [0]] <= 30, axis=1) == False)[0]
        data = data[(np.all(data[:, [0]] <= 30, axis=1)).T]  # z is less then 30 m
        info["Removed rows"] = np.concatenate(
            (
                (np.where(np.all(data[:, [1, 2]] > 0, axis=1) == False)[0]),
                info["Removed rows"],
            )
        ).tolist()
        data = data[np.all(data[:, [1, 2]] > 0, axis=1)]  # delete rows with zero qc, fs

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
