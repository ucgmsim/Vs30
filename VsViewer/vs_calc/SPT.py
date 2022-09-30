from io import BytesIO
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import HammerType, SoilType


class SPT:
    """
    Contains the data from an SPT file
    """

    def __init__(
        self,
        name: str,
        depth: np.ndarray,
        n: np.ndarray,
        hammer_type: HammerType = HammerType.Auto,
        borehole_diameter: float = 150,
        energy_ratio: float = None,
        soil_type: np.ndarray = None,
    ):
        self.name = name
        self.depth = depth
        self.N = n
        self.hammer_type = hammer_type
        self.borehole_diameter = borehole_diameter
        self.energy_ratio = energy_ratio
        self.soil_type = (
            np.repeat(SoilType.Clay, len(depth)) if soil_type is None else soil_type
        )
        self.info = {
            "z_min": depth[0],
            "z_max": depth[-1],
            "z_spread": depth[-1] - depth[0],
            "removed_rows": [],
        }

        # spt parameter info init for lazy loading
        self._n60 = None

    @property
    def N60(self):
        """
        Gets the N60 value and computes the value if not set
        """
        if self._n60 is None:
            N60_list = []
            for idx, N in enumerate(self.N):
                Ce, Cb, Cr = self.calc_n60_variables(
                    self.energy_ratio,
                    self.hammer_type,
                    self.borehole_diameter,
                    self.depth[idx],
                )
                N60 = round(N * Ce * Cb * Cr, 2)
                N60_list.append(N60)
            self._n60 = np.asarray(N60_list)
        return self._n60

    def to_json(self):
        """
        Creates a json response dictionary from the SPT
        """
        return {
            "name": self.name,
            "depth": self.depth.tolist(),
            "N": self.N.tolist(),
            "hammer_type": self.hammer_type.name,
            "borehole_diameter": self.borehole_diameter,
            "energy_ratio": self.energy_ratio,
            "soil_type": [soil_type.name for soil_type in self.soil_type],
            "info": self.info,
            "N60": self.N60.tolist(),
        }

    @staticmethod
    def from_json(json: Dict):
        """
        Creates a SPT from a json dictionary string
        """
        spt = SPT(
            json["name"],
            np.asarray(json["depth"]),
            np.asarray(json["N"]),
            HammerType[json["hammer_type"]],
            float(json["borehole_diameter"]),
            None if json["energy_ratio"] is None else float(json["energy_ratio"]),
            [SoilType[soil_type] for soil_type in json["soil_type"]],
        )
        spt._n60 = None if json["N60"] is None else np.asarray(json["N60"])
        return spt

    @staticmethod
    def from_file(spt_ffp: str):
        """
        Creates an SPT from an SPT file
        """
        spt_ffp = Path(spt_ffp)
        data = pd.read_csv(spt_ffp)
        soil_type = None if "Soil" in data.columns else data["Soil"]
        return SPT(
            spt_ffp.stem,
            data.iloc[:, 0].values,
            data.iloc[:, 1].values,
            soil_type=soil_type,
        )

    @staticmethod
    def from_byte_stream_form(file_name: str, stream: bytes, form: Dict):
        """
        Creates an SPT from a file stream and form data
        """
        file_name = Path(file_name)
        file_data = (
            pd.read_csv(BytesIO(stream))
            if file_name.suffix == ".csv"
            else pd.read_excel(BytesIO(stream))
        )
        # Manage soil type from file first then form
        soil_type = (
            file_data["Soil"]
            if "Soil" in file_data.columns
            else (None if form["soilType"] == "" else np.repeat(form["soilType"], len(file_data.values)))
        )
        return SPT(
            form.get("sptName", file_name.stem),
            np.asarray(file_data["Depth"]),
            np.asarray(file_data["NValue"]),
            HammerType.Auto
            if form["hammerType"] == ""
            else HammerType[form["hammerType"]],
            form["boreholeDiameter"],
            None if form["energyRatio"] == "" else form["energyRatio"],
            None if soil_type is None else np.asarray([SoilType[soil] for soil in soil_type]),
        )

    @staticmethod
    def calc_n60_variables(
        energy_ratio: float,
        hammer_type: HammerType,
        borehole_diameter: float,
        rod_length: float,
    ):
        """
        Calculates the variables needed to get N60 from N
        Returns the variables Ce, Cr, Cb
        """
        # Calc Ce
        # In case the data is messed up and none of the following condition can be meet
        # assume a relative average Ce value of 0.8
        Ce = 0.8
        if energy_ratio is not None:
            Ce = energy_ratio / 60
        else:
            if hammer_type == HammerType.Auto:
                # range 0.8 to 1.3
                Ce = 0.8
            elif hammer_type == HammerType.Safety:
                # safety hammer, it has range of 0.7 to 1.2
                Ce = 0.7
            elif hammer_type == HammerType.Standard:
                # for doughnut hammer range 0.5 to 1.0
                Ce = 0.5

        # Calc Cr
        if rod_length < 3:
            Cr = 0.75
        elif 3 <= rod_length < 4:
            Cr = 0.8
        elif 4 <= rod_length < 6:
            Cr = 0.85
        elif 6 <= rod_length < 10:
            Cr = 0.95
        else:
            Cr = 1

        # Calc Cb
        if 65 <= borehole_diameter <= 115:
            Cb = 1
        elif borehole_diameter == 200:
            Cb = 1.15
        else:
            Cb = 1.05

        return Ce, Cr, Cb
