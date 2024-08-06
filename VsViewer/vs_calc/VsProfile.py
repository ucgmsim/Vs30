from io import BytesIO
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from vs_calc import CPT, SPT, utils, spt_vs_correlations, cpt_vs_correlations, vs30_correlations


class VsProfile:
    """
    Contains the data for a Vs Profile
    """

    def __init__(
        self,
        name: str,
        vs: np.ndarray,
        vs_sd: np.ndarray,
        depth: np.ndarray,
        eff_stress: Optional[np.ndarray] = None,
        vs_correlation: Optional[str] = None,
        vs30_correlation: Optional[str] = None,
        layered: Optional[bool] = False,
        average_vs_under_3m: Optional[bool] = False
    ):
        self.name = name
        self.vs_correlation = vs_correlation
        self.vs30_correlation = vs30_correlation
        self.layered = layered
        # Ensures that the VsZ calculation will be done using the highest int depth below 30m
        # for correlations to Vs30
        reduce_to = min(int(max(depth)), 30)
        int_depth_mask = depth <= reduce_to
        to_remove = np.where(int_depth_mask == False)[0]
        to_keep = np.where(int_depth_mask == True)[0]
        if len(to_remove) != 0 and reduce_to != depth[to_keep[-1]]:
            first_remove = to_remove[0]
            last_to_keep = to_keep[-1]
            middle = (depth[first_remove] + depth[last_to_keep]) / 2
            if middle < reduce_to:
                # Higher value is closer set higher value to the float value
                int_depth_mask[first_remove] = True
                depth[first_remove] = reduce_to
            else:
                if reduce_to != depth[last_to_keep]:
                    # Grab lower value but raise it to a non float value as it's closer
                    int_depth_mask[first_remove] = True
                    depth[first_remove] = reduce_to
                    vs[first_remove] = vs[last_to_keep]
                    vs_sd[first_remove] = vs_sd[last_to_keep]
        self.max_depth = reduce_to
        if average_vs_under_3m:
            shal_vs_avg = np.average(vs[(depth >= 2.5) & (depth <= 3.5)])
            vs[depth <= 3.0] = shal_vs_avg
        self.vs = vs[int_depth_mask]
        self.vs_sd = vs_sd[int_depth_mask]
        self.depth = depth[int_depth_mask]
        self.eff_stress = None if eff_stress is None else eff_stress[int_depth_mask]

        self.info = {
            "z_min": min(depth),
            "z_max": max(depth),
            "z_spread": max(depth) - min(depth),
            "removed_rows": np.where(int_depth_mask == False)[0].tolist(),
        }

        # VsZ and Vs30 info init for lazy loading
        self._vsz = None
        self._vs30 = None
        self._vs30_sd = None

    @staticmethod
    def from_byte_stream(file_name: str, name: str, layered: bool, stream: bytes, vs30_correlation: Optional[str] = None):
        """
        Creates a VsProfile from a file stream

        Parameters
        ----------
        file_name : str
            The name of the file
        name : str
            The name of the VsProfile
        layered : bool
            Whether the VsProfile is layered
        stream : bytes
            The file stream
        vs30_correlation : str
            The Vs30 correlation to use
        """
        file_name = Path(file_name)
        file_data = (
            pd.read_csv(BytesIO(stream))
            if file_name.suffix == ".csv"
            else pd.read_excel(BytesIO(stream))
        )
        return VsProfile(
            name,
            np.asarray(file_data["Vs"]),
            np.asarray(file_data["Vs_SD"]),
            np.asarray(file_data["Depth"]),
            None,
            None,
            vs30_correlation,
            layered
        )

    @staticmethod
    def from_cpt(cpt: CPT, cpt_correlation: str, vs30_correlation: Optional[str] = None):
        """
        Creates a VsProfile from a CPT and correlation

        Parameters
        ----------
        cpt : CPT
            The CPT to use
        cpt_correlation : str
            The correlation to use for CPT to Vs
        vs30_correlation : str
            The correlation to use for Vs30
        """
        # Check Correlation string
        if cpt_correlation not in cpt_vs_correlations.CPT_CORRELATIONS.keys():
            raise KeyError(
                f"{cpt_correlation} not found in set of correlations {cpt_vs_correlations.CPT_CORRELATIONS.keys()}"
            )
        vs, vs_sd = cpt_vs_correlations.CPT_CORRELATIONS[cpt_correlation](cpt)
        return VsProfile(
            cpt.name, vs.squeeze(), vs_sd.squeeze(), cpt.depth, cpt.effStress.squeeze(), cpt_correlation, vs30_correlation
        )

    @staticmethod
    def from_spt(spt: SPT, spt_correlation: str, vs30_correlation: Optional[str] = None):
        """
        Creates a VsProfile from an SPT and correlation

        Parameters
        ----------
        spt : SPT
            The SPT to use
        spt_correlation : str
            The correlation to use for SPT to Vs
        vs30_correlation : str
            The correlation to use for Vs30
        """
        # Check Correlation string
        if spt_correlation not in spt_vs_correlations.SPT_CORRELATIONS.keys():
            raise KeyError(
                f"{spt_correlation} not found in set of correlations {spt_vs_correlations.SPT_CORRELATIONS.keys()}"
            )
        vs, vs_sd, depth, eff_stress = spt_vs_correlations.SPT_CORRELATIONS[spt_correlation](spt)
        return VsProfile(
            spt.name, vs.squeeze(), vs_sd.squeeze(), depth, eff_stress.squeeze(), spt_correlation, vs30_correlation
        )

    @staticmethod
    def from_json(json: Dict):
        """
        Creates a VsProfile from a json dictionary string
        """
        return VsProfile(
            json["name"],
            np.asarray(json["vs"]),
            np.asarray(json["vs_sd"]),
            np.asarray(json["depth"]),
            None,
            None if json["vs_correlation"] == "" else json["vs_correlation"],
            None if json["vs30_correlation"] == "" else json["vs30_correlation"],
            json["layered"] == "True",
        )

    def to_json(self):
        """
        Creates a json response dictionary from the VsProfile
        """
        return {
            "name": self.name,
            "vs_correlation": self.vs_correlation,
            "vs30_correlation": self.vs30_correlation,
            "layered": str(self.layered),
            "max_depth": self.max_depth,
            "vs": self.vs.tolist(),
            "vs_sd": self.vs_sd.tolist(),
            "depth": self.depth.tolist(),
            "info": self.info,
            "vsz": self._vsz,
            "vs30": self._vs30,
            "vs30_sd": self._vs30_sd,
        }

    def to_dataframe(self):
        """
        Turns the VsProfile into a dataframe for download
        """
        return pd.DataFrame(np.column_stack([self.depth, self.vs, self.vs_sd]), columns=["Depth", "Vs", "Vs_SD"])

    @property
    def vsz(self):
        """
        Gets the VsZ value and computes the value if not set
        """
        if self._vsz is None:
            self._vsz = self.calc_vsz()
        return self._vsz

    @property
    def vs30(self):
        """
        Gets the Vs30 value and computes the value if not set
        Will grab value from VsZ if max depth is 30m already
        then no conversion is needed
        """
        if self._vs30 is None:
            self._vs30, self._vs30_sd = self.calc_vs30()
        return self._vs30

    @property
    def vs30_sd(self):
        """
        Gets the Vs30 Standard Deviation value and computes the value if not set
        Will grab value from VsZ if max depth is 30m already
        then no conversion is needed
        """
        if self._vs30_sd is None:
            self._vs30, self._vs30_sd = self.calc_vs30()
        return self._vs30_sd

    def calc_vsz(self):
        """
        Calculates the average Vs at the max Z depth for the given VsProfile
        """
        vs_midpoint, depth_midpoint = utils.convert_to_midpoint(self.vs, self.depth)
        time = 0
        for ix in range(1, len(vs_midpoint), 2):
            change_in_z = depth_midpoint[ix] - depth_midpoint[ix - 1]
            time += change_in_z / vs_midpoint[ix]
        return self.max_depth / time

    def calc_vs30(self):
        """
        Calculates Vs30 and Vs30_sd for the given VsProfile based on VsZ and max depth
        Uses vs30 correlations if the max depth is less than 30m
        """
        if self.max_depth == 30:
            # Set Vs30 to VsZ as Z is 30
            vs30, vs30_sd = self.vsz, 0
        else:
            if self.vs30_correlation not in vs30_correlations.VS30_CORRELATIONS.keys():
                raise KeyError(
                    f"{self.vs30_correlation} not found in set of correlations {vs30_correlations.VS30_CORRELATIONS.keys()}"
                )
            vs30, vs30_sd = vs30_correlations.VS30_CORRELATIONS[self.vs30_correlation](self)
        return vs30, vs30_sd
