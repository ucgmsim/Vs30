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

    @property
    def qt(self):
        """
        Gets the qt value and computes the value if not set
        """
        if self._qt is None:
            self._qt = self.Qc - self.u * (1 - self.net_area_ratio)
        return self._qt

#    @property
#    def Ic(self):
#        """
#        Gets the Ic value and computes the value if not set
#        """
#        if self._Ic is None:
#            # atmospheric pressure (MPa)
#            pa = 0.1
#            # compute non-normalised Ic based on the correlation by Robertson (2010).
#            Rf = (self.Fs / self.Qc) * 100
#            self._Ic = (
#                (3.47 - np.log10(self.Qc / pa)) ** 2 + (np.log10(Rf) + 1.22) ** 2
#            ) ** 0.5
#        return self._Ic

    @property
    def Qtn(self):
        """
        Gets the Qtn value and computes the value if not set
        """
        if self._Qtn is None:
            self._Qtn, self._effStress, self._Ic, self._n = self.calc_cpt_params()
        return self._Qtn

    @property
    def effStress(self):
        """
        Gets the effStress value and computes the value if not set
        """
        if self._effStress is None:
            self._Qtn, self._effStress, self._Ic, self._n = self.calc_cpt_params()
        return self._effStress

    @property
    def Ic(self):
        """
        Gets the Ic value and computes the value if not set
        """
        if self._effStress is None:
            self._Qtn, self._effStress, self._Ic, self._n = self.calc_cpt_params()
        return self._Ic

    @property
    def n(self):
        """
        Gets the Ic value and computes the value if not set
        """
        if self._effStress is None:
            self._Qtn, self._effStress, self._Ic, self._n = self.calc_cpt_params()
        return self._n
                
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
        return gamma

    def calc_cpt_params(self):
        """Compute and save Qtn and effStress CPT parameters"""
        # atmospheric pressure (MPa)
        pa = 0.1
        # compute vertical stress profile
        totalStress = np.zeros(len(self.depth))
        u0 = np.zeros(len(self.depth))
        for i in range(1, len(self.depth)):
            totalStress[i] = (
                self.gamma[i] * (self.depth[i] - self.depth[i - 1]) + totalStress[i - 1]
            )
            if self.depth[i] >= self.ground_water_level:
                u0[i] = 0.00981 * (self.depth[i] - self.depth[i - 1]) + u0[i - 1]
        effStress = totalStress - u0
        effStress[0] = effStress[1]  # fix error caused by dividing 0

        Fr = (self.Fs / (self.qt - totalStress)) * 100
        Qtn = np.zeros(len(self.depth))
        Ic = np.zeros(len(self.depth))
        deltan = 1.0 * np.ones(len(self.depth))
#        print(deltan)
        n = 0.1 * np.ones(len(self.depth)) #assumed initial value of n
#        n = 0.381 * self.Ic + 0.05 * (effStress / pa) - 0.15
        for i in range(0, len(n)): #loop over each depth point
            # print(i)

            wait_x = 0
            plt_qtn = []
            plt_ic = []
            plt_n = []
            plt_change_n = []

            while deltan[i] >= 0.01: #iterate Qtn, Ic and n until convergence
                wait_x += 1

                n0 = n[i]
#                print(n0)
                Qtn[i] = ((self.qt[i] - totalStress[i]) / pa) * (pa / effStress[i]) ** n[i]
#                print(Qtn[i])
                Ic[i] = (
                    (3.47 - np.log10(Qtn[i])) ** 2 + (np.log10(Fr[i]) + 1.22) ** 2
                ) ** 0.5
#                print(Ic[i])
                n[i] = 0.381 * Ic[i] + 0.05 * (effStress[i] / pa) - 0.15
#                print(n[i])
                if n[i] > 1:
                    n[i] = 1
                deltan[i] = np.abs(n0 - n[i])
                # print(deltan[i])

                plt_qtn.append(Qtn[i])
                plt_ic.append(Ic[i])
                plt_n.append(n0)
                plt_change_n.append(deltan[i])

                if wait_x > 150:
                    print(f"Depth {self.depth[i]} is over")
                    deltan[i] = 0.009
                    from matplotlib import pyplot as plt

                    fig = plt.figure(figsize=(12, 9))

                    # Needed to add spacing between 1st and 2nd row
                    # Add a margin between the main title and sub-plots
                    fig.subplots_adjust(hspace=0.4, top=0.85)

                    # Add the main title
                    fig.suptitle(f"depth={self.depth[i]}\nqt={self.qt[i]}\ntotalStress={totalStress[i]}\neffStress={effStress[i]}\nFr={Fr[i]}", fontsize=10)

                    # Add the subplots
                    ax1 = fig.add_subplot(2, 2, 1)
                    ax2 = fig.add_subplot(2, 2, 2)
                    ax3 = fig.add_subplot(2, 2, 3)
                    ax4 = fig.add_subplot(2, 2, 4)

                    # Add the text for each subplot
                    ax1.title.set_text("Qtn")
                    ax1.plot(plt_qtn)
                    ax2.title.set_text("Ic")
                    ax2.plot(plt_ic)
                    ax3.title.set_text("n")
                    ax3.plot(plt_n)
                    ax4.title.set_text("Delta n")
                    ax4.plot(plt_change_n)
                    plt.show()
                    # plt.savefig(f"/home/joel/local/Vs/calc_n_{self.depth[i]}.png")

                    # plt.figure(figsize=(12, 3))
                    # plt.subplot(141)
                    # plt.plot(plt_qtn)
                    # plt.subplot(142)
                    # plt.plot(plt_ic)
                    # plt.subplot(143)
                    # plt.plot(plt_n)
                    # plt.subplot(144, label="Delta N")
                    # plt.plot(plt_change_n)
                    # plt.suptitle('Categorical Plotting')
                    # plt.show()
                    print(1)
                
#        Qtn = ((self.qt - totalStress) / pa) * (pa / effStress) ** n
#        print(n)
        return Qtn, effStress, Ic, n
        
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
    def from_file(cpt_ffp: str, gwl, a):
        """
        Creates a CPT from a CPT file
        """
        cpt_ffp = Path(cpt_ffp)
        data = np.loadtxt(cpt_ffp, dtype=float, delimiter=",", skiprows=1)
        depth, qc, fs, u, info = CPT.process_cpt(data)
        return CPT(cpt_ffp.stem, depth, qc, fs, u, info, ground_water_level=gwl, net_area_ratio=a)

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
        # below_30_filter = np.all(data[:, [0]] <= 30, axis=1)
        below_30_filter = True
        info["Removed rows"] = np.where(below_30_filter == False)[0]
        # data = data[below_30_filter.T]  # z is less than 30 m
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

        # z_raw = data[:, 0]  # m
        # qc_raw = data[:, 1]  # MPa
        # fs_raw = data[:, 2]  # MPa
        # u_raw = data[:, 3]  # Mpa

        z = data[:, 0]  # m
        qc = data[:, 1]  # MPa
        fs = data[:, 2]  # MPa
        u = data[:, 3]  # Mpa

        # downsize = np.arange(z_raw[0], 30.02, 0.02)
        # z = np.array([])
        # qc = np.array([])
        # fs = np.array([])
        # u = np.array([])
        # for j in range(len(downsize)):
        #     for i in range(len(z_raw)):
        #         if abs(z_raw[i] - downsize[j]) < 0.001:
        #             z = np.append(z, z_raw[i])
        #             qc = np.append(qc, qc_raw[i])
        #             fs = np.append(fs, fs_raw[i])
        #             u = np.append(u, u_raw[i])

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
