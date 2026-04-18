import numpy as np
from dataclasses import dataclass

@dataclass
class PlantState:
    id: float
    iq: float
    Vdc: float

class HybridACDCPlant:
    def __init__(self, params):
        self.p = params
        self.w = 2*np.pi*self.p.f_grid
        self.Vg_peak = np.sqrt(2)*self.p.Vg_rms_phase

    def grid_voltage_dq(self, t):
        return self.Vg_peak, 0.0

    def step(self, x: PlantState, vconv_d, vconv_q, Pdc_load_w, Pren_avail_w, t):
        Ts = self.p.Ts
        Lf, Rf, Cdc = self.p.Lf, self.p.Rf, self.p.Cdc
        w = self.w

        vgd, vgq = self.grid_voltage_dq(t)

        did = (1.0/Lf)*(vconv_d - Rf*x.id - vgd + w*Lf*x.iq)
        diq = (1.0/Lf)*(vconv_q - Rf*x.iq - vgq - w*Lf*x.id)

        P_ac = 1.5*(vgd*x.id + vgq*x.iq)

        Vdc_safe = max(50.0, x.Vdc)
        i_dc_conv = P_ac / Vdc_safe

        i_src_th = (self.p.Vsrc_nom - x.Vdc)/self.p.Rsrc
        i_src_max = max(0.0, Pren_avail_w / Vdc_safe)
        i_src = np.clip(i_src_th, 0.0, i_src_max)

        i_load = Pdc_load_w / Vdc_safe

        dVdc = (1.0/Cdc)*(i_src - i_load - i_dc_conv)

        return PlantState(
            id = x.id + Ts*did,
            iq = x.iq + Ts*diq,
            Vdc = x.Vdc + Ts*dVdc
        ), dict(P_ac=P_ac, vgd=vgd, vgq=vgq)
