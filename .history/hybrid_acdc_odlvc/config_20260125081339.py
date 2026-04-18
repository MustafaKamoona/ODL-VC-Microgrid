from dataclasses import dataclass

@dataclass
class PlantParams:
    f_grid: float = 50.0
    Vg_rms_phase: float = 230.0
    Lf: float = 2.5e-3
    Rf: float = 0.08
    Cdc: float = 12.0e-3          # increased
    Vsrc_nom: float = 680.0
    Rsrc: float = 0.4             # stronger source
    m_max: float = 0.95
    Ts: float = 1e-4              # 10 kHz

@dataclass
class ControlParams:
    Vdc_ref: float = 680.0
    Vdc_min: float = 676.0        # ±4 V
    Vdc_max: float = 684.0

    Kp_v: float = 0.9
    Ki_v: float = 120.0
    id_min: float = -120.0
    id_max: float = 120.0

    Kp_i: float = 7.0
    Ki_i: float = 1600.0

    nn_hidden: int = 12
    nn_lr: float = 3.0e-3
    nn_w2_max: float = 60.0

@dataclass
class ProfileParams:
    compress_24h_to_s: float = 24.0
    P_ren_max_w: float = 30000.0
