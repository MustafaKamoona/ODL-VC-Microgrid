import numpy as np
from .models import PlantState, HybridACDCPlant
from .controllers import CurrentControllerPI, ODLVC, PI
from .profiles import net_grid_power_reference, unbalance_envelope

def sat_vconv(vd, vq, Vdc, m_max):
    vmax = m_max * max(50.0, Vdc) / 2.0
    mag = np.sqrt(vd*vd + vq*vq)
    if mag <= vmax:
        return vd, vq, 1.0
    s = vmax / (mag + 1e-12)
    return vd*s, vq*s, s

def run_sim_24h(plant_params, ctrl_params, profile_params, prof24, controller_kind, t_end_s):
    plant = HybridACDCPlant(plant_params)
    inner = CurrentControllerPI(plant_params, ctrl_params) if controller_kind=="baseline_pi" else ODLVC(plant_params, ctrl_params)
    outer = PI(ctrl_params.Kp_v, ctrl_params.Ki_v, plant_params.Ts, umin=ctrl_params.id_min, umax=ctrl_params.id_max)

    Ts = plant_params.Ts
    n = int(np.floor(t_end_s/Ts)) + 1
    t = np.linspace(0, t_end_s, n)

    x = PlantState(id=0.0, iq=0.0, Vdc=679.0)
    log = {k: np.zeros(n) for k in [
        "t","Vdc","id","iq","id_ref","iq_ref","P_ren","Pdc1","Pdc2","Pac_load",
        "P_ref_net","P_ac","P_grid_net","i_a","i_b","i_c","ub"
    ]}
    if controller_kind=="odl_vc":
        log["d_hat_d"]=np.zeros(n); log["d_hat_q"]=np.zeros(n)

    Vg_peak = plant.Vg_peak

    for k, tk in enumerate(t):
        P_ref_net, Pren, Pdc1, Pdc2, PacL = net_grid_power_reference(tk, prof24, profile_params)
        Pdc_total = Pdc1 + Pdc2

        vgd, vgq = plant.grid_voltage_dq(tk)

        # desired net grid power -> id*
        id_from_pref = (2.0/3.0) * (P_ref_net / (Vg_peak + 1e-12))

        # hard constraint shaping
        Vdz = 1.0  # 1 V dead-zon
        
        if x.Vdc < (ctrl_params.Vdc_min - Vdz):
            id_from_pref -= 15.0*(ctrl_params.Vdc_min - x.Vdc)
        elif x.Vdc > (ctrl_params.Vdc_max + Vdz):
            id_from_pref += 15.0*(x.Vdc - ctrl_params.Vdc_max)
   
           # Outer Vdc regulation: if Vdc > Vref -> increase export (id positive). Hence use e = (Vdc - Vref)
        delta_id = outer(x.Vdc - ctrl_params.Vdc_ref)
        id_ref = id_from_pref + delta_id
        iq_ref = 0.0

        vcmd_d, vcmd_q, dbg = inner.voltage_cmd(id_ref, iq_ref, x.id, x.iq, vgd, vgq)
        vconv_d, vconv_q, s = sat_vconv(vcmd_d, vcmd_q, x.Vdc, plant_params.m_max)

        x, meas = plant.step(x, vconv_d, vconv_q, Pdc_total, Pren, tk)

        # abc currents for plot
        theta = 2*np.pi*plant_params.f_grid*tk
        ca, sa = np.cos(theta), np.sin(theta)
        i_alpha = x.id*ca - x.iq*sa
        i_beta  = x.id*sa + x.iq*ca
        i_a = i_alpha
        i_b = -0.5*i_alpha + (np.sqrt(3)/2)*i_beta
        i_c = -0.5*i_alpha - (np.sqrt(3)/2)*i_beta

        ub = float(unbalance_envelope(tk, profile_params.compress_24h_to_s))
        if ub>0:
            i_a *= (1.0 + 0.8*ub); i_b *= (1.0 - 0.5*ub); i_c *= (1.0 - 0.3*ub)

        P_grid_net = meas["P_ac"] - PacL

        log["t"][k]=tk; log["Vdc"][k]=x.Vdc; log["id"][k]=x.id; log["iq"][k]=x.iq
        log["id_ref"][k]=id_ref; log["iq_ref"][k]=iq_ref
        log["P_ren"][k]=Pren; log["Pdc1"][k]=Pdc1; log["Pdc2"][k]=Pdc2; log["Pac_load"][k]=PacL
        log["P_ref_net"][k]=P_ref_net; log["P_ac"][k]=meas["P_ac"]; log["P_grid_net"][k]=P_grid_net
        log["i_a"][k]=i_a; log["i_b"][k]=i_b; log["i_c"][k]=i_c; log["ub"][k]=ub

        if controller_kind=="odl_vc":
            log["d_hat_d"][k]=dbg.get("d_hat_d",0.0); log["d_hat_q"][k]=dbg.get("d_hat_q",0.0)

    return log
