import numpy as np

def default_profiles_24h():
    # Renewable availability (kW): designed to exceed total load during midday for export
    ren_kw = [0,0,0,0,3,6,10,14,18,22,26,28,30,30,28,26,22,16,10,6,3,0,0,0]
    dc1_kw = [3.5,3.8,4.1,4.3,4.8,5.2,5.6,6.0,5.5,5.1,4.9,4.8,4.7,4.6,4.6,4.7,5.0,5.4,5.8,6.1,5.6,5.0,4.4,4.0]
    dc2_kw = [2.5,2.6,2.7,3.0,3.4,3.6,3.8,4.0,3.7,3.5,3.4,3.3,3.2,3.2,3.2,3.3,3.5,3.8,4.1,4.2,4.0,3.6,3.1,2.8]
    ac_kw  = [3.0,3.1,3.2,3.3,3.5,13.8,14.2,14.6,15.0,15.4,15.8,16.2,16.0,15.8,15.5,15.2,14.8,14.4,14.0,13.7,3.4,3.2,3.1,3.0]
    return dict(ren_kw=ren_kw, dc1_kw=dc1_kw, dc2_kw=dc2_kw, ac_kw=ac_kw)

def _interp_24h_to_time(t_s, compress_24h_to_s, y24):
    y24 = np.asarray(y24, dtype=float)
    hour = (t_s / compress_24h_to_s) * 24.0
    hour = np.clip(hour, 0.0, 23.999999)
    i0 = int(np.floor(hour))
    i1 = (i0 + 1) % 24
    a = hour - i0
    return (1-a)*y24[i0] + a*y24[i1]

def profiles_at_time(t_s, prof24, profile_params):
    c = profile_params.compress_24h_to_s
    ren = _interp_24h_to_time(t_s, c, prof24["ren_kw"]) * 1000.0
    dc1 = _interp_24h_to_time(t_s, c, prof24["dc1_kw"]) * 1000.0
    dc2 = _interp_24h_to_time(t_s, c, prof24["dc2_kw"]) * 1000.0
    ac  = _interp_24h_to_time(t_s, c, prof24["ac_kw"])  * 1000.0
    return ren, dc1, dc2, ac

def net_grid_power_reference(t_s, prof24, profile_params):
    ren, dc1, dc2, ac = profiles_at_time(t_s, prof24, profile_params)
    ren = min(ren, profile_params.P_ren_max_w)
    P_load = (dc1 + dc2 + ac)
    return ren - P_load, ren, dc1, dc2, ac

def unbalance_envelope(t_s, compress_24h_to_s):
    hour = (t_s / compress_24h_to_s) * 24.0
    if 12.0 <= hour <= 15.0:
        return 0.25
    return 0.0
