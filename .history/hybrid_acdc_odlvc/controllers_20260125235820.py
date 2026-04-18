import numpy as np

class PI:
    def __init__(self, Kp, Ki, Ts, umin=None, umax=None):
        self.Kp, self.Ki, self.Ts = float(Kp), float(Ki), float(Ts)
        self.umin, self.umax = umin, umax
        self.xi = 0.0

    def reset(self):
        self.xi = 0.0

    def __call__(self, e):
        self.xi += self.Ki * self.Ts * float(e)
        u = self.Kp * float(e) + self.xi
        if (self.umin is not None) and (u < self.umin):
            u = self.umin
        if (self.umax is not None) and (u > self.umax):
            u = self.umax
        return u


class CurrentControllerPI:
    def __init__(self, plant_params, ctrl_params):
        self.p = plant_params
        self.c = ctrl_params
        Ts = self.p.Ts
        self.pi_d = PI(self.c.Kp_i, self.c.Ki_i, Ts)
        self.pi_q = PI(self.c.Kp_i, self.c.Ki_i, Ts)

    def reset(self):
        self.pi_d.reset()
        self.pi_q.reset()

    def voltage_cmd(self, id_ref, iq_ref, id_meas, iq_meas, vgd, vgq):
        w = 2.0 * np.pi * self.p.f_grid
        Lf, Rf = self.p.Lf, self.p.Rf

        ed = float(id_ref - id_meas)
        eq = float(iq_ref - iq_meas)

        v_pi_d = self.pi_d(ed)
        v_pi_q = self.pi_q(eq)

        # decoupling + grid-voltage feedforward
        vdec_d = vgd - w * Lf * iq_meas + Rf * id_meas
        vdec_q = vgq + w * Lf * id_meas + Rf * iq_meas

        return v_pi_d + vdec_d, v_pi_q + vdec_q, dict(ed=ed, eq=eq)


class ODLVC:
    """
    Online Disturbance Learning Vector Control (ODL-VC):
    - Inner dq PI vector control (with decoupling)
    - Lightweight NN estimates additive disturbance terms in vd/vq channels
    - NN contribution is ATTENUATED during large Vdc transients to prevent overshoot
    """
    def __init__(self, plant_params, ctrl_params):
        self.p = plant_params
        self.c = ctrl_params
        self.inner = CurrentControllerPI(plant_params, ctrl_params)

        # NN: inputs [id, iq, ed, eq] -> outputs [d_hat_d, d_hat_q]
        self.n_in = 4
        self.n_h = int(self.c.nn_hidden)

        rng = np.random.default_rng(2)
        self.W1 = 0.15 * rng.standard_normal((self.n_h, self.n_in))
        self.b1 = np.zeros((self.n_h,), dtype=float)

        # Start last layer at zero => no initial NN action
        self.W2 = np.zeros((2, self.n_h), dtype=float)
        self.b2 = np.zeros((2,), dtype=float)

        self.lr = float(self.c.nn_lr)
        self.w2_max = float(self.c.nn_w2_max)

        # Attenuation settings (you can tune these)
        self.vdc_fade_v = 3.0   # fade NN to 0 when |Vdc - Vref| >= 3 V
        self.vdc_dead_v = 0.5   # keep full NN when |Vdc - Vref| <= 0.5 V

    def reset(self):
        self.inner.reset()
        self.W2[:] = 0.0
        self.b2[:] = 0.0

    def _forward(self, x):
        z = self.W1 @ x + self.b1
        h = np.tanh(z)
        y = self.W2 @ h + self.b2
        return y, h

    def _project(self):
        np.clip(self.W2, -self.w2_max, self.w2_max, out=self.W2)

    def _vdc_alpha(self, Vdc):
        """
        Smooth NN attenuation factor alpha in [0,1]:
        - alpha=1 near Vref (within dead-band)
        - alpha -> 0 as |Vdc - Vref| approaches fade threshold
        """
        e_v = abs(float(Vdc) - float(self.c.Vdc_ref))

        # Full NN inside dead-band
        if e_v <= self.vdc_dead_v:
            return 1.0

        # Fully off beyond fade threshold
        if e_v >= self.vdc_fade_v:
            return 0.0

        # Linear fade between dead-band and fade threshold
        alpha = 1.0 - (e_v - self.vdc_dead_v) / (self.vdc_fade_v - self.vdc_dead_v + 1e-12)
        return float(np.clip(alpha, 0.0, 1.0))

    def voltage_cmd(self, id_ref, iq_ref, id_meas, iq_meas, vgd, vgq, Vdc):
        """
        NOTE: Signature changed (added Vdc).
        In sim.py you must call:
            inner.voltage_cmd(..., x.Vdc)
        """
        vcmd_d, vcmd_q, dbg = self.inner.voltage_cmd(id_ref, iq_ref, id_meas, iq_meas, vgd, vgq)

        x = np.array([id_meas, iq_meas, dbg["ed"], dbg["eq"]], dtype=float)
        d_hat, h = self._forward(x)

        # ---- Key fix: attenuate NN during large Vdc transients ----
        alpha = self._vdc_alpha(Vdc)

        vcmd_d2 = vcmd_d + alpha * d_hat[0]
        vcmd_q2 = vcmd_q + alpha * d_hat[1]

        # Online update (also scale learning during large transients)
        # This prevents weight bursts exactly when Vdc error is large.
        e = np.array([dbg["ed"], dbg["eq"]], dtype=float)
        lr_eff = self.lr * alpha

        self.W2 += lr_eff * (e.reshape(2, 1) @ h.reshape(1, self.n_h))
        self.b2 += lr_eff * 0.1 * e
        self._project()

        dbg.update({
            "d_hat_d": float(d_hat[0]),
            "d_hat_q": float(d_hat[1]),
            "alpha_nn": float(alpha),
        })
        return vcmd_d2, vcmd_q2, dbg
