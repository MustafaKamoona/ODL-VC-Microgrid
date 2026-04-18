import numpy as np

class PI:
    def __init__(self, Kp, Ki, Ts, umin=None, umax=None):
        self.Kp, self.Ki, self.Ts = Kp, Ki, Ts
        self.umin, self.umax = umin, umax
        self.xi = 0.0

    def reset(self):
        self.xi = 0.0

    def __call__(self, e):
        self.xi += self.Ki * self.Ts * e
        u = self.Kp * e + self.xi
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
        w = 2*np.pi*self.p.f_grid
        Lf, Rf = self.p.Lf, self.p.Rf

        ed = id_ref - id_meas
        eq = iq_ref - iq_meas

        v_pi_d = self.pi_d(ed)
        v_pi_q = self.pi_q(eq)

        vdec_d = vgd - w*Lf*iq_meas + Rf*id_meas
        vdec_q = vgq + w*Lf*id_meas + Rf*iq_meas

        return v_pi_d + vdec_d, v_pi_q + vdec_q, dict(ed=ed, eq=eq)

class ODLVC:
    def __init__(self, plant_params, ctrl_params):
        self.p = plant_params
        self.c = ctrl_params
        self.inner = CurrentControllerPI(plant_params, ctrl_params)

        self.n_in = 4
        self.n_h = int(self.c.nn_hidden)
        rng = np.random.default_rng(2)
        self.W1 = 0.15*rng.standard_normal((self.n_h, self.n_in))
        self.b1 = np.zeros((self.n_h,))
        self.W2 = np.zeros((2, self.n_h))
        self.b2 = np.zeros((2,))
        self.lr = float(self.c.nn_lr)
        self.w2_max = float(self.c.nn_w2_max)

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

    def voltage_cmd(self, id_ref, iq_ref, id_meas, iq_meas, vgd, vgq):
        vcmd_d, vcmd_q, dbg = self.inner.voltage_cmd(id_ref, iq_ref, id_meas, iq_meas, vgd, vgq)

        x = np.array([id_meas, iq_meas, dbg["ed"], dbg["eq"]], dtype=float)
        d_hat, h = self._forward(x)

        vcmd_d2 = vcmd_d + d_hat[0]
        vcmd_q2 = vcmd_q + d_hat[1]

        e = np.array([dbg["ed"], dbg["eq"]], dtype=float)
        self.W2 += self.lr * (e.reshape(2,1) @ h.reshape(1,self.n_h))
        self.b2 += self.lr * 0.1 * e
        self._project()

        dbg.update(dict(d_hat_d=float(d_hat[0]), d_hat_q=float(d_hat[1])))
        return vcmd_d2, vcmd_q2, dbg
