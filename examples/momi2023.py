# %% md
# Reproduce the MoMi 2023 results from the paper:
# 
# - Momi D, Wang Z, Griffiths JD. 2023. TMS-EEG evoked responses are driven by recurrent
#   large-scale network dynamics. eLife2023;12:e83232 DOI: https://doi.org/10.7554/eLife.83232
# %% md
# Create a folder named "reproduce_Momi_et_al_2022" in the same directory as this notebook.
# 
# 
# 1. Download the required data folder at:
# 
# https://drive.google.com/drive/folders/1iwsxrmu_rnDCvKNYDwTskkCNt709MPuF
# 
# and move the data inside ``reproduce_Momi_et_al_2022`` directory.
# 
# 2. Download the fsaverage folder at:
# 
# https://drive.google.com/drive/folders/1YPyf3h9YKnZi0zRwBwolROQuqDtxEfzF?usp=sharing
# 
# and move the `fsaverage` folders inside ``reproduce_Momi_et_al_2022`` directory.
# 
# 
# 3. Please download the individual leadfield matrix at:
# 
# https://drive.google.com/drive/folders/11jOvrzupbm2W8QdKTa9gPeG2Ivk4BdMN?usp=sharing
# 
# and replate the path in the ``reproduce_Momi_et_al_2022/leadfield_from_mne`` directory.
# %%
from typing import Union, Callable

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import cosine_similarity

import brainstate
from brainmass.param import Parameter
from brainstate import maybe_state


class JansenRitOutput:
    def __init__(self):
        self.M = []
        self.E = []
        self.I = []
        self.Mv = []
        self.Ev = []
        self.Iv = []
        self.eeg = []
        self.loss = []
        self.leadfield = []
        self.sc = []

    def finalize(self):
        self.M = jnp.stack(self.M, axis=0)
        self.E = jnp.stack(self.E, axis=0)
        self.I = jnp.stack(self.I, axis=0)
        self.Mv = jnp.stack(self.Mv, axis=0)
        self.Ev = jnp.stack(self.Ev, axis=0)
        self.Iv = jnp.stack(self.Iv, axis=0)
        self.eeg = jnp.stack(self.eeg, axis=0)


class Scale:
    def __init__(self, slope: float, fn: Callable = jnp.tanh):
        self.slope = slope
        self.fn = fn

    def __call__(self, x):
        x, unit = u.split_mantissa_unit(x)
        return u.maybe_decimal(self.slope * self.fn(x / self.slope) * unit)


class JansenRitNetwork(brainstate.nn.Module):
    r"""
    Network with Jansen-Rit neural mass models.

    This model is a modified version of the original Jansen-Rit model, incorporating additional
    parameters and dynamics to better capture the behavior of neural populations. The modifications
    include the addition of parameters such as `mu`, `k`, `y0`, `cy0`, and `ki`, which influence
    the dynamics of the system.

    The equations governing the modified Jansen-Rit model are as follows:

    $$
    \begin{aligned}
    & \dot{y}_{0}=y_1,\\
    & \dot{y_2}=y_3,\\
    & \dot{y_4}=y_5,\\
    &\dot{y}_{1}=A_eb_eS(I_p+a_2y_2-a_4y_4)-2b_ey_1-b_e^2y_0,\\
    &\dot{y}_{3}=A_eb_eS(a_1y_0)-2b_ey_3-b_e^2y_2,\\
    &\ddot{y}_{5}=A_ib_iS(I_i+a_3y_0)-2b_iy_5-b_i^2y_4 + \mu k S(y0 - cy0) - ki y5.
    \end{aligned}
    $$

    The sigmoid function $S(v)$ remains unchanged from the original Jansen-Rit model:

    $$
    S(v)=S_{\max } \cdot \frac{1}{1+e^{-r\left(v-v_0\right)}}
    $$

    The additional parameters introduced in this modified model are defined as follows:

    - `mu`: A scaling factor that modulates the influence of the additional term in the equation for $\ddot{y}_{5}$.
    - `k`: A parameter that scales the sigmoid function applied to the difference between `y0` and `cy0`.
    - `y0`: A reference potential that influences the dynamics of the inhibitory population.
    - `cy0`: A constant that shifts the reference potential `y0`.
    - `ki`: A damping factor that affects the rate of change of `y5`.

    """

    def __init__(
        self,
        sc: np.ndarray,
        lm: np.ndarray,
        dist: np.ndarray,
        w_bb: np.ndarray,

        A: Union[brainstate.typing.ArrayLike, Callable] = 3.25,
        a: Union[brainstate.typing.ArrayLike, Callable] = 100.0,
        B: Union[brainstate.typing.ArrayLike, Callable] = 22.0,
        b: Union[brainstate.typing.ArrayLike, Callable] = 50.0,
        g: Union[brainstate.typing.ArrayLike, Callable] = 1000.0,
        c1: Union[brainstate.typing.ArrayLike, Callable] = 135.0,
        c2: Union[brainstate.typing.ArrayLike, Callable] = 135 * 0.8,
        c3: Union[brainstate.typing.ArrayLike, Callable] = 135 * 0.25,
        c4: Union[brainstate.typing.ArrayLike, Callable] = 135 * 0.25,
        std_in: Union[brainstate.typing.ArrayLike, Callable] = 100.0,
        vmax: Union[brainstate.typing.ArrayLike, Callable] = 5.0,
        v0: Union[brainstate.typing.ArrayLike, Callable] = 6.0,
        r: Union[brainstate.typing.ArrayLike, Callable] = 0.56,
        y0: Union[brainstate.typing.ArrayLike, Callable] = 2.0,
        mu: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        k: Union[brainstate.typing.ArrayLike, Callable] = 10.0,
        cy0: Union[brainstate.typing.ArrayLike, Callable] = 5.0,
        ki: Union[brainstate.typing.ArrayLike, Callable] = 1.0,

        # constants
        lb: float = 0.01,  # lower bound of local gains
        k_lb: float = 0.5,  # lower bound of coefficient of external inputs
        s2o_coef=0.0001,  # coefficient from states (source EEG) to EEG
        conduct_lb: float = 1.5,  # lower bound for conduct velocity
        u_2ndsys_ub: float = 500.,  # the bound of the input for second order system
        noise_std_lb: float = 150.0,  # lower bound of std of noise

        # flags
        fit_gains_flat: bool = False,
        fit_lfm_flat: bool = False,

        # initializers
        hidden_init: Callable = brainstate.init.Uniform(0., 5.0),
        delay_init: Callable = brainstate.init.Uniform(0., 5.0),
    ):
        super().__init__()

        self.hidden_size = sc.shape[0]
        self.output_size = lm.shape[0]
        assert sc.shape == dist.shape
        assert lm.shape[1] == dist.shape[0] == dist.shape[1]
        self.hidden_init = hidden_init
        self.delay_init = delay_init

        self.sc = sc  # [hidden_size, hidden_size]
        self.lm = lm  # [output_size, hidden_size]
        self.dist = dist  # [hidden_size, hidden_size]

        self.A = brainstate.init.param(A, self.hidden_size)
        self.a = brainstate.init.param(a, self.hidden_size)
        self.B = brainstate.init.param(B, self.hidden_size)
        self.b = brainstate.init.param(b, self.hidden_size)
        self.g = brainstate.init.param(g, self.hidden_size)
        self.c1 = brainstate.init.param(c1, self.hidden_size)
        self.c2 = brainstate.init.param(c2, self.hidden_size)
        self.c3 = brainstate.init.param(c3, self.hidden_size)
        self.c4 = brainstate.init.param(c4, self.hidden_size)
        self.std_in = brainstate.init.param(std_in, self.hidden_size)
        self.vmax = brainstate.init.param(vmax, self.hidden_size)
        self.v0 = brainstate.init.param(v0, self.hidden_size)
        self.r = brainstate.init.param(r, self.hidden_size)
        self.mu = brainstate.init.param(mu, self.hidden_size)
        self.k = brainstate.init.param(k, self.hidden_size)
        self.cy0 = brainstate.init.param(cy0, self.hidden_size)
        self.ki = brainstate.init.param(ki, self.hidden_size)

        self.y0 = brainstate.init.param(y0, self.output_size)

        self.lb = brainstate.init.param(lb, self.hidden_size)
        self.k_lb = brainstate.init.param(k_lb, self.hidden_size)
        self.s2o_coef = brainstate.init.param(s2o_coef, self.hidden_size)
        self.conduct_lb = brainstate.init.param(conduct_lb, self.hidden_size)
        self.u_2ndsys_ub = brainstate.init.param(u_2ndsys_ub, self.hidden_size)
        self.noise_std_lb = brainstate.init.param(noise_std_lb, self.hidden_size)

        self.fit_gains_flat = fit_gains_flat
        self.fit_lfm_flat = fit_lfm_flat
        self.w_bb = Parameter(w_bb) if fit_gains_flat else w_bb
        self.lm = Parameter(lm) if fit_lfm_flat else lm

    def init_state(self, batch_size=None, **kwargs):
        size = (self.hidden_size,) if batch_size is None else (batch_size, self.hidden_size)
        self.M = brainstate.HiddenState(brainstate.init.param(self.hidden_init, size))
        self.E = brainstate.HiddenState(brainstate.init.param(self.hidden_init, size))
        self.I = brainstate.HiddenState(brainstate.init.param(self.hidden_init, size))
        self.Mv = brainstate.HiddenState(brainstate.init.param(self.hidden_init, size))
        self.Ev = brainstate.HiddenState(brainstate.init.param(self.hidden_init, size))
        self.Iv = brainstate.HiddenState(brainstate.init.param(self.hidden_init, size))
        self.delay = brainstate.HiddenState(brainstate.init.param(self.delay_init, (500,) + size))

    def reset_state(self, batch_size=None, **kwargs):
        size = self.hidden_size if batch_size is None else (batch_size, *self.hidden_size)
        self.M.value = brainstate.init.param(self.hidden_init, size)
        self.E.value = brainstate.init.param(self.hidden_init, size)
        self.I.value = brainstate.init.param(self.hidden_init, size)
        self.Mv.value = brainstate.init.param(self.hidden_init, size)
        self.Ev.value = brainstate.init.param(self.hidden_init, size)
        self.Iv.value = brainstate.init.param(self.hidden_init, size)
        self.delay.value = brainstate.init.param(self.delay_init, (500,) + size)

    def sys2nd(self, A, a, u, x, v):
        return A * a * u - 2 * a * v - a ** 2 * x

    def S(self, v):
        return maybe_state(self.vmax) / (1 + jnp.exp(maybe_state(self.r) * (maybe_state(self.v0) - v)))

    def effective_sc(self):
        w = jnp.exp(maybe_state(self.w_bb)) * maybe_state(self.sc)
        w2 = u.math.log1p((w + w.T) / 2)
        w_n = w2 / u.math.linalg.norm(w2)
        return w_n

    def update(self, input):
        # input: [n_duration, n_time, n_input]

        dt = brainstate.environ.get_dt() / u.second
        relu = brainstate.functional.relu

        g = relu(maybe_state(self.g))
        std_in = relu(maybe_state(self.std_in))
        lb = maybe_state(self.lb)
        c1 = relu(maybe_state(self.c1))
        c2 = relu(maybe_state(self.c2))
        c3 = relu(maybe_state(self.c3))
        c4 = relu(maybe_state(self.c4))
        k = relu(maybe_state(self.k))
        B = relu(maybe_state(self.B))
        b = relu(maybe_state(self.b)) + 1.0
        A = relu(maybe_state(self.A))
        a = relu(maybe_state(self.a)) + 1.0
        mu = relu(maybe_state(self.mu))

        w_n = self.effective_sc()
        dg = -jnp.diag(jnp.sum(w_n, axis=1))

        scaleV = Scale(1e3)
        scaleI = Scale(self.u_2ndsys_ub)
        lm_t = self.lm - 1 / self.output_size * jnp.matmul(jnp.ones((1, self.output_size)), self.lm)
        delay_step = jnp.asarray(self.dist / (self.conduct_lb + mu), dtype=np.int32)

        def one_time(input_one_time):
            # delayed E
            Ed = u.math.gather(self.delay.value, 0, delay_step)
            # weights on delayed E
            LEd = jnp.sum(w_n * Ed, axis=1)

            # firing rate for Main population
            rM = self.S(self.E.value - self.I.value)
            # firing rate for Excitatory population
            noiseE = brainstate.random.randn(self.hidden_size)
            rE = ((self.noise_std_lb + std_in) * noiseE +
                  (lb + g) * (LEd + jnp.matmul(dg, self.E.value)) +
                  (lb + c2) * self.S((lb + c1) * self.M.value))
            # firing rate for Inhibitory population
            rI = (lb + c4) * self.S((lb + c3) * self.M.value)

            # Update the states by step-size.
            ddM = self.M.value + dt * self.Mv.value
            ddE = self.E.value + dt * self.Ev.value
            ddI = self.I.value + dt * self.Iv.value
            ddMv = self.Mv.value + dt * self.sys2nd(A, a, scaleI(rM), self.M.value, self.Mv.value)
            ccc = scaleI(rE) + (self.k_lb + k) * self.ki * input_one_time
            ddEv = self.Ev.value + dt * self.sys2nd(A, a, ccc, self.E.value, self.Ev.value)
            ddIv = self.Iv.value + dt * self.sys2nd(B, b, scaleI(rI), self.I.value, self.Iv.value)

            # Calculate the saturation for model states (for stability and gradient calculation).
            self.E.value = ddE
            self.I.value = ddI
            self.M.value = ddM
            self.Ev.value = ddEv
            self.Iv.value = ddIv
            self.Mv.value = ddMv
            # self.E.value = scaleV(ddE)
            # self.I.value = scaleV(ddI)
            # self.M.value = scaleV(ddM)
            # self.Ev.value = scaleV(ddEv)
            # self.Iv.value = scaleV(ddIv)
            # self.Mv.value = scaleV(ddMv)

            # update placeholders for E buffer
            self.delay.value = self.delay.value.at[0].set(self.E.value)

        def one_duration(input_one_batch):
            # input_one_batch: [n_time, n_input]
            brainstate.transform.for_loop(one_time, input_one_batch)
            self.delay.value = jnp.concatenate(
                (jnp.expand_dims(self.E.value, axis=0), self.delay.value[:-1]), axis=0
            )
            eeg = self.s2o_coef * self.cy0 * jnp.matmul(lm_t, self.E.value - self.I.value) - self.y0
            return (
                self.M.value, self.I.value, self.E.value,
                self.Mv.value, self.Ev.value, self.Iv.value,
                eeg
            )

        return brainstate.transform.for_loop(one_duration, input)


class ModelFitting:
    def __init__(
        self,
        model: JansenRitNetwork,
        optimizer: brainstate.optim.Optimizer,
        duration_per_batch: u.Quantity,
        time_per_duration: u.Quantity,
    ):
        self.model = model
        self.optimizer = optimizer
        self.weights = self.model.states(brainstate.ParamState)
        self.optimizer.register_trainable_weights(self.weights)
        self.cost = DistCost()

        dt = brainstate.environ.get_dt()
        self.n_duration_per_batch = int(duration_per_batch / time_per_duration)
        self.n_time_per_duration = int(time_per_duration / dt)

    @brainstate.transform.jit(static_argnums=0)
    def _batch_train(self, inputs, targets):
        # inputs: [n_duration, n_time_per_duration, n_input]
        # targets: [n_duration, n_time_per_duration, n_output]

        brainstate.nn.init_all_states(self.model)

        def f_loss():
            out = self.model(inputs)
            M_batch, I_batch, E_batch, Mv_batch, Ev_batch, Iv_batch, eeg_batch = out
            loss = 10. * self.cost(eeg_batch, targets)
            return loss, out

        grads, loss, out_batch = brainstate.transform.grad(
            f_loss, grad_states=self.weights, return_value=True, has_aux=True)()

        # jax.debug.print("Gradients = {g}", g=jax.tree.map(lambda x: jnp.max(jnp.abs(x)), grads))
        self.optimizer.update(grads)
        return loss, out_batch

    def train(
        self,
        data: np.ndarray,
        uuu: np.ndarray,
        n_epoch: int,
        epoch_min: int = 110,  # run minimum epoch # part of stop criteria
        r_lb: float = 0.85,  # lowest pearson correlation # part of stop criteria
    ) -> JansenRitOutput:
        # data: [n_data, n_time, n_out]
        # uuu: [n_data, n_in, n_time]
        duration = self.n_duration_per_batch
        time = self.n_time_per_duration

        # define masks for getting lower triangle matrix
        mask = np.tril_indices(self.model.hidden_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # placeholders for the history of model parameters
        output = JansenRitOutput()
        if self.model.fit_gains_flat:
            output.sc.append(self.model.effective_sc()[mask])  # sc weights history
        if self.model.fit_lfm_flat:
            output.leadfield.append(self.model.lm.value)

        num_durations = int(data.shape[0] / duration)
        for i_epoch in range(n_epoch):
            losses = []
            output_eeg = []
            for i_duration in range(num_durations):
                inputs = uuu[i_duration * duration:(i_duration + 1) * duration]
                targets = data[i_duration * duration:(i_duration + 1) * duration]
                loss, out_batch = self._batch_train(inputs, targets)
                output.M.append(out_batch[0])
                output.E.append(out_batch[1])
                output.I.append(out_batch[2])
                output.Mv.append(out_batch[3])
                output.Ev.append(out_batch[4])
                output.Iv.append(out_batch[5])
                output_eeg.append(out_batch[6])
                output.loss.append(loss)
                losses.append(loss)
                if self.model.fit_gains_flat:
                    output.sc.append(self.model.effective_sc()[mask])
                if self.model.fit_lfm_flat:
                    output.leadfield.append(self.model.lm.value)

            fc = np.corrcoef(data.T)
            ts_sim = np.concatenate(output_eeg, axis=0)
            fc_sim = np.corrcoef(ts_sim[10:].T)
            corr = np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1]
            cos_sim = np.diag(cosine_similarity(ts_sim.T, data.T)).mean()

            print(
                f'epoch {i_epoch}, '
                f'loss = {np.mean(losses)}, '
                f'pearson correlation = {corr}, '
                f'cosine similarity = {cos_sim}'
            )

            if i_epoch > epoch_min and corr > r_lb:
                break
        return output

    @brainstate.transform.jit(static_argnums=0)
    def _batch_predict(self, inputs):
        # inputs: [n_duration, n_time, n_input]
        brainstate.nn.init_all_states(self.model)
        out_batch = self.model(inputs)
        # M_batch, I_batch, E_batch, Mv_batch, Ev_batch, Iv_batch, eeg_batch = out
        return out_batch

    # def test(
    #     self,
    #     data: np.ndarray,
    #     uuu: np.ndarray,
    # ) -> JansenRitOutput:
    #     # data: [n_data, n_out, n_time]
    #     # uuu: [n_data, n_in, n_time]
    #
    #     duration = self.n_duration_per_batch
    #     time = self.n_time_per_duration
    #     num_batches = int(data.shape[0] / duration)
    #
    #     output = JansenRitOutput()
    #     u_hat = np.zeros(
    #         (duration, time, base_batch_num * batch_size + self.ts.shape[2]))
    #     u_hat[:, :, base_batch_num * batch_size:] = uuu
    #
    #     # Perform the training in batches.
    #     for i_batch in range(num_batches):
    #         inputs = u_hat[i_batch * duration:(i_batch + 1) * duration]
    #         out_batch = self._batch_predict(inputs)
    #         output.M.append(out_batch[0])
    #         output.E.append(out_batch[1])
    #         output.I.append(out_batch[2])
    #         output.Mv.append(out_batch[3])
    #         output.Ev.append(out_batch[4])
    #         output.Iv.append(out_batch[5])
    #         output.eeg.append(out_batch[6])
    #     return output


class DistCost:
    def __call__(self, sim, emp):
        return jnp.sqrt(jnp.mean((sim - emp) ** 2))


# %%
files_dir = 'D:/codes/githubs/computational_neuroscience/PyTepFit/reproduce_Momi_et_al_2022'
sc_file = files_dir + '/Schaefer2018_200Parcels_7Networks_count.csv'
high_file = files_dir + '/only_high_trial.mat'
dist_file = files_dir + '/Schaefer2018_200Parcels_7Networks_distance.csv'
file_leadfield = files_dir + '/leadfield'
file_eeg = files_dir + '/real_EEG'
eeg = np.load(file_eeg, allow_pickle=True)
eeg
# %%
eeg.shape
# %%
lm = np.load(file_leadfield, allow_pickle=True)
# %%
lm.shape
# %%
data_high = scipy.io.loadmat(high_file)
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(data_high['only_high_trial'][0].T)
# ax[1].plot(data_high['only_high_trial'][6].T)
# plt.show()
# %%
data_high['only_high_trial'].shape
# %%
sc_df = pd.read_csv(sc_file, header=None, sep=' ')
sc = sc_df.values

sc = 0.5 * (sc + sc.T)
sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))
# %%
sc.shape
# %%
dist_df = pd.read_csv(dist_file, header=None, sep=' ')
dist = dist_df.values
# %%
dist.shape
# %%
stim_weights_file = files_dir + '/stim_weights.npy'
stim_weights = np.load(stim_weights_file)

# plt.plot(stim_weights)
# plt.show()
# %%
stim_weights.shape
# %%
node_size = stim_weights.shape[0]
output_size = lm.shape[0]
batch_size = 50
input_size = 3
num_epoches = 120
tr = 0.001 * u.second
step_size = 0.0001 * u.second
n_time_per_duration = int(tr / step_size)
lm_v = np.zeros((output_size, node_size))
# %%
brainstate.environ.set(dt=step_size)


# %%
def train_one_subject(sub_index):
    print(f'sub: {sub_index}')
    data_mean = np.array(data_high['only_high_trial'][sub_index]).T
    # data_mean: [2000, n_channel]
    # data_mean: [2000, 62]

    lm = np.load(f'{files_dir}/leadfield_from_mne/sub{str(sub_index + 1).zfill(3)}/leadfield.npy', allow_pickle=True)
    net = JansenRitNetwork(
        sc=sc,
        lm=brainstate.random.normal(lm, 1.0 / (1.0 + lm_v)),
        w_bb=brainstate.random.normal(sc, 1 / 50.0),
        dist=dist,
        A=3.25,
        a=Parameter(brainstate.random.normal(100., 1 / 0.5)),
        B=22.,
        b=Parameter(brainstate.random.normal(50., 1.0)),
        g=Parameter(brainstate.random.normal(1000., 1 / 0.1)),
        c1=Parameter(brainstate.random.normal(135, 1 / 0.2)),
        c2=Parameter(brainstate.random.normal(135 * 0.8, 1 / 0.4)),
        c3=Parameter(brainstate.random.normal(135 * 0.25, 1 / 0.8)),
        c4=Parameter(brainstate.random.normal(135 * 0.25, 1 / 0.8)),
        std_in=Parameter(brainstate.random.normal(100., 1 / 10.)),
        vmax=5.0,
        v0=6.0,
        r=0.56,
        y0=Parameter(brainstate.random.normal(2.0, 1 / 2.0, output_size)),
        mu=Parameter(brainstate.random.normal(1., 1 / 2.5)),
        k=Parameter(brainstate.random.normal(10., 1 / 0.3)),
        cy0=5.0,
        ki=stim_weights,
        fit_gains_flat=True,
    )

    fitter = ModelFitting(
        net,
        optimizer=brainstate.optim.Adam(1e-3),
        duration_per_batch=batch_size * tr,
        time_per_duration=tr,
    )

    # uuu: [n_time, batch_size, node_size]
    uuu = np.zeros((400, n_time_per_duration, node_size))
    uuu[110:120] = 1000
    train_out = fitter.train(data_mean[900:1300], uuu, n_epoch=120)

    uuu = np.zeros((400, n_time_per_duration, node_size))
    uuu[110:120] = 1000
    test_out = fitter.test(data_mean[900:1300], uuu)

    # filename = f'reproduce_fig/sub_{sub_index}_fittingresults_stim_exp.pkl'
    # with open(filename, 'wb') as f:
    #     pickle.dump(fitter, f)
    # outfilename = f'reproduce_fig/sub_{sub_index}_simEEG_stim_exp.pkl'
    # with open(outfilename, 'wb') as f:
    #     pickle.dump({'train': train_out, 'test': test_out}, f)
    #
    # sc_mod = np.zeros((node_size, node_size))
    # mask = np.tril_indices(node_size, -1)
    # sc_mod[mask] = train_out.weights[-10:, :].mean(0)
    # sc_mod = sc_mod + sc_mod.T
    #
    # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    # ax.imshow(np.log1p(sc_mod), cmap='bwr')
    # plt.show()
    #
    # fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    # ax[0].plot((test_out.E - test_out.I).T)
    # ax[0].set_title('Test: sourced EEG')
    # ax[1].plot(test_out.eeg.T)
    # ax[1].set_title('Test')
    # ax[2].plot(data_high['only_high_trial'][sub_index].T[900:1300, :])
    # ax[2].set_title('empirical')
    # plt.show()


# %%
train_one_subject(0)
