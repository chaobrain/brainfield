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
import pickle
from pathlib import Path
from typing import Union, Callable

import jax
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy
from brainstate import maybe_state
from sklearn.metrics.pairwise import cosine_similarity


# %%
class Parameter(brainstate.ParamState, u.CustomArray):
    pass


# %%
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

    def finalize(self):
        self.M = jnp.stack(self.M, axis=0)
        self.E = jnp.stack(self.E, axis=0)
        self.I = jnp.stack(self.I, axis=0)
        self.Mv = jnp.stack(self.Mv, axis=0)
        self.Ev = jnp.stack(self.Ev, axis=0)
        self.Iv = jnp.stack(self.Iv, axis=0)
        self.eeg = jnp.stack(self.eeg, axis=0)

    def pickle(self, fn: str | Path):
        pickle.dump


# %%
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
    include the addition of parameters such as `mu`, `k`, `M`, `cy0`, and `ki`, which influence
    the dynamics of the system.

    The equations governing the modified Jansen-Rit model are as follows:

    $$
    \begin{aligned}
    & \dot{y}_{0}=y_1,\\
    & \dot{y_2}=y_3,\\
    & \dot{y_4}=y_5,\\
    &\dot{y}_{1}=A_eb_eS(I_p+a_2y_2-a_4y_4)-2b_ey_1-b_e^2y_0,\\
    &\dot{y}_{3}=A_eb_eS(a_1y_0)-2b_ey_3-b_e^2y_2,\\
    &\ddot{y}_{5}=A_ib_iS(I_i+a_3y_0)-2b_iy_5-b_i^2y_4 + \mu k S(M - cy0) - ki Iv.
    \end{aligned}
    $$

    The sigmoid function $S(v)$ remains unchanged from the original Jansen-Rit model:

    $$
    S(v)=S_{\max } \cdot \frac{1}{1+e^{-r\left(v-v_0\right)}}
    $$

    The additional parameters introduced in this modified model are defined as follows:

    - `mu`: A scaling factor that modulates the influence of the additional term in the equation for $\ddot{y}_{5}$.
    - `k`: A parameter that scales the sigmoid function applied to the difference between `M` and `cy0`.
    - `M`: A reference potential that influences the dynamics of the inhibitory population.
    - `cy0`: A constant that shifts the reference potential `M`.
    - `ki`: A damping factor that affects the rate of change of `Iv`.

    """

    def __init__(
        self,
        sc: np.ndarray,
        lm: np.ndarray,
        dist: np.ndarray,
        w_bb: np.ndarray,

        # parameters
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
        self.w_bb = w_bb  # [hidden_size, hidden_size]

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
        return maybe_state(self.vmax) / (1 + u.math.exp(maybe_state(self.r) * (maybe_state(self.v0) - v)))

    def effective_sc(self):
        w = u.math.exp(maybe_state(self.w_bb)) * maybe_state(self.sc)
        w2 = u.math.log1p((w + w.T) / 2)
        w_n = w2 / u.math.linalg.norm(w2)
        return w_n

    def update(self, inputs):
        # input: [n_duration, n_time, n_input]

        dt = brainstate.environ.get_dt() / u.second
        relu = u.math.relu

        lb = maybe_state(self.lb)
        g = relu(maybe_state(self.g)) + lb
        std_in = relu(maybe_state(self.std_in))
        c1 = relu(maybe_state(self.c1)) + lb
        c2 = relu(maybe_state(self.c2)) + lb
        c3 = relu(maybe_state(self.c3)) + lb
        c4 = relu(maybe_state(self.c4)) + lb
        k = relu(maybe_state(self.k))
        B = relu(maybe_state(self.B))
        b = relu(maybe_state(self.b)) + 1.0
        A = relu(maybe_state(self.A))
        a = relu(maybe_state(self.a)) + 1.0
        mu = relu(maybe_state(self.mu))

        w_n = self.effective_sc()
        dg = -jnp.sum(w_n, axis=1)

        scaleI = Scale(self.u_2ndsys_ub)
        # scaleI = lambda x: x
        lm_t = self.lm - u.math.matmul(jnp.ones((1, self.output_size)), self.lm) / self.output_size
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
                  g * (LEd + dg * self.E.value) +
                  (self.k_lb + k) * self.ki * input_one_time +
                  c2 * self.S(c1 * self.M.value))
            # firing rate for Inhibitory population
            rI = c4 * self.S(c3 * self.M.value)

            # jax.debug.print('rM = {rM}, rE = {rE}, rI = {rI}',
            #                 rM=u.math.max(u.math.abs(rM)),
            #                 rE=u.math.max(u.math.abs(rE)),
            #                 rI=u.math.max(u.math.abs(rI)),)

            # Update the states by step-size.
            ddM = self.M.value + dt * self.Mv.value
            ddE = self.E.value + dt * self.Ev.value
            ddI = self.I.value + dt * self.Iv.value
            ddMv = self.Mv.value + dt * self.sys2nd(A, a, scaleI(rM), self.M.value, self.Mv.value)
            ddEv = self.Ev.value + dt * self.sys2nd(A, a, scaleI(rE), self.E.value, self.Ev.value)
            ddIv = self.Iv.value + dt * self.sys2nd(B, b, scaleI(rI), self.I.value, self.Iv.value)

            # Calculate the saturation for model states (for stability and gradient calculation).
            self.E.value = ddE
            self.I.value = ddI
            self.M.value = ddM
            self.Ev.value = ddEv
            self.Iv.value = ddIv
            self.Mv.value = ddMv

            # update placeholders for E buffer
            # self.delay.value = self.delay.value.at[0].set(self.E.value)
            # self.delay.value = jnp.concatenate((jnp.expand_dims(self.E.value, axis=0), self.delay.value[:-1]), axis=0)

        def one_duration(input_one_batch):
            # input_one_batch: [n_time, n_input]
            brainstate.transform.for_loop(one_time, input_one_batch)
            self.delay.value = jnp.concatenate((jnp.expand_dims(self.E.value, axis=0), self.delay.value[:-1]), axis=0)
            eeg_ = self.s2o_coef * self.cy0 * jnp.matmul(lm_t, self.E.value - self.I.value) - self.y0
            return {
                'M': self.M.value,
                'I': self.I.value,
                'E': self.E.value,
                'Mv': self.Mv.value,
                'Ev': self.Ev.value,
                'Iv': self.Iv.value,
                'eeg': eeg_,
            }

        return brainstate.transform.for_loop(one_duration, inputs)


# %%
class ModelFitting:
    def __init__(
        self,
        model: JansenRitNetwork,
        optimizer: brainstate.optim.Optimizer,
        duration_per_batch: u.Quantity,
        time_per_duration: u.Quantity,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.weights = self.model.states(brainstate.ParamState)
        self.optimizer.register_trainable_weights(self.weights)
        self.cost = DistCost()

        dt = brainstate.environ.get_dt()
        self.n_duration_per_batch = int(duration_per_batch / time_per_duration)
        self.n_time_per_duration = int(time_per_duration / dt)
        self.grad_clip = grad_clip

    @brainstate.transform.jit(static_argnums=0)
    def _batch_train(self, inputs, targets):
        # inputs: [n_duration, n_time_per_duration, n_input]
        # targets: [n_duration, n_time_per_duration, n_output]

        def f_loss():
            out = self.model(inputs)
            loss_ = 10. * self.cost(out['eeg'], targets)
            return loss_, out

        f_grad = brainstate.transform.grad(f_loss, grad_states=self.weights, return_value=True, has_aux=True)
        grads, loss, out_batch = f_grad()
        grads = brainstate.functional.clip_grad_norm(grads, self.grad_clip)
        # jax.debug.print("Gradients = {g}", g=jax.tree.map(lambda x: jnp.max(jnp.abs(x)), grads))
        self.optimizer.update(grads)
        return loss, out_batch

    def train(
        self,
        data: np.ndarray,
        uuu: np.ndarray,
        n_epoch: int,
        epoch_min: int = 10,  # run minimum epoch # part of stop criteria
        r_lb: float = 0.85,  # lowest pearson correlation # part of stop criteria
    ) -> JansenRitOutput:
        """
        Train function following model_fit_LM.py logic
        
        Parameters
        ---------- 
        data: np.ndarray
            Empirical data [n_time, n_channels]
        uuu: np.ndarray
            Input stimulation [n_time, n_nodes]
        n_epoch: int
            Number of training epochs
        epoch_min: int
            Minimum number of epochs to run before checking stop criteria
        r_lb: float
            Lower bound of Pearson correlation to stop training
        """
        brainstate.nn.init_all_states(self.model)

        # define masks for getting lower triangle matrix
        mask = np.tril_indices(self.model.hidden_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # placeholders for the history of model parameters
        output = JansenRitOutput()

        duration = self.n_duration_per_batch
        num_durations = int(data.shape[0] / duration)

        for i_epoch in range(n_epoch):
            losses = []
            output_eeg = []
            for i_duration in range(num_durations):
                inputs = uuu[i_duration * duration:(i_duration + 1) * duration]
                targets = data[i_duration * duration:(i_duration + 1) * duration]
                loss, out_batch = self._batch_train(inputs, targets)
                output.M.append(out_batch['M'])
                output.E.append(out_batch['E'])
                output.I.append(out_batch['I'])
                output.Mv.append(out_batch['Mv'])
                output.Ev.append(out_batch['Ev'])
                output.Iv.append(out_batch['Iv'])
                output_eeg.append(out_batch['eeg'])
                output.loss.append(loss)
                losses.append(loss)

            # Calculate metrics like model_fit_LM.py
            fc = np.corrcoef(data.T)
            ts_sim = np.concatenate(output_eeg, axis=0)
            fc_sim = np.corrcoef(ts_sim[10:].T)  # Skip first 10 timepoints
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
        return self.model(inputs)

    def test(
        self,
        uuu: np.ndarray,
        base_batch_num: int = 20,
        data: np.ndarray = None,
    ) -> JansenRitOutput:
        """
        Test function following model_fit_LM.py logic
        
        Parameters
        ----------
        uuu: np.ndarray  
            Input stimulation [n_time, n_nodes]
        base_batch_num: int
            Number of baseline batches before actual data (like warmup)
        data: np.ndarray
            Empirical data [n_time, n_channels]
        """
        duration = self.n_duration_per_batch
        transient_num = 10  # Skip first 10 timepoints like model_fit_LM.py

        # Initialize model with specific initial conditions
        brainstate.nn.init_all_states(self.model)

        output = JansenRitOutput()

        # Create extended input like model_fit_LM.py:
        # u_hat has extra baseline batches + actual data
        total_time_steps = base_batch_num * duration + uuu.shape[0]
        u_hat = np.zeros((total_time_steps, *uuu.shape[1:]))
        u_hat[base_batch_num * duration:] = uuu  # Put actual input after baseline

        # Calculate total number of batches (baseline + actual)
        total_batches = int(total_time_steps / duration)

        # Process in batches like model_fit_LM.py
        for i_batch in range(total_batches):
            start_idx = i_batch * duration
            end_idx = (i_batch + 1) * duration

            # Get batch input and reshape to [n_duration, n_time, n_input]
            batch_input = u_hat[start_idx:end_idx]

            # Run model prediction (no gradients needed for test)
            out_batch = self._batch_predict(batch_input)

            # Only collect outputs after baseline batches (like model_fit_LM.py)
            if i_batch >= base_batch_num:
                output.M.append(out_batch['M'])
                output.E.append(out_batch['E'])
                output.I.append(out_batch['I'])
                output.Mv.append(out_batch['Mv'])
                output.Ev.append(out_batch['Ev'])
                output.Iv.append(out_batch['Iv'])
                output.eeg.append(out_batch['eeg'])

        # Compute correlation metrics like model_fit_LM.py
        if data is not None:
            ts_sim = np.concatenate(output.eeg, axis=0)
            fc_sim = np.corrcoef(ts_sim[transient_num:].T)  # Skip transients
            fc_emp = np.corrcoef(data.T)

            mask_e = np.tril_indices(self.model.output_size, -1)
            corr = np.corrcoef(fc_sim[mask_e], fc_emp[mask_e])[0, 1]
            cos_sim = np.diag(cosine_similarity(ts_sim.T, data.T)).mean()
            print(f'Test - Pearson correlation: {corr:.3f}, Cosine similarity: {cos_sim:.3f}')

        return output


# %%
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
    # Initialize parameters to match model_fit_LM.py initialization strategy
    net = JansenRitNetwork(
        sc=sc,
        lm=lm + lm_v,  # Use base leadfield matrix
        w_bb=Parameter(sc + 0.05 * np.ones_like(sc)),  # Initialize w_bb similar to model_fit_LM.py
        dist=dist,
        A=Parameter(3.25 + np.random.randn() * 0.1),  # Add small random noise
        a=Parameter(100. + np.random.randn() * 2.0),  # Match model_fit_LM.py variance
        B=Parameter(22. + np.random.randn() * 0.1),
        b=Parameter(50. + np.random.randn() * 1.0),
        g=Parameter(1000. + np.random.randn() * 100.0),  # Match model_fit_LM.py variance
        c1=Parameter(135 + np.random.randn() * 5.0),
        c2=Parameter(135 * 0.8 + np.random.randn() * 2.5),
        c3=Parameter(135 * 0.25 + np.random.randn() * 1.25),
        c4=Parameter(135 * 0.25 + np.random.randn() * 1.25),
        std_in=Parameter(100. + np.random.randn() * 10.),
        vmax=5.0,
        v0=6.0,
        r=0.56,
        y0=Parameter(2.0 * np.ones(output_size) + np.random.randn(output_size) * 0.5),
        mu=Parameter(1. + np.random.randn() * 0.4),
        k=Parameter(10. + np.random.randn() * 3.3),
        cy0=5.0,
        ki=stim_weights,
    )

    fitter = ModelFitting(
        net,
        optimizer=brainstate.optim.Adam(5e-2),  # Match model_fit_LM.py learning rate
        duration_per_batch=batch_size * tr,
        time_per_duration=tr,
    )

    # Create input stimulation like model_fit_LM.py: [n_time, n_nodes]
    uuu = np.zeros((400, node_size))  # [n_time, n_nodes]
    uuu[110:120] = 1000  # Stimulation from time 110 to 120
    train_out = fitter.train(data_mean[900:1300], uuu, n_epoch=200)

    # Test with same stimulation
    uuu_test = np.zeros((400, node_size))
    uuu_test[110:120] = 1000
    test_out = fitter.test(uuu_test, base_batch_num=20, data=data_mean[900:1300])

    # outfilename = f'reproduce_fig/sub_{sub_index}_simEEG_stim_exp.pkl'
    # with open(outfilename, 'wb') as f:
    #     pickle.dump({'train': train_out, 'test': test_out}, f)
    #
    # sc = net.effective_sc()
    # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    # ax.imshow(np.log1p(sc), cmap='bwr')
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
