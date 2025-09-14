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
import functools
import pickle
from pathlib import Path
from typing import Callable

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import cosine_similarity

import brainmass
import braintools


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


# %%
class Parameter(brainstate.ParamState, u.CustomArray):
    def __init__(
        self,
        value,
        transform: braintools.Transform = braintools.IdentityTransform()
    ):
        value = transform.inverse(value)
        super().__init__(value)
        self.transform = transform

    @property
    def data(self):
        return self.transform(self.value)

    # @data.setter
    # def data(self, v):
    #     self.value = self.transform.inverse(v)


# %%
class Network(brainstate.nn.Module):
    def __init__(
        self,
        sc: np.ndarray,
        dist: np.ndarray,
        lm: u.Quantity,
        w_bb: brainstate.typing.ArrayLike,
        gc=1e3,

        # node parameters
        Ae=3.25 * u.mV,  # Excitatory gain
        Ai=22. * u.mV,  # Inhibitory gain
        be=100. / u.second,  # Excit. time const
        bi=50. / u.second,  # Inhib. time const.
        C=135.,  # Connect. const.
        a1=1.,  # Connect. param.
        a2=0.8,  # Connect. param.
        a3=0.25,  # Connect. param
        a4=0.25,  # Connect. param.
        s_max=5. * u.Hz,  # Max firing rate
        v0=6. * u.mV,  # Firing threshold
        r=0.56,
        std_in=150. * u.Hz,
        var_init: Callable = brainstate.init.Uniform(0., 5.0, unit=u.mV),
        mom_init: Callable = brainstate.init.Uniform(0., 5.0, unit=u.mV / u.second),
        conduct_lb: float = 1.5,  # lower bound for conduct velocity
    ):
        super().__init__()

        self.w_bb = w_bb
        self.sc = sc
        self.gc = gc

        self.hidden_size = sc.shape[0]
        self.output_size = lm.shape[1]
        delay_time = dist / conduct_lb * brainstate.environ.get_dt()
        delay_indices = np.expand_dims(np.arange(self.hidden_size), axis=0)
        delay_indices = np.tile(delay_indices, (self.hidden_size, 1))

        self.node = brainmass.JansenRitModel(
            in_size=sc.shape[0],
            Ae=Ae,
            Ai=Ai,
            be=be,
            bi=bi,
            C=C,
            a1=a1,
            a2=a2,
            a3=a3,
            a4=a4,
            s_max=s_max,
            v0=v0,
            r=r,
            M_init=var_init,
            E_init=var_init,
            I_init=var_init,
            Mv_init=mom_init,
            Ev_init=mom_init,
            Iv_init=mom_init,
            noise_E=brainmass.GaussianNoise(sc.shape[0], sigma=std_in),
            fr_scale=Scale(5e2),
        )
        self.delay_E = brainstate.nn.StateWithDelay(self.node, 'E', init=var_init)
        self.delay_E.register_entry('d', delay_time, delay_indices)
        self.lm = brainmass.EEGLeadFieldModel(self.sc.shape[0], lm.shape[-1], L=lm)

    def effective_sc(self):
        w = u.math.exp(self.w_bb) * self.sc
        w2 = u.math.log1p((w + w.T) / 2)
        w_n = w2 / u.math.linalg.norm(w2)
        return w_n

    def update(self, conn, E_delayed_inp, E_inp):
        E_inp = E_inp + E_delayed_inp - self.gc * self.node.E.value * u.math.sum(conn, axis=1)
        self.node(E_inp=E_inp)

    def update_duration(self, E_inp_duration, index):
        # E_inp_duration: [n_time_per_duration, n_feature]
        # index : int
        with brainstate.environ.context(i=index):
            conn = self.effective_sc() * u.Hz / u.mV
            E_delayed_inp = brainmass.additive_coupling(self.delay_E.at('d'), conn, self.gc)
            brainstate.transform.for_loop(functools.partial(self.update, conn, E_delayed_inp), E_inp_duration)
            self.delay_E.update(self.node.E.value)
            eeg = self.lm(self.node.eeg())
            return {
                'M': self.node.M.value,
                'I': self.node.I.value,
                'E': self.node.E.value,
                'Mv': self.node.Mv.value,
                'Ev': self.node.Ev.value,
                'Iv': self.node.Iv.value,
                'eeg': eeg,
            }

    def update_batch(self, i_duration_start, E_inp_batch):
        # E_inp_batch: [n_duration, n_time_per_duration, n_feature]
        return brainstate.transform.for_loop(
            self.update_duration,
            E_inp_batch,
            i_duration_start + np.arange(E_inp_batch.shape[0])
        )


# %%
class ModelFitting:
    def __init__(
        self,
        model: Network,
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
    def _batch_train(self, i_duration, inputs, targets):
        # inputs: [n_duration, n_time_per_duration, n_input]
        # targets: [n_duration, n_time_per_duration, n_output]

        def f_loss():
            out = self.model.update_batch(i_duration, inputs)
            loss_ = self.cost(out['eeg'], targets)
            return loss_, out

        f_grad = brainstate.transform.grad(f_loss, grad_states=self.weights, return_value=True, has_aux=True)
        grads, loss, out_batch = f_grad()
        grads = brainstate.functional.clip_grad_norm(grads, self.grad_clip)
        # jax.debug.print("Gradients = {g}", g=jax.tree.map(lambda x: jnp.max(jnp.abs(x)), grads))
        self.optimizer.update(grads)
        return loss, out_batch

    def train(
        self,
        data: u.Quantity,
        uuu: u.Quantity,
        n_epoch: int,
        epoch_min: int = 10,  # run minimum epoch # part of stop criteria
        r_lb: float = 0.85,  # lowest pearson correlation # part of stop criteria
    ) -> JansenRitOutput:
        """
        Train function following model_fit_LM.py logic
        
        Parameters
        ---------- 
        data: ArrayLike
            Empirical data [n_time, n_channels]
        uuu: ArrayLike
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
        num_batches = int(data.shape[0] / duration)

        for i_epoch in range(n_epoch):
            losses = []
            output_eeg = []
            for i_batch in range(num_batches):
                inputs = uuu[i_batch * duration:(i_batch + 1) * duration]
                targets = data[i_batch * duration:(i_batch + 1) * duration]
                loss, out_batch = self._batch_train(i_batch, inputs, targets)
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
            fc = np.corrcoef(u.get_mantissa(data.T))
            ts_sim = u.get_magnitude(u.math.concatenate(output_eeg, axis=0))
            fc_sim = np.corrcoef(ts_sim[10:].T)  # Skip first 10 timepoints
            corr = u.math.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1]
            cos_sim = u.math.diag(cosine_similarity(ts_sim.T, u.get_magnitude(data.T))).mean()

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
    def _batch_predict(self, i_duration, inputs):
        # inputs: [n_duration, n_time, n_input]
        return self.model.update_batch(i_duration, inputs)

    def test(
        self,
        uuu: u.Quantity,
        base_batch_num: int = 20,
        data: np.ndarray = None,
    ) -> JansenRitOutput:
        """
        Test function following model_fit_LM.py logic
        
        Parameters
        ----------
        uuu: ArrayLike
            Input stimulation [n_time, n_nodes]
        base_batch_num: int
            Number of baseline batches before actual data (like warmup)
        data: ArrayLike
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
        u_hat = np.zeros((total_time_steps, *uuu.shape[1:])) * uuu.unit
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
            out_batch = self._batch_predict(start_idx, batch_input)

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
            ts_sim = u.get_magnitude(u.math.concatenate(output.eeg, axis=0))
            fc_sim = np.corrcoef(ts_sim[transient_num:].T)  # Skip transients
            fc_emp = np.corrcoef(u.get_mantissa(data.T))

            mask_e = np.tril_indices(self.model.output_size, -1)
            corr = np.corrcoef(fc_sim[mask_e], fc_emp[mask_e])[0, 1]
            cos_sim = np.diag(cosine_similarity(ts_sim.T, u.get_magnitude(data.T))).mean()
            print(f'Test - Pearson correlation: {corr:.3f}, Cosine similarity: {cos_sim:.3f}')

        return output


# %%
class DistCost:
    def __call__(self, sim, emp):
        return u.get_mantissa(u.math.sqrt(u.math.mean((sim - emp) ** 2)))


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
n_duration_per_batch = 50
num_epoches = 120
duration_length = 0.001 * u.second
step_size = 0.0001 * u.second
n_time_per_duration = int(duration_length / step_size)

# %%
brainstate.environ.set(dt=step_size)


# %%
def train_one_subject(sub_index):
    print(f'sub: {sub_index}')
    data_mean = np.array(data_high['only_high_trial'][sub_index]).T * u.mV
    # data_mean: [2000, n_channel]
    # data_mean: [2000, 62]

    lm = np.load(f'{files_dir}/leadfield_from_mne/sub{str(sub_index + 1).zfill(3)}/leadfield.npy', allow_pickle=True)
    lm = lm - u.math.mean(lm, axis=0, keepdims=True)

    net = Network(
        sc=sc,
        lm=lm.T * u.mV / (u.nA * u.meter),  # Use base leadfield matrix
        w_bb=Parameter(sc + 0.05),
        dist=dist,
        # Ae=Parameter(3.25 * u.mV, braintools.SigmoidTransform(2 * u.mV, 10 * u.mV)),
        # Ai=Parameter(22. * u.mV, braintools.SigmoidTransform(17.0 * u.mV, 110 * u.mV)),
        # be=Parameter(100. * u.Hz, braintools.SigmoidTransform(5.0 * u.Hz, 150. * u.Hz)),
        # bi=Parameter(50. * u.Hz, braintools.SigmoidTransform(25.0 * u.Hz, 75 * u.Hz)),
        # a1=Parameter(1.0, braintools.SigmoidTransform(0.5, 1.5)),
        # a2=Parameter(0.8, braintools.SigmoidTransform(0.4, 1.2)),
        # a3=Parameter(0.25, braintools.SigmoidTransform(0.125, 0.375)),
        # a4=Parameter(0.25, braintools.SigmoidTransform(0.125, 0.375)),
        # gc=Parameter(1e3, braintools.SigmoidTransform(10, 2e3)),
        Ae=Parameter(3.25 * u.mV, braintools.SoftplusTransform(1.0 * u.mV)),
        Ai=Parameter(22. * u.mV, braintools.SoftplusTransform(1.0 * u.mV)),
        be=Parameter(100. * u.Hz, braintools.SoftplusTransform(1.0 * u.Hz)),
        bi=Parameter(50. * u.Hz, braintools.SoftplusTransform(1.0 * u.Hz)),
        a1=Parameter(1.0, braintools.SoftplusTransform(0.01)),
        a2=Parameter(0.8, braintools.SoftplusTransform(0.01)),
        a3=Parameter(0.25, braintools.SoftplusTransform(0.01)),
        a4=Parameter(0.25, braintools.SoftplusTransform(0.01)),
        gc=Parameter(1e3, braintools.SoftplusTransform(100.0)),
        std_in=Parameter(250 * u.Hz, braintools.SoftplusTransform(100.0 * u.Hz)),
        # Ae=Parameter(3.25 * u.mV),
        # Ai=Parameter(22. * u.mV),
        # be=Parameter(100. * u.Hz),
        # bi=Parameter(50. * u.Hz),
        # a1=Parameter(1.0),
        # a2=Parameter(0.8),
        # a3=Parameter(0.25),
        # a4=Parameter(0.25),
        # gc=Parameter(1e3),
        s_max=5.0 * u.Hz,
        v0=6.0 * u.mV,
        r=0.56,
        conduct_lb=2.5,
    )

    fitter = ModelFitting(
        net,
        optimizer=brainstate.optim.Adam(1e-2),  # Match model_fit_LM.py learning rate
        duration_per_batch=n_duration_per_batch * duration_length,
        time_per_duration=duration_length,
    )

    # Create input stimulation like model_fit_LM.py: [n_time, n_nodes]
    uuu = np.zeros((400, n_time_per_duration, node_size)) * u.Hz  # [n_time, n_nodes]
    uuu[110:120] = 1000. * u.Hz  # Stimulation from time 110 to 120
    uuu = uuu * stim_weights
    train_out = fitter.train(data_mean[900:1300], uuu, n_epoch=200)

    # Test with same stimulation
    uuu_test = np.zeros((400, n_time_per_duration, node_size)) * u.Hz
    uuu_test[110:120] = 1000. * u.Hz
    uuu_test = uuu_test * stim_weights
    test_out = fitter.test(uuu_test, base_batch_num=20, data=data_mean[900:1300])


# %%
train_one_subject(0)
