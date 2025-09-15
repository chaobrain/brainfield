import brainstate
import braintools
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from datasets import Dataset
import time
import os
import h5py

from brainmass import AlnModel, OUProcess
from brainstate.environ import get_dt 

brainstate.environ.set(dt=0.1)     
plt.rcParams['image.cmap'] = 'plasma'

"""
单点模拟
"""
"""
print("\nRunning single node simulation...")
node = AlnModel(
    1,
    noise_E=OUProcess(1, sigma=0.1),
    noise_I=OUProcess(1, sigma=0.1),
)
brainstate.nn.init_all_states(node)

ext_E_current = 0.82
ext_I_current = 0.7

def step_single(i):
    with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
        return node.update(ext_exc_current=ext_E_current, ext_inh_current=ext_I_current)

indices = np.arange(10000)
rE_trace, rI_trace, _, _ = brainstate.transform.for_loop(step_single, indices)

t = indices * brainstate.environ.get_dt()
plt.figure(figsize=(6, 3))
plt.plot(t, rE_trace[:, 0], label='rE')
plt.plot(t, rI_trace[:, 0], label='rI')
plt.title('Single ALN oscillator (neurolib-consistent)')
plt.xlabel('time (ms)'); plt.ylabel('rate'); plt.legend()
plt.tight_layout(); plt.show()
"""

"""
分岔图
"""
lookup_file = os.path.join(
    os.path.dirname(__file__), "datasets", "quantities_cascade.h5"
)  

with h5py.File(lookup_file, "r") as f:
    Irange        = jnp.asarray(f["mu_vals"][:])          # (n_mu,)
    sigmarange    = jnp.asarray(f["sigma_vals"][:])       # (n_sigma,)
    precalc_r     = jnp.asarray(f["r_ss"][:])             # (n_sigma, n_mu)  [kHz]
    precalc_V     = jnp.asarray(f["V_mean_ss"][:])        # (n_sigma, n_mu)  [mV]
    precalc_tau_mu= jnp.asarray(f["tau_mu_exp"][:])       # (n_sigma, n_mu)  [ms]

dI = Irange[1] - Irange[0]
ds = sigmarange[1] - sigmarange[0]
I_vals = np.linspace(0.0, 5.0, 80)          # 外部兴奋电流（mV/ms）
bif_model = AlnModel(
    in_size=I_vals.size,            # 每个节点对应一个 I_ext 值
    precalc_r=precalc_r,
    precalc_V=precalc_V,
    precalc_tau_mu=precalc_tau_mu,
    Irange=Irange,
    sigmarange=sigmarange,
    dI=dI,
    ds=ds,
)
brainstate.nn.init_all_states(bif_model)

ext_inh_current_const = 0.7          

# ------------------------------------------------------------------
def step_bif(i):
    with brainstate.environ.context(i=i, t=i * get_dt()):
        return bif_model.update(
            rowsum=0.,
            rowsumq=0.,
            ext_exc_current=I_vals,
            ext_inh_current=ext_inh_current_const
        )

    
n_steps = 40000
indices = np.arange(n_steps)
rE_bif, *_ = brainstate.transform.for_loop(step_bif, indices)
print("rE_bif min/max:", rE_bif.min(), rE_bif.max())

rE_last = rE_bif[-10000:]
rE_max = rE_bif 
rE_min = rE_bif
print("I_vals min/max:", I_vals.min(), I_vals.max())
print("I_vals shape:", I_vals.shape)
print("rE_max shape:", rE_max.shape)
print("rE_max:", rE_max)
print("rE_bif shape:", rE_bif.shape)
plt.figure(figsize=(5, 3))
plt.plot(I_vals, rE_max, 'k', lw=2, label='max rE')
plt.plot(I_vals, rE_min, 'k', lw=2, label='min rE')
plt.xlabel('External excitatory current (mV/ms)')
plt.ylabel('Excitatory rate (Hz)')
plt.title('ALN bifurcation diagram (single node, real lookup table)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


"""
全脑
"""
print("\nRunning brain network simulation...")
hcp = Dataset('hcp')
Cmat = hcp.Cmat.copy()
np.fill_diagonal(Cmat, 0)
Dmat = hcp.Dmat / 0.1  

class ALNNetwork(brainstate.nn.Module):
    def __init__(self, k_gl=0.3):
        super().__init__()
        n_node = Cmat.shape[0]

        self.node = AlnModel(
            n_node,            
            noise_E=OUProcess(n_node, sigma=0.01),
            noise_I=OUProcess(n_node, sigma=0.01),
        )



