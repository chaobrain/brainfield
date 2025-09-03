# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import brainfield
import brainstate
import braintools
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset


hcp = Dataset('hcp')
brainstate.environ.set(dt=0.1)


class Network(brainstate.nn.Module):
    def __init__(self, signal_speed=2.):
        super().__init__()

        conn_weight = hcp.Cmat
        np.fill_diagonal(conn_weight, 0)
        delay_time = hcp.Dmat / signal_speed
        np.fill_diagonal(delay_time, 0)
        indices_ = np.tile(np.arange(conn_weight.shape[1]), conn_weight.shape[0])

        self.node = brainfield.WilsonCowanModel(
            80,
            noise_E=brainfield.OUProcess(80, sigma=0.01),
            noise_I=brainfield.OUProcess(80, sigma=0.01),
        )
        self.coupling = brainfield.DiffusiveCoupling(
            self.node.prefetch_delay('rE', (delay_time.flatten(), indices_), init=brainstate.init.Uniform(0, 0.05)),
            self.node.prefetch('rE'),
            conn_weight,
            k=1.0
        )

    def update(self):
        current = self.coupling()
        rE = self.node(current)
        return rE

    def step_run(self, i):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            return self.update()


net = Network()
brainstate.nn.init_all_states(net)
indices = np.arange(0, 6e3 // brainstate.environ.get_dt())
exes = brainstate.transform.for_loop(net.step_run, indices)

plt.rcParams['image.cmap'] = 'plasma'
fig, gs = braintools.visualize.get_figure(1, 2, 4, 6)
ax1 = fig.add_subplot(gs[0, 0])
fc = braintools.metric.functional_connectivity(exes)
ax = ax1.imshow(fc)
plt.colorbar(ax, ax=ax1)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(indices, exes[:, ::5], alpha=0.8)
plt.show()
