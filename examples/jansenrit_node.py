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


import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import brainmass

brainstate.environ.set(dt=0.1 * u.ms)


def show(times, data):
    M, E, I, eeg = data
    fig, gs = braintools.visualize.get_figure(4, 1, 3, 10)
    fig.add_subplot(gs[0, 0])
    plt.plot(times, M)
    plt.ylabel('M (mV)')

    fig.add_subplot(gs[1])
    plt.plot(times, E)
    plt.ylabel('E (mV)')

    fig.add_subplot(gs[2])
    plt.plot(times, I)
    plt.ylabel('I (mV)')

    fig.add_subplot(gs[3])
    plt.plot(eeg)
    plt.xlabel('Time (ms)')
    plt.ylabel('EEG (mV)')
    plt.show()


def alpha_oscillation():
    node = brainmass.JansenRitModel(
        1,
    )
    brainstate.nn.init_all_states(node)

    def step_run(inp):
        eeg = node.update(E_inp=inp)
        return node.M.value, node.E.value, node.I.value, eeg

    indices = np.arange(10000)
    inputs = brainstate.random.normal(120., 30., indices.shape) * u.Hz
    data = brainstate.transform.for_loop(step_run, inputs)
    show(indices * brainstate.environ.get_dt(), data)


def beta_oscillation():
    node = brainmass.JansenRitModel(
        # 1, a1=135 * 1., a2=135 * 0.8, a3=135 * 0.25, a4=135 * 0.25,
        1,
    )
    brainstate.nn.init_all_states(node)

    def step_run(i):
        eeg = node.update(E_inp=320. * u.Hz)
        return node.M.value, node.E.value, node.I.value, eeg

    indices = np.arange(10000)
    data = brainstate.transform.for_loop(step_run, indices)
    show(indices * brainstate.environ.get_dt(), data)


def resting_state():
    node = brainmass.JansenRitModel(
        # 1, a1=135 * 1., a2=135 * 0.8, a3=135 * 0.25, a4=135 * 0.25,
        1,
    )
    brainstate.nn.init_all_states(node)

    def step_run(i):
        eeg = node.update(E_inp=90. * u.Hz)
        return node.M.value, node.E.value, node.I.value, eeg

    indices = np.arange(10000)
    data = brainstate.transform.for_loop(step_run, indices)
    show(indices * brainstate.environ.get_dt(), data)


if __name__ == '__main__':
    alpha_oscillation()
    # resting_state()
