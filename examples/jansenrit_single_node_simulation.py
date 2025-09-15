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

brainstate.environ.set(dt=0.0005 * u.second)
brainstate.environ.set(dt=0.5 * u.ms)


def show(times, data, title):
    M, E, I, eeg = data
    fig, gs = braintools.visualize.get_figure(4, 1, 1.2, 10)
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
    plt.plot(times, eeg)
    plt.xlabel('Time (ms)')
    plt.ylabel('EEG (mV)')

    plt.suptitle(title)
    plt.show()


def alpha_oscillation():
    # Alpha-like idle rhythm (baseline JR)
    node = brainmass.JansenRitModel(1)
    brainstate.nn.init_all_states(node)

    def step_run(inp):
        eeg = node.update(E_inp=inp)
        return node.M.value, node.E.value, node.I.value, eeg

    dt = brainstate.environ.get_dt()
    indices = np.arange(int(10 * u.second / dt))
    inputs = brainstate.random.normal(120., 30. / (dt / u.second) ** 0.5, indices.shape) * u.Hz
    data = brainstate.transform.for_loop(step_run, inputs)
    show(indices * dt, data, title='Alpha-like oscillation')


def sinusoidal_oscillation():
    # Driven/entrained rhythm via sinusoidal input
    node = brainmass.JansenRitModel(1)
    brainstate.nn.init_all_states(node)
    f_drive = 10. * u.Hz

    def step_run(i):
        tt = i * dt
        inp = 80 + 70.0 * u.math.sin(2 * u.math.pi * f_drive * tt)
        eeg = node.update(E_inp=inp * u.Hz)
        return node.M.value, node.E.value, node.I.value, eeg

    dt = brainstate.environ.get_dt()
    indices = np.arange(int(5. * u.second / dt))
    data = brainstate.transform.for_loop(step_run, indices)
    show(indices * dt, data, title='Sinusoidal-driven oscillation')


def spike_wave_oscillation():
    # Spike–wave–like regime (strong excitation / reduced inhibition)
    node = brainmass.JansenRitModel(1, Ae=4.5 * u.mV, Ai=18. * u.mV, bi=40. * u.Hz)
    brainstate.nn.init_all_states(node)

    def step_run(inp):
        eeg = node.update(E_inp=inp)
        return node.M.value, node.E.value, node.I.value, eeg

    dt = brainstate.environ.get_dt()
    indices = np.arange(int(10 * u.second / dt))
    inputs = brainstate.random.normal(90., 10. / (dt / u.second) ** 0.5, indices.shape) * u.Hz
    data = brainstate.transform.for_loop(step_run, inputs)
    show(indices * dt, data, title='Spike-wave-like oscillation')


def low_inhibition():
    # Low inhibition (disinhibited, larger amplitude, slower)
    node = brainmass.JansenRitModel(1, Ai=15. * u.mV, bi=40. * u.Hz)
    brainstate.nn.init_all_states(node)

    def step_run(inp):
        eeg = node.update(E_inp=inp)
        return node.M.value, node.E.value, node.I.value, eeg

    dt = brainstate.environ.get_dt()
    indices = np.arange(int(10 * u.second / dt))
    inputs = brainstate.random.normal(100., 20. / (dt / u.second) ** 0.5, indices.shape) * u.Hz
    data = brainstate.transform.for_loop(step_run, inputs)
    show(indices * dt, data, title='Low inhibition oscillation')


def irregular_noisy():
    # Irregular/noisy (higher drive + noise)
    node = brainmass.JansenRitModel(1)
    brainstate.nn.init_all_states(node)

    def step_run(inp):
        eeg = node.update(E_inp=inp)
        return node.M.value, node.E.value, node.I.value, eeg

    dt = brainstate.environ.get_dt()
    indices = np.arange(int(10 * u.second / dt))
    inputs = brainstate.random.normal(220., 80. / (dt / u.second) ** 0.5, indices.shape) * u.Hz
    data = brainstate.transform.for_loop(step_run, inputs)
    show(indices * dt, data, title='Irregular/noisy oscillation')


if __name__ == '__main__':
    alpha_oscillation()
    sinusoidal_oscillation()
    spike_wave_oscillation()
    low_inhibition()
    irregular_noisy()
