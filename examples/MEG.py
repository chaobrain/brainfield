import brainstate
import braintools
import brainmass
import xarray as xr
from scipy.signal import resample, butter, filtfilt, hilbert
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ipywidgets as widgets

from datasets import Dataset
brainstate.environ.set(dt=0.1)


plt.rcParams['image.cmap'] = 'plasma'
hcp = Dataset('hcp')
class Network(brainstate.nn.Module):
    def __init__(self, signal_speed=2., k=1.):
        super().__init__()

        conn_weight = hcp.Cmat
        np.fill_diagonal(conn_weight, 0)
        delay_time = hcp.Dmat / signal_speed
        np.fill_diagonal(delay_time, 0)
        indices_ = np.tile(np.arange(conn_weight.shape[1]), conn_weight.shape[0])

        self.node = brainmass.WilsonCowanModel(
            80,
            noise_E=brainmass.OUProcess(80, sigma=0.01),
            noise_I=brainmass.OUProcess(80, sigma=0.01),
        )
        self.coupling = brainmass.DiffusiveCoupling(
            self.node.prefetch_delay('rE', (delay_time.flatten(), indices_), init=brainstate.init.Uniform(0, 0.05)),
            self.node.prefetch('rE'),
            conn_weight,
            k=k
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

fig, gs = braintools.visualize.get_figure(1, 2, 4, 6)
ax1 = fig.add_subplot(gs[0, 0])
fc = braintools.metric.functional_connectivity(exes)
ax = ax1.imshow(fc)
plt.colorbar(ax, ax=ax1)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(indices, exes[:, ::5], alpha=0.8)
plt.show()


dt_ms = brainstate.environ.get_dt()
original_fs = 1000. / dt_ms # 原始采样率 (10000 Hz)
num_regions = exes.shape[1]
time_points_orig = indices * dt_ms / 1000. # 原始时间轴 (秒)

region_labels = [f'Region_{i}' for i in range(num_regions)]
sim_signal_raw = xr.DataArray(
    exes, 
    dims=("time", "regions"), 
    coords={"time": time_points_orig, "regions": region_labels}
)

# resample
target_fs = 100.0 # 目标采样率
num_samples_new = int(len(time_points_orig) * target_fs / original_fs)
resampled_data, resampled_time = resample(sim_signal_raw, num_samples_new, t=time_points_orig)

sim_signal = xr.DataArray(
    resampled_data,
    dims=("time", "regions"),
    coords={"time": resampled_time, "regions": region_labels}
)
print(f"Original sampling rate: {original_fs} Hz. Resampled to: {target_fs} Hz.")

#  bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# low-pass filter
def butter_lowpass_filter(data, highcut, fs, order=3):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, [high], btype='low')
    y = filtfilt(b, a, data)
    return y

target_region = 'Region_0' # 选择第一个脑区进行分析
signal_to_process = sim_signal.sel(regions=target_region).values

# bandpass filter
freq_band = [8.0, 12.0]
filtered_signal = butter_bandpass_filter(signal_to_process, freq_band[0], freq_band[1], fs=target_fs)

# hilbert
analytic_signal = hilbert(filtered_signal)
signal_envelope = np.abs(analytic_signal)

# low-pass
low_pass_cutoff = 4.0
smoothed_envelope = butter_lowpass_filter(signal_envelope, low_pass_cutoff, fs=target_fs)

fig_signal, ax_signal = plt.subplots(figsize=(15, 5))
plot_duration_s = 4 # 画4秒
plot_timepoints = int(plot_duration_s * target_fs)
time_axis = sim_signal.time.values[:plot_timepoints]
ax_signal.plot(time_axis, filtered_signal[:plot_timepoints], label=f'Filtered Signal ({freq_band[0]}-{freq_band[1]} Hz)', alpha=0.8)
ax_signal.plot(time_axis, signal_envelope[:plot_timepoints], label='Signal Envelope', linewidth=2, color='red')
ax_signal.plot(time_axis, smoothed_envelope[:plot_timepoints], label=f'Low-Pass Envelope (<{low_pass_cutoff} Hz)', linewidth=2.5, color='black')
ax_signal.set_title(f'Signal Processing for {target_region}')
ax_signal.set_xlabel('Time (s)')
ax_signal.set_ylabel('Amplitude')
ax_signal.legend(loc='upper right')
ax_signal.spines['top'].set_visible(False)
ax_signal.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


meg_data = xr.open_dataset('datasets/rs-meg.nc')
region_labels = meg_data.regions.values
sampling_rate = 1 / (meg_data.time[1] - meg_data.time[0]).item() # Calculate sampling rate from time coordinates
print(f"Data loaded with {len(region_labels)} regions. Sampling rate: {sampling_rate:.2f} Hz")

print('Select a region from the AAL2 atlas and a frequency range')
# Select a Region 
target = widgets.Select(options=region_labels, value='PreCG.L', description='Regions', 
                        tooltips=['Description of slow', 'Description of regular', 'Description of fast'], 
                        layout=widgets.Layout(width='50%', height='150px'))
print(target)

# Select Frequency Range
freq = widgets.IntRangeSlider(min=1, max=46, description='Frequency (Hz)', value=[8, 12], layout=widgets.Layout(width='80%'), 
                              style={'description_width': 'initial'})
print(freq)

plot_timepoints = 1000


meg_array = meg_data['__xarray_dataarray_variable__']
if meg_array.dims[0] != 'time':
    meg_array = meg_array.transpose('time', 'regions')    # 让 time 在第一维
fs = sampling_rate                                    # 100 Hz
low, high = freq.value                                # 例如 [8,12]
plot_len = int(plot_timepoints)                       # 确保是整数
time_vals = meg_array.time[:plot_len].values 

# 3. 取目标区域
fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
y_raw_target = meg_array.sel(regions=target.value).values 
sns.lineplot(x=time_vals, y=y_raw_target[:plot_len], ax=ax[0], color='k', alpha=0.6)
ax[0].set_title(f'Unfiltered Signal ({target.value})')

# Band Pass Filter the Signal
y_filt_target = butter_bandpass_filter(y_raw_target, low, high, fs)
y_env_target  = np.abs(hilbert(y_filt_target))  

sns.lineplot(x=time_vals, y=y_filt_target[:plot_len], ax=ax[1], label='Bandpass-Filtered Signal')
sns.lineplot(x=time_vals, y=y_env_target[:plot_len], ax=ax[1], label='Signal Envelope')
ax[1].set_title(f'Filtered Signal ({target.value})');
ax[1].legend(bbox_to_anchor=(1.2, 1),borderaxespad=0)
sns.despine(trim=True)
plt.show()

# Orthogonalized signal envelope
print('Select a reference region for the orthogonalization')
# Select a Region 
referenz = widgets.Select(options=region_labels, value='PreCG.R', description='Regions',
                          tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
                          layout=widgets.Layout(width='50%', height='150px'))
print(referenz)

y_raw_reference = meg_array.sel(regions=referenz.value).values
y_filt_reference = butter_bandpass_filter(y_raw_reference, low, high, fs)

complex_target = hilbert(y_filt_target)
complex_reference = hilbert(y_filt_reference)
env_reference = np.abs(complex_reference)
env_target = np.abs(complex_target)


signal_conj_div_env = complex_reference.conj() / env_reference
orth_signal_imag = (complex_target * signal_conj_div_env).imag

orth_env = np.abs(orth_signal_imag)

fig_final, ax_final = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax_final[0].plot(time_vals, y_filt_reference[:plot_len], label=f'Filtered Signal ({referenz.value})')
ax_final[0].plot(time_vals, env_reference[:plot_len], label=f'Signal Envelope ({referenz.value})', color='red', linewidth=2)
ax_final[0].set_title(f'Reference Region X ({referenz.value})')
ax_final[0].legend()

ax_final[1].plot(time_vals, y_filt_target[:plot_len], label='Bandpass-Filtered Signal', alpha=0.8)
ax_final[1].plot(time_vals, env_target[:plot_len], label='Signal Envelope', linewidth=2.5)
ax_final[1].plot(time_vals, orth_env[:plot_len], label='Orthogonalized Envelope', linewidth=2.5, linestyle='--') # 使用虚线以区分

ax_final[1].set_title(f'Target Region Y ({target.value})')
ax_final[1].legend(bbox_to_anchor=(1.2, 1), borderaxespad=0)

sns.despine(trim=True)
plt.tight_layout()
plt.show()


low_pass = widgets.FloatSlider(value=1.0, min=0, max=2.0, step=0.1, description='Low-Pass Frequency (Hz)', 
                               disabled=False, readout=True, readout_format='.1f', layout=widgets.Layout(width='80%'), 
                               style={'description_width': 'initial'})
print(low_pass)

low_orth_env = butter_lowpass_filter(orth_env, highcut=low_pass.value, fs=fs)
low_signal_env = butter_lowpass_filter(env_reference, highcut=low_pass.value, fs=fs)

# Plot the smoothed envelopes for comparison
fig_corr, ax_corr = plt.subplots(1, 2, figsize=(15, 4), sharey=True)

# Plot for the reference region
ax_corr[0].plot(time_vals, env_reference[:plot_len], alpha=0.5, label='Original Envelope')
ax_corr[0].plot(time_vals, low_signal_env[:plot_len], color='black', lw=2, label=f'Smoothed Envelope (<{low_pass.value} Hz)')
ax_corr[0].set_title(f'Reference Region X ({referenz.value})')
ax_corr[0].legend()

# Plot for the target region
ax_corr[1].plot(time_vals, orth_env[:plot_len], alpha=0.5, label='Orthogonalized Envelope')
ax_corr[1].plot(time_vals, low_orth_env[:plot_len], color='black', lw=2, label='Smoothed Orthogonalized Env.')
ax_corr[1].set_title(f'Target Region Y ({target.value})')
ax_corr[1].legend()

sns.despine(trim=True)
plt.tight_layout()
plt.show()

correlation = np.corrcoef(low_orth_env, low_signal_env)[0, 1]
print(f'\nOrthogonalized envelope correlation between {referenz.value} and {target.value}: {correlation:.2f}')

def calculate_orthogonalized_fc(meg_data, band, fs, low_pass_cutoff):
    """
    计算全脑范围内的正交化功能连接矩阵。

    Args:
        meg_data (xr.DataArray): MEG 数据 (time, regions)。
        band (list): 带通滤波的频率范围, e.g., [8, 12].
        fs (float): 采样率.
        low_pass_cutoff (float): 包络平滑处理的低通截止频率.

    Returns:
        np.ndarray: 对称的功能连接矩阵 (N_regions x N_regions).
    """
    print("\nCalculating full Orthogonalized Functional Connectivity matrix...")
    
    # --- 1. 数据预处理与信号准备 ---
    num_regions = meg_data.shape[1]
    
    # 步骤 a: 对所有脑区的原始信号进行 Z-score 标准化，确保结果的鲁棒性
    meg_array_mean = meg_data.mean(dim='time')
    meg_array_std = meg_data.std(dim='time')
    meg_data_normalized = (meg_data - meg_array_mean) / meg_array_std
    
    # 步骤 b: 对所有标准化后的信号进行带通滤波
    # 注意: filtfilt 的 axis=0 默认对列操作，正好适用于 (time, regions) 格式的数据
    y_filt_all = butter_bandpass_filter(meg_data_normalized.values, band[0], band[1], fs)
    
    # 步骤 c: 获取所有区域的复数解析信号和包络
    complex_all = hilbert(y_filt_all, axis=0)
    env_all = np.abs(complex_all)
    
    # 步骤 d: 预先计算共轭项以提高效率
    conj_div_env_all = complex_all.conj() / env_all
    
    # 步骤 e: 对所有原始包络进行一次性低通滤波
    low_pass_env_all = butter_lowpass_filter(env_all, highcut=low_pass_cutoff, fs=fs)

    # --- 2. 循环计算正交化与相关性 ---
    fc_matrix = np.zeros((num_regions, num_regions))
    
    progress = widgets.IntProgress(min=0, max=num_regions, description='Computing FC:',
                                   layout=widgets.Layout(width='80%'), style={'description_width': 'initial'})
    print(progress)
    
    for i in range(num_regions):
        # 将第 i 个脑区作为 "目标"
        complex_target = complex_all[:, i]
        
        # 使用广播机制一次性计算目标 `i` 相对于所有其他脑区的正交化信号
        orth_signal_imag = (complex_target[:, np.newaxis] * conj_div_env_all).imag
        
        # 获取这些正交化信号的包络（绝对值）
        orth_env = np.abs(orth_signal_imag)
        
        # 对这些正交化包络进行低通滤波
        low_pass_orth_env = butter_lowpass_filter(orth_env, highcut=low_pass_cutoff, fs=fs)
        
        # 计算N个平滑后的正交包络与N个平滑后的原始包络之间的相关性
        # .T 转置是必需的，因为 np.corrcoef 要求输入格式为 (变量数, 观测数)
        corr_mat = np.corrcoef(low_pass_orth_env.T, low_pass_env_all.T)
        
        # 从生成的 2N x 2N 矩阵中提取我们需要的 N x N 相关性部分
        corr_row = np.diag(corr_mat, k=num_regions)
        
        fc_matrix[i, :] = corr_row
        progress.value += 1
        
    # --- 3. 对矩阵进行对称化处理 ---
    # 因为 i->j 和 j->i 的连接强度可能不同，通常取平均值使其对称
    fc_matrix = (fc_matrix + fc_matrix.T) / 2
    np.fill_diagonal(fc_matrix, 0)
    
    print("FC Matrix calculation complete.")
    return fc_matrix

def plot_fc_heatmap(fc_matrix, labels, title, ax):
    """
    一个统一的热图绘制函数，确保所有FC矩阵图的风格一致。
    """
    # 计算对称的颜色范围
    vmax = np.max(np.abs(fc_matrix))
    
    # 绘制热图
    sns.heatmap(
        fc_matrix,
        square=True,
        ax=ax,
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"shrink": .8}
    )
    
    # 自动计算刻度间距，使其大约显示20-25个标签
    num_regions = len(labels)
    tick_spacing = max(1, num_regions // 20)
    
    # 清理标签（移除.L/.R后缀，如果有的话）
    cleaned_labels = [str(label).replace('.L', '').replace('.R', '') for label in labels]
    
    # 设置坐标轴
    ax.set_xticks(np.arange(0, num_regions, tick_spacing))
    ax.set_yticks(np.arange(0, num_regions, tick_spacing))
    ax.set_xticklabels(cleaned_labels[::tick_spacing], rotation=90, fontsize=8)
    ax.set_yticklabels(cleaned_labels[::tick_spacing], rotation=0, fontsize=8)
    ax.set_title(title)

# Execute the function with the parameters from the widgets
fc_matrix = calculate_orthogonalized_fc(
    meg_data=meg_array, 
    band=freq.value, 
    fs=fs, 
    low_pass_cutoff=low_pass.value
)
# --- A. 为【真实MEG数据】生成FC图 ---
print("\nPlotting FC Matrix for real MEG data (Default Order)...")
# 创建画布和坐标轴
fig_real, ax_real = plt.subplots(figsize=(10, 8))
# 调用统一的绘图函数
plot_fc_heatmap(
    fc_matrix=fc_matrix,
    labels=region_labels, # 使用真实的脑区标签
    title=f'Real Data Orthogonalized FC ({freq.value[0]}-{freq.value[1]} Hz)',
    ax=ax_real
)
plt.tight_layout()
plt.show()


print("\nPlease select parameters for the simulation FC calculation:")
sim_freq_band = widgets.IntRangeSlider(
    value=[8, 12], min=1, max=45, description='Frequency (Hz)',
    layout=widgets.Layout(width='80%'), style={'description_width': 'initial'}
)
sim_low_pass = widgets.FloatSlider(
    value=2.0, min=0, max=4.0, step=0.1, description='Low-Pass (Hz)',
    readout=True, readout_format='.1f',
    layout=widgets.Layout(width='80%'), style={'description_width': 'initial'}
)
print(sim_freq_band)
print(sim_low_pass)

fc_matrix_sim = calculate_orthogonalized_fc(
    meg_data=sim_signal,
    band=sim_freq_band.value,
    fs=target_fs,
    low_pass_cutoff=sim_low_pass.value
)

# 为模拟数据准备正确的标签
sim_labels = sim_signal.regions.values

# 创建画布和坐标轴
fig_sim, ax_sim = plt.subplots(figsize=(10, 8))
# 再次调用统一的绘图函数
plot_fc_heatmap(
    fc_matrix=fc_matrix_sim,
    labels=sim_labels, # 使用模拟数据的正确标签
    title=f'Simulated Orthogonalized FC ({sim_freq_band.value[0]}-{sim_freq_band.value[1]} Hz)',
    ax=ax_sim
)
plt.tight_layout()
plt.show()

num_sim_regions = fc_matrix_sim.shape[0]
fc_matrix_real_matched = fc_matrix[:num_sim_regions, :num_sim_regions]
print(f"\nComparing the {num_sim_regions}x{num_sim_regions} simulated FC with the corresponding subset of the real MEG FC.")

emp_fc_sub = fc_matrix_real_matched          # 真实 MEG 正交化 FC（已在上一步切好）
sc_sub     = hcp.Cmat[:emp_fc_sub.shape[0], :emp_fc_sub.shape[1]]  # 对应子矩阵 SC
sim_fc_sub = fc_matrix_sim                   # 模拟正交化 FC（维度已对齐）

# ------------------------------------------------------------------
# 2. 计算与真实 FC 的皮尔逊相关
# ------------------------------------------------------------------
struct_emp = np.corrcoef(emp_fc_sub.flatten(), sc_sub.flatten())[0, 1]
sim_emp    = np.corrcoef(emp_fc_sub.flatten(), sim_fc_sub.flatten())[0, 1]

# ------------------------------------------------------------------
# 3. 画图（与示例完全一致的风格）
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
splot = sns.barplot(
    x=['Structural Connectivity', 'Simulated Connectivity'],
    y=[struct_emp, sim_emp],
    ax=ax
)
ax.set_title('Correlation to Empirical Functional Connectivity', pad=10)

# 数值标签
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   size=20, color='white',
                   xytext=(0, -12),
                   textcoords='offset points')
sns.despine()

num_sim_regions = fc_matrix_sim.shape[0]
fc_matrix_real_matched = fc_matrix[:num_sim_regions, :num_sim_regions]
print(f"\nComparing the {num_sim_regions}x{num_sim_regions} simulated FC with the corresponding subset of the real MEG FC.")

# Flatten the upper triangle of the matrices to create 1D vectors for correlation
# (we ignore the diagonal and the redundant lower triangle)
indices_triu = np.triu_indices(num_sim_regions, k=1)
real_fc_flat = fc_matrix_real_matched[indices_triu]
sim_fc_flat = fc_matrix_sim[indices_triu]

# Calculate the Pearson correlation between the two flattened FC vectors
sim_vs_real_corr = np.corrcoef(real_fc_flat, sim_fc_flat)[0, 1]

# --- Visualize the Comparison ---
fig_comp, ax_comp = plt.subplots(figsize=(6, 6))

splot = sns.barplot(
    x=['Simulated vs. Real Data'],
    y=[sim_vs_real_corr],
    ax=ax_comp
)

ax_comp.set_ylabel('Pearson Correlation')
ax_comp.set_title('Model Fit: Correlation between Simulated and Real FC', pad=15)
ax_comp.set_ylim(0, max(0.5, sim_vs_real_corr * 1.2)) # Adjust y-axis limit

# Add the correlation value as text on the bar
for p in splot.patches:
    splot.annotate(
        format(p.get_height(), '.3f'), 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', va='center', 
        size=18, color='white',
        xytext=(0, -15), 
        textcoords='offset points'
    )

sns.despine()
plt.tight_layout()
plt.show()

print(f"The correlation between the simulated FC and the real MEG data FC is: {sim_vs_real_corr:.3f}")
