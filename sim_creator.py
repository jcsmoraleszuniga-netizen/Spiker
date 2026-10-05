import copy

import matplotlib.pyplot as plt
import numpy as np
from lib_utility import adaptive_smoothing, differentiate, exp_decay, smoothing, vtp
from scipy.signal import bessel, correlate, filtfilt, lfilter, savgol_filter
import gc

t_delta = 0.0001  # Sampling frequency is 10kHz
sampling_f = int(1 / t_delta)


def psc(time_delta=0.0001, rise_time=0.002, amplitude=-15.0, exp_dec_params=(0.0, -0.01), min_amplitude=0.1):
    # Dynamically calculate the slope required to reach the target amplitude in the fixed rise_time
    rise_slope = amplitude / rise_time
    art_event_rtime = np.arange(0, rise_time + time_delta, time_delta)
    art_event_rise = rise_slope * art_event_rtime
    i0 = exp_dec_params[0]
    tau = exp_dec_params[1]
    # K0 becomes the peak amplitude actually reached by the end of the rise phase
    k0 = float(art_event_rise[-1])
    decay_time = np.round(tau * (np.log(np.abs(k0)) - np.log(min_amplitude)), 3)
    if decay_time <= 0:
        print(f"Wrong decay time {decay_time=}, it should be positive")
        decay_time = abs(decay_time)
    else:
        print(f"{decay_time=}")
    art_event_dtime = np.arange(0, decay_time, time_delta)
    art_event_decay = exp_decay(art_event_dtime, i0, k0, tau)
    psc_time = np.concatenate((art_event_rtime, art_event_dtime))
    print(f"{psc_time[-1]=}")
    psc_resp = np.concatenate((art_event_rise, art_event_decay))
    return np.array((psc_time, psc_resp))


def insert_event(artificial_resp, time_delta, epsc_baseline, epsc):
    """
    Inserts an event signal into the artificial response array at the calculated baseline position.
    """
    event_start_pos = int((1 / time_delta) * epsc_baseline)
    event_end_pos = event_start_pos + len(epsc[1])

    artificial_resp[event_start_pos:event_end_pos] += epsc[1]

    return artificial_resp


def apply_bessel_filter(data, sampling_rate, cutoff_freq):
    """
    Applies a 4th-order analog-style Bessel low-pass filter, mimicking
    the hardware filter in the amplifier.
    """
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = bessel(4, normalized_cutoff, btype='low', analog=False)

    return filtfilt(b, a, data)


def pipette_ra_filter(data, sampling_rate, r_access_mohm, c_membrane_pf):
    """
    Applies a true causal 1st-order RC low-pass filter mimicking
    pipette access resistance (Ra) and membrane capacitance (Cm).
    """
    r_ohms = r_access_mohm * 1e6
    c_farads = c_membrane_pf * 1e-12
    tau = r_ohms * c_farads  # RC time constant in seconds (\tau = Ra * Cm)

    dt = 1.0 / sampling_rate

    # Exact discrete 1st-order RC filter coefficients
    alpha = 1.0 - np.exp(-dt / tau)
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]

    # lfilter is strictly CAUSAL (no pre-event smoothing, introduces real physical time lag)
    return lfilter(b, a, data)


def save_array_as_atf(filename, array, sampling_rate_hz=10000, yunits="pA", channel_name="Trace"):
    dt_ms = 1000.0 / sampling_rate_hz
    time = np.arange(len(array)) * dt_ms

    header = (
            "ATF\t1.0\n"
            "1\t2\n"
            f'"SignalsExported=Time,{channel_name}"\n'
            f'"Time (ms)"\t"{channel_name} ({yunits})"\n'
    )

    combined = np.column_stack((time, array))

    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, combined, delimiter="\t", fmt="%.6f")


def reconstruct_original_signal(
        recorded_signal, sampling_rate_hz, r_access_mohm, c_membrane_pf, smooth_derivative=True
        ):
    """
    Reconstructs I_true(t) from I_recorded(t) by removing Ra*Cm low-pass filtering.

    Parameters:
    -----------
    recorded_signal : np.ndarray
        The recorded current array (pA).
    sampling_rate_hz : float
        Sampling frequency in Hz (e.g. 10000).
    r_access_mohm : float
        Access resistance in MegaOhms (e.g. 23.5).
    c_membrane_pf : float
        Membrane capacitance in picoFarads (e.g. 20.0).
    smooth_derivative : bool
        If True, applies a mild Savitzky-Golay filter to dI/dt to prevent
        high-frequency noise amplification.
    """
    # 1. Calculate time constant tau = Ra * Cm (in seconds)
    r_ohms = r_access_mohm * 1e6
    c_farads = c_membrane_pf * 1e-12
    tau = r_ohms * c_farads  # e.g., 23.5M * 20p = 0.00047 s (0.47 ms)

    dt = 1.0 / sampling_rate_hz

    # 2. Compute numerical derivative (dI/dt)
    dI_dt = np.gradient(recorded_signal, dt)

    # 3. Optional: Smooth the derivative
    # Taking a derivative amplifies high-frequency noise. A light Savitzky-Golay
    # filter removes high-frequency noise spikes without altering the underlying slope.
    if smooth_derivative:
        # Window length of ~5-9 points works well for 10kHz sampling
        window_len = 7
        poly_order = 2
        dI_dt = savgol_filter(dI_dt, window_length=window_len, polyorder=poly_order)

    # 4. Apply Traynelis inverse filter: I_true = I_rec + tau * (dI/dt)
    i_true = recorded_signal + (tau * dI_dt)

    return i_true


def main(
        kernel_width1=0.05, kernels=10, kernel_threshold_width=0.1, events=(), max_end_time=3.0, section=(2.5, 2.5075),
        initial_holding=100.0, slope_pa=-5.0, direction=-1, antipeak_factor=2.5,
        ):
    sim_events = events
    kernel_sd1 = 0.0
    s_start = vtp(section[0], t_delta)
    s_end = vtp(section[1], t_delta)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    ax1, ax2, ax3, ax4 = axes.ravel()
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Dynamically find the absolute latest end time across ALL events
    for event in sim_events:
        event_start_time = event[0]
        event_duration = event[1][0][-1]
        event_end_time = event_start_time + event_duration
        print(f"{event_start_time=} {event_duration=} {event_end_time=}")
        if event_end_time > max_end_time:
            max_end_time = event_end_time
    # Total time is the furthest point reached by any event, plus a 10ms buffer
    total_time = max_end_time + 0.01
    artificial_time = np.arange(0, total_time, t_delta)
    a_r_length = len(artificial_time)  # Dynamically pull the exact length
    # 1. Base recording
    # (clean baseline at 48 pA)
    artificial_resp = np.linspace(initial_holding, initial_holding + slope_pa, a_r_length)
    # 2. Add events
    for event in sim_events:
        artificial_resp = insert_event(artificial_resp, t_delta, event[0], event[1])
    ax1.plot(artificial_time[s_start: s_end], artificial_resp[s_start: s_end], "k:", alpha=1.0, linewidth=1.0)
    # 3. Pipette filtering (RC circuit: e.g., 23.5 MOhm Ra, 5 pF effective Cm (55 pF total))
    artificial_resp = pipette_ra_filter(
            data=artificial_resp,
            sampling_rate=sampling_f,
            r_access_mohm=23.5,
            c_membrane_pf=5  # approximate to the somatic capacitance only, for mEPSCs (fast small events)
            )
    # 4. Add noise (Amplifier/Thermal noise injection)
    noise = np.random.normal(loc=0.0, scale=2.0, size=a_r_length)
    artificial_resp += noise
    # 5. Amplifier hardware filter (3 kHz Bessel)
    artificial_resp = apply_bessel_filter(
            data=artificial_resp,
            sampling_rate=sampling_f,
            cutoff_freq=3000.0
            )
    ax1.plot(artificial_time[s_start: s_end], artificial_resp[s_start: s_end], "r:", alpha=0.75, linewidth=1.0)
    # ----------------------------------
    # Smoothing of the artificial response to remove noise.
    kernel_width_null = 0.005
    kernel_width_der = 0.001
    max_slope_ori = 20000
    min_slope_ori = 5000
    artificial_resp_ori = copy.deepcopy(artificial_resp)
    slope_clean = np.abs(differentiate(artificial_resp, t_delta))
    slope_clean, _ = smoothing(
            slope_clean, vtp(kernel_width_der, t_delta), sharpness=8, c_type="g", fs=sampling_f
            )
    artificial_resp_so, _ = smoothing(
            artificial_resp, vtp(kernel_width_null, t_delta), sharpness=8, c_type="g", fs=sampling_f
            )
    artificial_resp, weight = adaptive_smoothing(
            artificial_resp, vtp(kernel_width_null, t_delta), sharpness=8, c_type="g", fs=sampling_f,
            max_slope=max_slope_ori, min_slope=min_slope_ori, smooth_points=vtp(kernel_width_der, t_delta)
            )
    # ----------------------------------
    # Start of the Gaussian convolution

    ax1.plot(
            artificial_time[s_start: s_end], artificial_resp[s_start: s_end], "r", alpha=0.5, linewidth=2.0
            )
    ax1.plot(
            artificial_time[s_start: s_end], artificial_resp_so[s_start: s_end], "b", alpha=0.25, linewidth=1.0
            )
    smoothed_art_resp, kernel_sd = smoothing(
            artificial_resp, vtp(kernel_threshold_width, t_delta), sharpness=8, c_type="g", fs=sampling_f
            )
    ax1.plot(
            artificial_time[s_start: s_end], (smoothed_art_resp + direction*3.0)[s_start: s_end],
            "k:", alpha=0.75, linewidth=2.0
            )
    # # right top panel: Derivative |dI/dt|
    # Plot all signals on ax2
    ax2.plot(
            artificial_time[s_start: s_end], np.abs(differentiate(artificial_resp_ori[s_start: s_end], t_delta)),
            label="dI/dt Original", color="black", alpha=1.0, linewidth=2.0
            )
    ax2.plot(
            artificial_time[s_start: s_end], np.abs(differentiate(artificial_resp[s_start: s_end], t_delta)),
            label="dI/dt Filtered", color="red", alpha=0.8, linewidth=2.0
            )
    ax2.plot(
            artificial_time[s_start: s_end], slope_clean[s_start: s_end],
            label="Slope Clean", color="gray", alpha=0.6, linewidth=2.0, linestyle='--'
            )

    # Plot weight mapped to the axes vertical fraction (10% to 90% panel height)
    ax2.plot(
            artificial_time[s_start: s_end], 0.1 + 0.8 * weight[s_start: s_end],
            label="Weight", color="green", alpha=0.25, linewidth=2.0,
            transform=ax2.get_xaxis_transform(), linestyle='--'
            )
    ax2.axhline(max_slope_ori, linestyle='--')
    ax2.axhline(min_slope_ori, linestyle='--')
    # 6. Digital smoothing algorithm
    kernel_width0 = 0.001
    n_kernels = kernels
    kernel_width_arr = np.linspace(kernel_width0, kernel_width1, n_kernels)
    print(f"{kernel_width0=} {kernel_width1=} {len(kernel_width_arr)=}  {kernel_width_arr[0]=}")
    # 1. Generate all smoothed responses as a 2D array (shape: [num_kernels, time_points])
    # Don't use adapting smoothing, it needs to be the normal smoothing to preserve the differences accurately
    all_smooth = np.array(
            [
                    smoothing(artificial_resp, vtp(kw, t_delta), sharpness=8, c_type="g", fs=sampling_f)[0]
                    for kw in kernel_width_arr
                    ]
            )
    # Plotting smoothed curves and diff smoothed curves
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    for smoothed_art_resp in all_smooth[:, s_start: s_end]:
        ax1.plot(
                artificial_time[s_start: s_end], smoothed_art_resp,
                "g", alpha=0.1, linewidth=1.0
                )
        ax3.plot(
                artificial_time[s_start: s_end], artificial_resp[s_start: s_end] - smoothed_art_resp,
                "b", alpha=0.1, linewidth=1.0
                )
    # Array operations
    all_smooth = np.array(all_smooth)
    all_diff = artificial_resp - all_smooth
    smoothed_diff_std = np.std(all_diff, axis=0)
    smoothed_diff_std = smoothed_diff_std - np.min(smoothed_diff_std)
    smoothed_diff_min = np.min(all_diff, axis=0)
    smoothed_diff_max = np.max(all_diff, axis=0)
    # Smoothing of the max and min from the diff smoothed family of curves
    max_slope_diff = 7500
    min_slope_diff = 2500
    smoothed_diff_min, _ = adaptive_smoothing(
            smoothed_diff_min, vtp(kernel_width_null, t_delta), sharpness=8, c_type="g",
            fs=sampling_f, max_slope=max_slope_diff, min_slope=min_slope_diff, smooth_points=vtp(0.0075, t_delta)
            )
    smoothed_diff_max, _ = adaptive_smoothing(
            smoothed_diff_max, vtp(kernel_width_null, t_delta), sharpness=8, c_type="g",
            fs=sampling_f, max_slope=max_slope_diff, min_slope=min_slope_diff, smooth_points=vtp(0.0075, t_delta)
            )
    # Apply polarity-specific scaling to the non-dominant envelope
    if direction < 0:
        # Negative-going response (e.g., inward EPSCs)
        smoothed_diff_max = antipeak_factor * smoothed_diff_max
    else:
        # Positive-going response (e.g., outward IPSCs / APs)
        smoothed_diff_min = antipeak_factor * smoothed_diff_min
    # {smoothed_diff_min + smoothed_diff_max}: to increase the valley between events
    # {- smoothed_diff_std}: to increase the size of the peak signal
    # {* smoothed_diff_std}: to remove the noise
    transformed = ((smoothed_diff_min + smoothed_diff_max) - smoothed_diff_std) * smoothed_diff_std
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.plot(artificial_time[s_start: s_end], smoothed_diff_min[s_start: s_end], "r:", alpha=0.75, linewidth=2.0)
    ax3.plot(artificial_time[s_start: s_end], smoothed_diff_max[s_start: s_end], "r:", alpha=0.75, linewidth=2.0)
    ax3.plot(artificial_time[s_start: s_end], smoothed_diff_std[s_start: s_end], "r", alpha=0.5, linewidth=2.0)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.plot(artificial_time[s_start: s_end], transformed[s_start: s_end], "k", alpha=0.75, linewidth=2.0)
    ax4_der = ax4.twinx()
    ax4_der.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    slope_smooth, _ = smoothing(
            differentiate(transformed, t_delta), vtp(kernel_width_der, t_delta), sharpness=8, c_type="g", fs=sampling_f
            )
    ax4_der.plot(
            artificial_time[s_start: s_end], slope_smooth[s_start: s_end],
            "r", alpha=0.75, linewidth=2.0
            )
    # Single trace export
    filename_ar = f"simulated_epsc_kw{kernel_width1 * 1000}_ksd{kernel_sd1 * 1000:.3f}.atf"
    filename_ts = f"transformed_epsc_kw{kernel_width1 * 1000}_ksd{kernel_sd1 * 1000:.3f}.atf"
    save_array_as_atf(
            filename=filename_ar,
            array=artificial_resp,
            sampling_rate_hz=sampling_f,  # 10000 Hz
            yunits="pA",
            channel_name="Trace"
            )
    save_array_as_atf(
            filename=filename_ts,
            array=transformed,
            sampling_rate_hz=sampling_f,  # 10000 Hz
            yunits="pA",
            channel_name="Trace"
            )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (pA)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Slope")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Current (pA)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Current (pA)")

    # =========================================================
    # 6. ASYNC RAM CLEARING BLOCK
    # =========================================================
    def on_close(gui_event):
        gui_event.canvas.figure.clear()
        plt.close(gui_event.canvas.figure)
        gc.collect()

    fig.canvas.mpl_connect('close_event', on_close)
    plt.tight_layout()
    plt.show()
    fig.canvas.draw()


if __name__ == "__main__":
    min_amplitude = 0.001
    events_tuple0 = (
            (1.100, psc(t_delta, 0.0005, -5, (0.0, 0.0025), min_amplitude)),
            (1.103, psc(t_delta, 0.0005, -5, (0.0, 0.0025), min_amplitude)),
            (1.250, psc(t_delta, 0.0005, -10, (0.0, 0.0025), min_amplitude)),
            (1.253, psc(t_delta, 0.0005, -10, (0.0, 0.0025), min_amplitude)),
            (1.500, psc(t_delta, 0.0005, -20, (0.0, 0.0025), min_amplitude)),
            (1.503, psc(t_delta, 0.0005, -20, (0.0, 0.0025), min_amplitude)),
            (2.000, psc(t_delta, 0.0060, -20, (0.0, 0.0150), min_amplitude)),
            (2.012, psc(t_delta, 0.0060, -20, (0.0, 0.0150), min_amplitude)),
            (2.550, psc(t_delta, 0.0120, -20, (0.0, 0.0300), min_amplitude)),
            (2.570, psc(t_delta, 0.0120, -20, (0.0, 0.0300), min_amplitude)),
            (3.500, psc(t_delta, 0.0005, -50, (0.0, 0.0025), min_amplitude)),
            (3.503, psc(t_delta, 0.0005, -50, (0.0, 0.0025), min_amplitude)),
            )
    events_tuple1 = (
            (1.010, psc(t_delta, 0.0005, -10, (0.0, 0.0025), min_amplitude)),
            (1.013, psc(t_delta, 0.0005, -10, (0.0, 0.0025), min_amplitude)),
            )
    events_tuple2 = (
            (1.110, psc(t_delta, 0.0005, -50, (0.0, 0.0025), min_amplitude)),
            (1.113, psc(t_delta, 0.0005, -50, (0.0, 0.0025), min_amplitude)),
            (1.180, psc(t_delta, 0.001, -50, (0.0, 0.005), min_amplitude)),
            (1.183, psc(t_delta, 0.001, -50, (0.0, 0.005), min_amplitude)),
            )
    events_tuple3 = (
            (2.500, psc(t_delta, 0.0005, -50, (0.0, 0.0025), min_amplitude)),
            (2.503, psc(t_delta, 0.0005, -50, (0.0, 0.0025), min_amplitude)),
            )
    events_tuple4 = (
            (1.100, psc(t_delta, 0.0005, -5, (0.0, 0.0025), min_amplitude)),
            (1.250, psc(t_delta, 0.0005, -10, (0.0, 0.0025), min_amplitude)),
            (1.500, psc(t_delta, 0.0005, -20, (0.0, 0.0025), min_amplitude)),
            (2.000, psc(t_delta, 0.0060, -20, (0.0, 0.0150), min_amplitude)),
            (2.550, psc(t_delta, 0.0120, -20, (0.0, 0.0300), min_amplitude)),
            (3.500, psc(t_delta, 0.0005, -50, (0.0, 0.0025), min_amplitude)),
            )
    events_tuple5 = (
            (1.110, psc(t_delta, 0.0005, -5, (0.0, 0.0025), min_amplitude)),
            (1.113, psc(t_delta, 0.0005, -5, (0.0, 0.0025), min_amplitude)),
            (1.200, psc(t_delta, 0.0005, -10, (0.0, 0.0025), min_amplitude)),
            (1.203, psc(t_delta, 0.0005, -10, (0.0, 0.0025), min_amplitude)),
            (1.290, psc(t_delta, 0.0005, -50, (0.0, 0.0025), min_amplitude)),
            (1.293, psc(t_delta, 0.0005, -50, (0.0, 0.0025), min_amplitude)),
            )
    events_tuple6 = (
            (1.100, psc(t_delta, 0.0005, 5, (0.0, 0.0025), min_amplitude)),
            (1.103, psc(t_delta, 0.0005, 5, (0.0, 0.0025), min_amplitude)),
            (1.250, psc(t_delta, 0.0005, 10, (0.0, 0.0025), min_amplitude)),
            (1.253, psc(t_delta, 0.0005, 10, (0.0, 0.0025), min_amplitude)),
            (1.500, psc(t_delta, 0.0005, 20, (0.0, 0.0025), min_amplitude)),
            (1.503, psc(t_delta, 0.0005, 20, (0.0, 0.0025), min_amplitude)),
            (2.000, psc(t_delta, 0.0060, 20, (0.0, 0.0150), min_amplitude)),
            (2.012, psc(t_delta, 0.0060, 20, (0.0, 0.0150), min_amplitude)),
            (2.550, psc(t_delta, 0.0120, 20, (0.0, 0.0300), min_amplitude)),
            (2.570, psc(t_delta, 0.0120, 20, (0.0, 0.0300), min_amplitude)),
            (3.500, psc(t_delta, 0.0005, 50, (0.0, 0.0025), min_amplitude)),
            (3.503, psc(t_delta, 0.0005, 50, (0.0, 0.0025), min_amplitude)),
            )
    # events_tuple = events_tuple0  # Overlapping events, different amplitude and kinetics
    # events_tuple = events_tuple1  # Overlapping events, two only, standard
    # events_tuple = events_tuple2  # Overlapping events, same amplitude, but different kinetics
    # events_tuple = events_tuple3  # Overlapping events, two only, high amplitude
    # events_tuple = events_tuple4  # Single events, different amplitude and kinetics
    # events_tuple = events_tuple5  # Overlapping events, different amplitude, but same kinetics
    events_tuple = events_tuple6  # Overlapping events, different amplitude and kinetics, positive going
    main(
            kernel_width1=0.75,
            kernels=100,
            kernel_threshold_width=0.125,
            events=events_tuple,
            max_end_time=10.0,
            section=(0.0, 5.0),
            # section=(0.09, 0.12),
            # section=(1.52, 1.62),
            # section=(2.4825, 2.5003),
            # section=(2.4825, 2.5075),
            # section=(2.5, 2.5075),
            # section=(2.5003, 2.5075),
            # section=(2.5025, 2.5033),
            initial_holding=100.0,
            slope_pa=0.0,
            direction=1,
            antipeak_factor=3.5,
            )
