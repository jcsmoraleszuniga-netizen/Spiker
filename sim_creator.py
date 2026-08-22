import matplotlib.pyplot as plt
import numpy as np
from lib_utility import exp_decay, smoothing, vtp
from scipy.signal import bessel, filtfilt, lfilter, savgol_filter

t_delta = 0.0001  # Sampling frequency is 10kHz
sampling_f = int(1 / t_delta)


def compare_function():
    ...


def psc(time_delta=0.0001, rise_time=0.002, amplitude=-15.0, exp_dec_params=(0.0, -0.01), min_amplitude=0.01):
    # Dynamically calculate the slope required to reach the target amplitude in the fixed rise_time
    rise_slope = amplitude / rise_time
    art_event_rtime = np.arange(0, rise_time, time_delta)
    art_event_rise = rise_slope * art_event_rtime
    i0 = exp_dec_params[0]
    tau = exp_dec_params[1]
    # K0 becomes the peak amplitude actually reached by the end of the rise phase
    k0 = float(art_event_rise[-1])
    decay_time = np.round(tau * (np.log(min_amplitude) - np.log(np.abs(k0))), 3)  # time to reach min_amplitude
    art_event_dtime = np.arange(0, decay_time, time_delta)
    art_event_decay = exp_decay(art_event_dtime, i0, k0, tau)
    psc_time = np.concatenate((art_event_rtime, art_event_dtime))
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


def reconstruct_original_signal(recorded_signal, sampling_rate_hz, r_access_mohm, c_membrane_pf, smooth_derivative=True):
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


def main(total_delay=0.02, jump_delay=0.001, kernel_width1=0.05):
    plt.figure()
    delay_arr = np.arange(jump_delay, total_delay, jump_delay)
    resp_alpha = 0.5
    resp_alpha_decrement = resp_alpha / len(delay_arr)
    soft_alpha = 1.0
    soft_alpha_decrement = soft_alpha / len(delay_arr)
    kernel_sd1 = 0.0
    absolute_start_time = 0.015
    rise_time = 0.0005
    tau = -0.0025
    i0 = 0.0
    resp_amp0 = -55
    resp_amp1 = -12
    for delay in delay_arr:
        sim_events = (
                (
                        absolute_start_time,
                        psc(t_delta, rise_time, resp_amp0, (i0, tau))
                        ),
                (
                        absolute_start_time + delay,
                        psc(t_delta, rise_time, -12, (i0, tau))
                        )
                )
        # Dynamically find the absolute latest end time across ALL events
        max_end_time = 0.0
        for event in sim_events:
            event_start_time = event[0]
            event_duration = len(event[1][1]) * t_delta
            event_end_time = event_start_time + event_duration

            if event_end_time > max_end_time:
                max_end_time = event_end_time

        # Total time is the furthest point reached by any event, plus a 10ms buffer
        total_time = max_end_time + 0.01

        artificial_time = np.arange(0, total_time, t_delta)
        a_r_length = len(artificial_time)  # Dynamically pull the exact length

        # 1. Base recording (clean baseline at 48 pA)
        artificial_resp = np.full(a_r_length, 48.0)

        # 2. Add events
        for event in sim_events:
            artificial_resp = insert_event(artificial_resp, t_delta, event[0], event[1])

        # 3. Pipette filtering (RC circuit: e.g., 23.5 MOhm Ra, 5 pF effective Cm (55 pF total))
        artificial_resp = pipette_ra_filter(
                data=artificial_resp,
                sampling_rate=sampling_f,
                r_access_mohm=23.5,
                c_membrane_pf=5  # approximate to the somatic capacitance only, for mEPSCs (fast small events)
                )

        # 4. Add noise (Amplifier/Thermal noise injection)
        noise = np.random.normal(loc=0.0, scale=3.0, size=a_r_length)
        artificial_resp += noise

        # 5. Amplifier hardware filter (3 kHz Bessel)
        artificial_resp = apply_bessel_filter(
                data=artificial_resp,
                sampling_rate=sampling_f,
                cutoff_freq=3000.0
                )

        # 6. Digital smoothing algorithm
        kernel_width0 = 0.005
        points0 = vtp(kernel_width0, t_delta)
        smoothed_art_resp, kernel_sd0 = smoothing(artificial_resp, points0, sharpness=8, c_type='g', fs=sampling_f)

        plt.plot(artificial_time, artificial_resp, "r", alpha=resp_alpha, linewidth=1.0)
        plt.plot(artificial_time, smoothed_art_resp, "b", alpha=resp_alpha, linewidth=1.0)

        points1 = vtp(kernel_width1, t_delta)
        smooth_thres_art_resp, kernel_sd1 = smoothing(
                smoothed_art_resp, points1, sharpness=8, c_type='g', fs=sampling_f
                )
        plt.plot(artificial_time, smooth_thres_art_resp, "b", alpha=soft_alpha, linewidth=1.0)

        resp_alpha -= resp_alpha_decrement
        soft_alpha -= soft_alpha_decrement

        # Single trace export
        filename = (f"simulated_epsc_td{total_delay*1000}_jd{jump_delay*1000}_kw{kernel_width1*1000}"
                    f"_ksd{kernel_sd1*1000:.2f}_rt{rise_time*1000}_tau{tau*1000}_r1{resp_amp0}_r2{resp_amp1}.atf")
        save_array_as_atf(
                filename=filename,
                array=artificial_resp,
                sampling_rate_hz=sampling_f,  # 10000 Hz
                yunits="pA",
                channel_name="Trace"
                )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Current (pA)")
    plt.title(f"Artificial recording: {kernel_sd1=:3.5f} {kernel_width1=}")
    plt.show()


if __name__ == "__main__":
    main(total_delay=0.004, jump_delay=0.003, kernel_width1=0.05)
