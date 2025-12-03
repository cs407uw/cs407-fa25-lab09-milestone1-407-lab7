"""
CS407 Lab 9 – Milestone 2 (Alt Version)
End-to-end processing and visualization of motion sensor data from:

    • ACCELERATION.csv
    • WALKING.csv
    • TURNING.csv
    • WALKING_AND_TURNING.csv

Overall flow:

  1. Compare ideal vs noisy acceleration and show how errors propagate
     into speed and distance.
  2. Estimate step count from accelerometer magnitude while walking.
  3. Identify ~90° turns using gyroscope readings.
  4. Fuse detected steps and turns to approximate a 2D walking path.

Run this file from the same directory as the four CSVs:

    python milestone2_analysis_refactored.py
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Generic helper functions
# ============================================================

def integrate_trap(values: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """
    Integrate a 1D signal using the trapezoidal rule.

    Parameters
    ----------
    values : np.ndarray
        Samples of the function f(t).
    dt : np.ndarray
        Time increment array, where dt[i] corresponds to the step between
        sample i-1 and i.

    Returns
    -------
    np.ndarray
        Cumulative integral, same length as `values`.
    """
    result = np.zeros_like(values, dtype=float)
    for i in range(1, len(values)):
        avg_val = 0.5 * (values[i] + values[i - 1])
        result[i] = result[i - 1] + avg_val * dt[i]
    return result


def exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    First-order exponential smoothing / low-pass filter.

    Parameters
    ----------
    x : np.ndarray
        Input samples.
    alpha : float
        Smoothing factor in [0, 1]. Higher alpha → more reactive,
        lower alpha → smoother.

    Returns
    -------
    np.ndarray
        Filtered output sequence.
    """
    s = np.zeros_like(x, dtype=float)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1.0 - alpha) * s[i - 1]
    return s


# ============================================================
# Part 1 – Accumulation of error in acceleration data
# ============================================================

def part1_acceleration():
    """
    Part 1:
      - Load ACCELERATION.csv
      - Integrate both the “true” and “noisy” accelerations to obtain
        velocity and then distance
      - Save three PNGs:

            part1_acceleration.png
            part1_speed.png
            part1_distance.png

      - Print the final distances along with their difference.
    """
    df = pd.read_csv("ACCELERATION.csv")

    t = df["timestamp"].values.astype(float)           # seconds
    a_true = df["acceleration"].values.astype(float)
    a_noisy = df["noisyacceleration"].values.astype(float)

    # Sampling should be roughly uniform (~0.1 s), but we still compute dt.
    dt = np.diff(t, prepend=t[0])

    # Acceleration → velocity → distance via numerical integration.
    v_true = integrate_trap(a_true, dt)
    v_noisy = integrate_trap(a_noisy, dt)

    s_true = integrate_trap(v_true, dt)
    s_noisy = integrate_trap(v_noisy, dt)

    final_dist_true = float(s_true[-1])
    final_dist_noisy = float(s_noisy[-1])
    dist_error = final_dist_noisy - final_dist_true

    # --- Plot acceleration traces ---
    plt.figure()
    plt.plot(t, a_true, label="Ground truth acceleration")
    plt.plot(t, a_noisy, label="Measured (noisy) acceleration")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title("Part 1 – True vs Noisy Acceleration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part1_acceleration.png")
    plt.close()

    # --- Plot velocities ---
    plt.figure()
    plt.plot(t, v_true, label="Velocity from true a(t)")
    plt.plot(t, v_noisy, label="Velocity from noisy a(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Part 1 – Speed Over Time (True vs Noisy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part1_speed.png")
    plt.close()

    # --- Plot distances ---
    plt.figure()
    plt.plot(t, s_true, label="Position from true a(t)")
    plt.plot(t, s_noisy, label="Position from noisy a(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.title("Part 1 – Distance Estimates vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part1_distance.png")
    plt.close()

    print("=== Part 1: ACCELERATION.csv ===")
    print(f"Final distance (true acceleration):   {final_dist_true:.4f} m")
    print(f"Final distance (noisy acceleration):  {final_dist_noisy:.4f} m")
    print(f"Offset (noisy − true):                {dist_error:.4f} m\n")

    return final_dist_true, final_dist_noisy, dist_error


# ============================================================
# Part 2 – Step detection using accelerometer magnitude
# ============================================================

def detect_steps_from_magnitude(
    t_ns: np.ndarray,
    mag_smooth: np.ndarray,
    threshold: float,
    min_step_interval: float = 0.4,
):
    """
    Find footsteps by examining peaks in smoothed acceleration magnitude.

    A sample i is marked as a step if:
      • mag_smooth[i] is a local maximum,
      • mag_smooth[i] exceeds `threshold`,
      • at least `min_step_interval` seconds have passed since the
        previously detected step.

    Parameters
    ----------
    t_ns : np.ndarray
        Timestamps in nanoseconds.
    mag_smooth : np.ndarray
        Smoothed magnitude |a|.
    threshold : float
        Minimum amplitude considered to be a valid step.
    min_step_interval : float
        Minimum time gap (in seconds) between consecutive steps.

    Returns
    -------
    step_indices : np.ndarray
        Indices where steps were detected.
    t_sec : np.ndarray
        Time array converted to seconds since start.
    """
    t_sec = (t_ns - t_ns[0]) / 1e9
    step_indices = []
    last_step_time = t_sec[0]

    for i in range(1, len(mag_smooth) - 1):
        is_local_peak = (mag_smooth[i] > mag_smooth[i - 1]
                         and mag_smooth[i] > mag_smooth[i + 1])
        passes_threshold = mag_smooth[i] > threshold
        gap_ok = (t_sec[i] - last_step_time) >= min_step_interval

        if is_local_peak and passes_threshold and gap_ok:
            step_indices.append(i)
            last_step_time = t_sec[i]

    return np.array(step_indices, dtype=int), t_sec


def part2_step_detection():
    """
    Part 2:
      - Load WALKING.csv
      - Compute acceleration magnitude sqrt(ax² + ay² + az²)
      - Apply exponential smoothing (alpha = 0.1)
      - Use a threshold of (mean + 1.0) and a local-max test to find steps
      - Save the plot as: part2_walking_mag_steps.png
      - Print the total number of detected steps.
    """
    df = pd.read_csv("WALKING.csv")

    t_ns = df["timestamp"].values.astype(np.float64)
    ax = df["accel_x"].values.astype(float)
    ay = df["accel_y"].values.astype(float)
    az = df["accel_z"].values.astype(float)

    mag = np.sqrt(ax**2 + ay**2 + az**2)

    alpha = 0.1
    mag_smooth = exp_smooth(mag, alpha)

    threshold = mag_smooth.mean() + 1.0
    min_step_interval = 0.4
    step_indices, t_sec = detect_steps_from_magnitude(
        t_ns,
        mag_smooth,
        threshold,
        min_step_interval=min_step_interval,
    )

    plt.figure()
    plt.plot(t_sec, mag, label="Raw |a|")
    plt.plot(t_sec, mag_smooth, label="Smoothed |a|")
    plt.scatter(
        t_sec[step_indices],
        mag_smooth[step_indices],
        marker="o",
        label="Detected steps",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration magnitude [m/s²]")
    plt.title("Part 2 – Walking: Magnitude and Step Detections")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part2_walking_mag_steps.png")
    plt.close()

    print("=== Part 2: WALKING.csv ===")
    print(f"Estimated number of steps: {len(step_indices)} (around 37 expected)\n")

    return step_indices, t_sec, mag_smooth


# ============================================================
# Part 3 – Turn detection using gyro_z
# ============================================================

def detect_turns_from_gyro(
    t_ns: np.ndarray,
    gyro_z_smooth: np.ndarray,
    vel_threshold: float = 0.3,
):
    """
    Segment turning motions by integrating gyro_z whenever the angular
    speed is large enough.

    A turn is defined as a contiguous region where |gyro_z| > vel_threshold.
    For each such region we integrate gyro_z to estimate the net rotation.

    Parameters
    ----------
    t_ns : np.ndarray
        Timestamps in nanoseconds.
    gyro_z_smooth : np.ndarray
        Smoothed angular rate about the z-axis (rad/s).
    vel_threshold : float
        Angular velocity cutoff used to decide if the user is rotating.

    Returns
    -------
    list of (int, int, float)
        Each entry is (start_idx, end_idx, angle_rad).
    """
    dt = np.diff(t_ns, prepend=t_ns[0]) / 1e9

    turns = []
    in_turn = False
    start_idx = None
    accumulated_angle = 0.0

    for i in range(len(gyro_z_smooth)):
        spinning_fast = abs(gyro_z_smooth[i]) > vel_threshold

        if spinning_fast:
            if not in_turn:
                in_turn = True
                start_idx = i
                accumulated_angle = 0.0
            accumulated_angle += gyro_z_smooth[i] * dt[i]
        else:
            if in_turn:
                end_idx = i
                turns.append((start_idx, end_idx, accumulated_angle))
                in_turn = False
                accumulated_angle = 0.0

    # If we were still in a turn at the final sample, close it out.
    if in_turn:
        end_idx = len(gyro_z_smooth) - 1
        turns.append((start_idx, end_idx, accumulated_angle))

    return turns


def part3_direction_detection():
    """
    Part 3:
      - Load TURNING.csv
      - Strip stray trailing commas so pandas can read the file
      - Smooth gyro_z (alpha = 0.1)
      - Declare a turn when |gyro_z| > 0.3 rad/s and integrate to
        estimate total angle
      - Summarize each turn in degrees (≈ ±90°)
      - Save: part3_turning_gyro.png
    """
    # Some datasets include an extra comma at the end of certain lines.
    # Here we remove one trailing comma per line before parsing.
    with open("TURNING.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    cleaned = [line.rstrip("\n").rstrip(",") + "\n" for line in lines]

    df = pd.read_csv(io.StringIO("".join(cleaned)))

    t_ns = df["timestamp"].values.astype(np.float64)
    gyro_z = df["gyro_z"].values.astype(float)

    alpha = 0.1
    gyro_z_smooth = exp_smooth(gyro_z, alpha)
    t_sec = (t_ns - t_ns[0]) / 1e9

    # Plot raw vs smoothed gyro_z
    plt.figure()
    plt.plot(t_sec, gyro_z, label="Raw gyro_z")
    plt.plot(t_sec, gyro_z_smooth, label="Smoothed gyro_z")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular velocity [rad/s]")
    plt.title("Part 3 – TURNING.csv: Gyro_z Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part3_turning_gyro.png")
    plt.close()

    vel_threshold = 0.3
    turns = detect_turns_from_gyro(t_ns, gyro_z_smooth, vel_threshold)

    summaries = []
    print("=== Part 3: TURNING.csv ===")
    for start_idx, end_idx, angle_rad in turns:
        angle_deg = angle_rad * 180.0 / np.pi
        direction = "CW" if angle_deg < 0 else "CCW"
        start_time = (t_ns[start_idx] - t_ns[0]) / 1e9
        end_time = (t_ns[end_idx] - t_ns[0]) / 1e9
        summaries.append(
            {
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "angle_deg": angle_deg,
                "direction": direction,
            }
        )
        print(
            f"Turn from t = {start_time:.3f}s to t = {end_time:.3f}s: "
            f"{angle_deg:.2f}° ({direction})"
        )
    print()

    return summaries, t_sec, gyro_z_smooth


# ============================================================
# Part 4 – Approximate 2D trajectory (steps + turns)
# ============================================================

def part4_trajectory():
    """
    Part 4:
      - Load WALKING_AND_TURNING.csv (which has a slightly odd layout)
      - Treat the row index as the timestamp in nanoseconds
      - Remap the remaining columns to accel / gyro channels
      - Use accel magnitude to find steps and gyro_z to locate turns
      - Only keep turns whose magnitude exceeds 30°
      - Assume 1 m per step and integrate heading to get a rough path
      - Save: part4_trajectory.png
    """
    # WALKING_AND_TURNING.csv uses an unusual column configuration, so
    # we treat the index as time and reinterpret the fields by inspection.
    raw = pd.read_csv("WALKING_AND_TURNING.csv", engine="python")
    df = raw.copy()

    # Row index as timestamps (ns), then convert to seconds.
    t_ns = df.index.values.astype(np.float64)
    t_sec = (t_ns - t_ns[0]) / 1e9

    # Reassign columns to accelerometer and gyro channels.
    accel_x = df["timestamp"].values.astype(float)
    accel_y = df["accel_x"].values.astype(float)
    accel_z = df["accel_y"].values.astype(float)
    gyro_x = df["accel_z"].values.astype(float)
    gyro_y = df["gyro_x"].values.astype(float)
    gyro_z = df["gyro_y"].values.astype(float)

    # ----- Step detection via magnitude (similar to Part 2) -----
    mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    mag_smooth = exp_smooth(mag, alpha=0.1)

    step_threshold = mag_smooth.mean() + 1.0
    step_indices, t_sec = detect_steps_from_magnitude(
        t_ns,
        mag_smooth,
        threshold=step_threshold,
        min_step_interval=0.4,
    )

    # ----- Turn detection via gyro_z (similar to Part 3) -----
    gyro_z_smooth = exp_smooth(gyro_z, alpha=0.1)

    # Optional reference plot for gyro_z while walking & turning.
    plt.figure()
    plt.plot(t_sec, gyro_z, label="Raw gyro_z")
    plt.plot(t_sec, gyro_z_smooth, label="Smoothed gyro_z")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular velocity [rad/s]")
    plt.title("Part 4 – WALKING_AND_TURNING.csv: Gyro_z Trace")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part4_walkturn_gyro.png")
    plt.close()

    all_turns = detect_turns_from_gyro(t_ns, gyro_z_smooth, vel_threshold=0.3)

    # Keep only substantial turns (rotate at least ±30°).
    significant_turns = []
    for start_idx, end_idx, angle_rad in all_turns:
        angle_deg = angle_rad * 180.0 / np.pi
        if abs(angle_deg) > 30.0:
            direction = "CW" if angle_deg < 0 else "CCW"
            significant_turns.append((start_idx, end_idx, angle_deg, direction))

    # ----- Build a single time-ordered event stream -----
    # Events are either "step" (change position) or "turn" (change heading).
    events = []  # (time_sec, type, index, value)
    for idx in step_indices:
        events.append((t_sec[idx], "step", idx, None))

    for start_idx, end_idx, angle_deg, _direction in significant_turns:
        mid_idx = (start_idx + end_idx) // 2
        events.append((t_sec[mid_idx], "turn", mid_idx, angle_deg))

    events.sort(key=lambda e: e[0])

    # ----- Propagate a simple 2D pose estimate -----
    step_length = 1.0  # meters per detected step
    x_positions = [0.0]
    y_positions = [0.0]
    heading_deg = 0.0  # 0° = facing up the y-axis (“north”)

    for _t, event_type, _idx, value in events:
        if event_type == "turn":
            # value is the signed rotation in degrees.
            heading_deg += value
        elif event_type == "step":
            heading_rad = np.deg2rad(heading_deg)
            x_next = x_positions[-1] + step_length * np.sin(heading_rad)
            y_next = y_positions[-1] + step_length * np.cos(heading_rad)
            x_positions.append(x_next)
            y_positions.append(y_next)

    # Draw the reconstructed path.
    plt.figure()
    plt.plot(x_positions, y_positions, marker="o")
    plt.xlabel("X [m] (east)")
    plt.ylabel("Y [m] (north)")
    plt.title("Part 4 – Approximate Walking Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("part4_trajectory.png")
    plt.close()

    print("=== Part 4: WALKING_AND_TURNING.csv ===")
    print(f"Steps detected: {len(step_indices)}")
    print(f"Large turns (|angle| > 30°): {len(significant_turns)}")
    for start_idx, end_idx, angle_deg, direction in significant_turns:
        mid_idx = (start_idx + end_idx) // 2
        mid_time = (t_ns[mid_idx] - t_ns[0]) / 1e9
        print(f"  Turn around t = {mid_time:.2f}s: {angle_deg:.2f}° ({direction})")
    print(f"Final heading estimate: {heading_deg:.2f}°")
    print(f"Final position estimate: x = {x_positions[-1]:.2f} m, "
          f"y = {y_positions[-1]:.2f} m\n")


# ============================================================
# Main entry point
# ============================================================

def main():
    print("Starting Milestone 2 sensor analysis script...\n")
    part1_acceleration()
    part2_step_detection()
    part3_direction_detection()
    part4_trajectory()
    print("Processing complete. All figures have been saved as PNG files.")


if __name__ == "__main__":
    main()
