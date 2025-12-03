
"""
CS407 Lab 9 - Milestone 2: Sensor Processing
Analysis script for ACCELERATION.csv, WALKING.csv, TURNING.csv,
and WALKING_AND_TURNING.csv.

This script:
  * Part 1: Computes and plots acceleration, speed, and distance
            for true vs noisy acceleration.
  * Part 2: Detects steps from WALKING.csv and plots the raw + smoothed
            acceleration magnitude with detected steps.
  * Part 3: Detects 90-degree turns from TURNING.csv using gyro_z.
  * Part 4: Detects steps + turns from WALKING_AND_TURNING.csv and
            reconstructs the walking trajectory.

Place this file in the same directory as the 4 CSVs and run:
    python milestone2_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io


# -------------------------
# Helper functions
# -------------------------

def integrate_trap(values: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """
    Integrate a 1D time series using the trapezoidal rule.
    Assumes dt[i] is the time step between sample i-1 and i.
    Returns the integrated series (same length as input).
    """
    out = np.zeros_like(values, dtype=float)
    for i in range(1, len(values)):
        out[i] = out[i-1] + 0.5 * (values[i] + values[i-1]) * dt[i]
    return out


def exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Simple exponential smoothing (first-order IIR):
        s[0] = x[0]
        s[i] = alpha * x[i] + (1 - alpha) * s[i-1]
    """
    s = np.zeros_like(x, dtype=float)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i-1]
    return s


# -------------------------
# Part 1: Sensor data errors
# -------------------------

def part1_acceleration():
    """
    Part 1:
      * Read ACCELERATION.csv
      * Compute speeds and distances for true vs noisy acceleration
      * Save 3 plots:
          - part1_acceleration.png
          - part1_speed.png
          - part1_distance.png
      * Print final distances and their difference.
    """
    df = pd.read_csv("ACCELERATION.csv")
    t = df["timestamp"].values           # seconds
    a_true = df["acceleration"].values
    a_noisy = df["noisyacceleration"].values

    # Time step (regular at 0.1 s)
    dt = np.diff(t, prepend=t[0])

    # Integrate acceleration -> velocity -> distance
    v_true = integrate_trap(a_true, dt)
    v_noisy = integrate_trap(a_noisy, dt)

    s_true = integrate_trap(v_true, dt)
    s_noisy = integrate_trap(v_noisy, dt)

    final_dist_true = float(s_true[-1])
    final_dist_noisy = float(s_noisy[-1])
    diff_dist = final_dist_noisy - final_dist_true

    # --- Plots ---

    # 1) acceleration vs noisy acceleration
    plt.figure()
    plt.plot(t, a_true, label="Actual acceleration")
    plt.plot(t, a_noisy, label="Noisy acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.title("Acceleration vs Noisy Acceleration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part1_acceleration.png")
    plt.close()

    # 2) speed vs time
    plt.figure()
    plt.plot(t, v_true, label="Speed from actual acceleration")
    plt.plot(t, v_noisy, label="Speed from noisy acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part1_speed.png")
    plt.close()

    # 3) distance vs time
    plt.figure()
    plt.plot(t, s_true, label="Distance from actual acceleration")
    plt.plot(t, s_noisy, label="Distance from noisy acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Distance vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part1_distance.png")
    plt.close()

    print("=== Part 1: ACCELERATION.csv ===")
    print(f"Final distance (actual acceleration): {final_dist_true:.4f} m")
    print(f"Final distance (noisy acceleration):  {final_dist_noisy:.4f} m")
    print(f"Difference (noisy - actual):          {diff_dist:.4f} m")
    print()

    return final_dist_true, final_dist_noisy, diff_dist


# -------------------------
# Part 2: Step detection
# -------------------------

def detect_steps_from_magnitude(
    t_ns: np.ndarray,
    mag_smooth: np.ndarray,
    threshold: float,
    min_step_interval: float = 0.4
):
    """
    Detect steps from smoothed acceleration magnitude.

    We mark a step at index i if:
      * mag_smooth[i] is a local maximum (greater than neighbors), AND
      * mag_smooth[i] > threshold, AND
      * at least min_step_interval seconds since the last step.
    """
    t_sec = (t_ns - t_ns[0]) / 1e9
    step_indices = []
    last_step_time = t_sec[0]

    for i in range(1, len(mag_smooth) - 1):
        if (
            mag_smooth[i] > threshold
            and mag_smooth[i] > mag_smooth[i - 1]
            and mag_smooth[i] > mag_smooth[i + 1]
        ):
            if t_sec[i] - last_step_time >= min_step_interval:
                step_indices.append(i)
                last_step_time = t_sec[i]

    return np.array(step_indices, dtype=int), t_sec


def part2_step_detection():
    """
    Part 2:
      * Use WALKING.csv
      * Compute acceleration magnitude sqrt(ax^2 + ay^2 + az^2)
      * Smooth with exponential smoothing (alpha = 0.1)
      * Detect steps using local maxima above threshold = mean + 1.0
      * Save plot: part2_walking_mag_steps.png
      * Print total step count.
    """
    df = pd.read_csv("WALKING.csv")

    t_ns = df["timestamp"].values.astype(np.float64)
    ax = df["accel_x"].values
    ay = df["accel_y"].values
    az = df["accel_z"].values

    mag = np.sqrt(ax**2 + ay**2 + az**2)

    # Smooth the magnitude
    alpha = 0.1
    mag_smooth = exp_smooth(mag, alpha)

    # Threshold and step detection
    threshold = mag_smooth.mean() + 1.0
    min_step_interval = 0.4  # seconds
    step_indices, t_sec = detect_steps_from_magnitude(
        t_ns, mag_smooth, threshold, min_step_interval
    )

    # Plot raw vs smoothed with detected steps
    plt.figure()
    plt.plot(t_sec, mag, label="Raw accel magnitude")
    plt.plot(t_sec, mag_smooth, label="Smoothed accel magnitude")
    plt.scatter(t_sec[step_indices], mag_smooth[step_indices], marker="o", label="Detected steps")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration magnitude (m/s^2)")
    plt.title("WALKING.csv: Raw vs Smoothed Accel Magnitude with Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part2_walking_mag_steps.png")
    plt.close()

    print("=== Part 2: WALKING.csv ===")
    print(f"Detected steps: {len(step_indices)} (expected around 37)")
    print()

    return step_indices, t_sec, mag_smooth


# -------------------------
# Part 3: Direction detection
# -------------------------

def detect_turns_from_gyro(
    t_ns: np.ndarray,
    gyro_z_smooth: np.ndarray,
    vel_threshold: float = 0.3
):
    """
    Detect turns by integrating smoothed gyro_z when |gyro_z| exceeds
    a velocity threshold. Each continuous region above threshold is
    one turn. Returns list of (start_idx, end_idx, angle_rad).
    """
    dt = np.diff(t_ns, prepend=t_ns[0]) / 1e9

    turns = []
    in_turn = False
    start_idx = None
    current_angle = 0.0

    for i in range(len(gyro_z_smooth)):
        if abs(gyro_z_smooth[i]) > vel_threshold:
            if not in_turn:
                in_turn = True
                start_idx = i
                current_angle = 0.0
            # integrate
            current_angle += gyro_z_smooth[i] * dt[i]
        else:
            if in_turn:
                end_idx = i
                in_turn = False
                turns.append((start_idx, end_idx, current_angle))
                current_angle = 0.0

    # If still in turn at end
    if in_turn:
        end_idx = len(gyro_z_smooth) - 1
        turns.append((start_idx, end_idx, current_angle))

    return turns


def part3_direction_detection():
    """
    Part 3:
      * Use TURNING.csv
      * Smooth gyro_z with exponential smoothing (alpha = 0.1)
      * Detect turns when |gyro_z| > 0.3 rad/s
      * Integrate to get angle in degrees
      * Print list of ~8 turns (around +/-90 degrees)
      * Save plot: part3_turning_gyro.png
    """
    # Some rows in TURNING.csv have an extra trailing comma which creates
    # an extra empty field and causes pandas' parser to fail. Read the
    # file as text, strip a single trailing comma from each line, and
    # then let pandas parse from the cleaned text buffer.
    with open("TURNING.csv", "r", encoding="utf-8") as fh:
        raw_lines = fh.readlines()
    cleaned_lines = [ln.rstrip('\n').rstrip(',') + '\n' for ln in raw_lines]
    df = pd.read_csv(io.StringIO(''.join(cleaned_lines)))

    t_ns = df["timestamp"].values.astype(np.float64)
    gyro_z = df["gyro_z"].values.astype(float)

    # Smooth gyro_z
    alpha = 0.1
    gyro_z_smooth = exp_smooth(gyro_z, alpha)

    t_sec = (t_ns - t_ns[0]) / 1e9

    # Plot raw vs smoothed gyro_z
    plt.figure()
    plt.plot(t_sec, gyro_z, label="Raw gyro_z")
    plt.plot(t_sec, gyro_z_smooth, label="Smoothed gyro_z")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title("TURNING.csv: Raw vs Smoothed gyro_z")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part3_turning_gyro.png")
    plt.close()

    # Detect turns
    vel_threshold = 0.3  # rad/s
    turns = detect_turns_from_gyro(t_ns, gyro_z_smooth, vel_threshold)

    # Summarize
    summaries = []
    for (start_idx, end_idx, angle_rad) in turns:
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

    print("=== Part 3: TURNING.csv ===")
    for s in summaries:
        print(
            f"Turn from {s['start_time_sec']:.3f}s to {s['end_time_sec']:.3f}s: "
            f"{s['angle_deg']:.2f} degrees, {s['direction']}"
        )
    print()

    return summaries, t_sec, gyro_z_smooth


# -------------------------
# Part 4: Trajectory plotting
# -------------------------

def part4_trajectory():
    """
    Part 4:
      * Use WALKING_AND_TURNING.csv
      * Reinterpret the columns based on the file's structure
      * Smooth accel magnitude, detect steps
      * Smooth gyro_z, detect larger turns (> 30 deg)
      * Simulate trajectory (1 m per step, heading changes at turns)
      * Save trajectory plot: part4_trajectory.png
    """
    # The WALKING_AND_TURNING.csv file in the provided dataset has
    # an unusual structure where the timestamp values appear as the
    # index when read with pandas. We read it normally, then treat
    # the index as the real timestamps and reinterpret the remaining
    # columns as sensor readings.
    raw = pd.read_csv("WALKING_AND_TURNING.csv", engine="python")
    df = raw.copy()

    # Use the index as the timestamp in ns
    t_ns = df.index.values.astype(np.float64)

    # Reinterpret columns to be accelerations and gyros
    # based on inspection of the file:
    accel_x = df["timestamp"].values.astype(float)
    accel_y = df["accel_x"].values.astype(float)
    accel_z = df["accel_y"].values.astype(float)
    gyro_x = df["accel_z"].values.astype(float)
    gyro_y = df["gyro_x"].values.astype(float)
    gyro_z = df["gyro_y"].values.astype(float)

    # Time in seconds from start
    t_sec = (t_ns - t_ns[0]) / 1e9

    # ---- Step detection (similar to Part 2) ----
    mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    alpha = 0.1
    mag_smooth = exp_smooth(mag, alpha)

    threshold_steps = mag_smooth.mean() + 1.0
    min_step_interval = 0.4
    step_indices, t_sec = detect_steps_from_magnitude(
        t_ns, mag_smooth, threshold_steps, min_step_interval
    )

    # ---- Turn detection (similar to Part 3) ----
    alpha_g = 0.1
    gyro_z_smooth = exp_smooth(gyro_z, alpha_g)

    # (Optional) save a reference plot of gyro_z
    plt.figure()
    plt.plot(t_sec, gyro_z, label="Raw gyro_z")
    plt.plot(t_sec, gyro_z_smooth, label="Smoothed gyro_z")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title("WALKING_AND_TURNING.csv: gyro_z")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part4_walkturn_gyro.png")
    plt.close()

    # Detect all turn segments
    vel_threshold = 0.3
    all_turns = detect_turns_from_gyro(t_ns, gyro_z_smooth, vel_threshold)

    # Keep only significant turns (|angle| > 30 degrees)
    significant_turns = []
    for (start_idx, end_idx, angle_rad) in all_turns:
        angle_deg = angle_rad * 180.0 / np.pi
        if abs(angle_deg) > 30.0:
            direction = "CW" if angle_deg < 0 else "CCW"
            significant_turns.append((start_idx, end_idx, angle_deg, direction))

    # ---- Build combined event list (steps + turns) ----
    events = []  # (time, type, index, value)
    for idx in step_indices:
        events.append((t_sec[idx], "step", idx, None))

    for (s_idx, e_idx, angle_deg, direction) in significant_turns:
        mid_idx = (s_idx + e_idx) // 2
        events.append((t_sec[mid_idx], "turn", mid_idx, angle_deg))

    # Sort events by time
    events.sort(key=lambda x: x[0])

    # ---- Simulate trajectory ----
    step_length = 1.0  # meters
    x_positions = [0.0]
    y_positions = [0.0]
    heading_deg = 0.0  # 0 deg means facing "north" (positive y-axis)

    for (time_val, etype, idx, value) in events:
        if etype == "turn":
            # value is signed angle in degrees
            heading_deg += value
        elif etype == "step":
            heading_rad = np.deg2rad(heading_deg)
            x_new = x_positions[-1] + step_length * np.sin(heading_rad)  # east
            y_new = y_positions[-1] + step_length * np.cos(heading_rad)  # north
            x_positions.append(x_new)
            y_positions.append(y_new)

    # Plot trajectory
    plt.figure()
    plt.plot(x_positions, y_positions, marker="o")
    plt.xlabel("X position (m, east)")
    plt.ylabel("Y position (m, north)")
    plt.title("Reconstructed Walking Trajectory (WALKING_AND_TURNING.csv)")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("part4_trajectory.png")
    plt.close()

    print("=== Part 4: WALKING_AND_TURNING.csv ===")
    print(f"Detected steps: {len(step_indices)}")
    print(f"Significant turns (|angle| > 30 deg): {len(significant_turns)}")
    for (s_idx, e_idx, angle_deg, direction) in significant_turns:
        mid_time = (t_ns[(s_idx + e_idx) // 2] - t_ns[0]) / 1e9
        print(f"  Turn at t={mid_time:.2f}s: {angle_deg:.2f} degrees, {direction}")
    print(f"Final heading: {heading_deg:.2f} degrees")
    print(f"Final position: x={x_positions[-1]:.2f} m, y={y_positions[-1]:.2f} m")
    print()


# -------------------------
# Main
# -------------------------

def main():
    print("Running Milestone 2 analysis...")
    part1_acceleration()
    part2_step_detection()
    part3_direction_detection()
    part4_trajectory()
    print("Done. Plots have been saved as PNG files in this directory.")


if __name__ == "__main__":
    main()
