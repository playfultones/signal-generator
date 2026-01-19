#!/usr/bin/env python3
"""
Signal Generator for Reference Signal Capture

Generates test signals for acoustic measurement and analysis:
- Log sine sweep: Frequency response measurement (configurable range, default 10Hz to Nyquist-500Hz)
- White noise: Broadband verification, FFT analysis
- Pink noise: Broadband verification with 1/f spectrum
- Impulse clicks: Phase response / group delay check

Output: 32-bit float WAV files with descriptive filenames encoding parameters.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


# Constants
FADE_DURATION_MS = 10  # Fade in/out duration in milliseconds
SWEEP_FREQ_START = 10  # Hz - start low to capture sub-bass
SWEEP_FREQ_END_OFFSET = 500  # Hz below Nyquist - headroom for anti-aliasing
IMPULSE_INTERVAL = 1.0  # Seconds between impulses


def generate_log_sweep(sample_rate: int, duration: float, 
                       f_start: float = None, f_end: float = None) -> np.ndarray:
    """
    Generate logarithmic sine sweep with full bandwidth coverage.
    
    Uses direct phase accumulation method for accurate frequency sweep
    all the way to near-Nyquist frequencies.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        f_start: Start frequency in Hz (default: 10 Hz)
        f_end: End frequency in Hz (default: Nyquist - 500 Hz)
    
    Returns:
        Normalized signal array [-1, 1]
    """
    if f_start is None:
        f_start = SWEEP_FREQ_START
    if f_end is None:
        f_end = sample_rate / 2 - SWEEP_FREQ_END_OFFSET
    

    nyquist = sample_rate / 2
    if f_end > nyquist * 0.99:
        f_end = nyquist * 0.99
    
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    
    k = f_end / f_start
    L = duration * f_start / np.log(k)
    
    phase = 2 * np.pi * L * (np.power(k, t / duration) - 1)
    
    signal = np.sin(phase)
    
    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    
    return signal


def generate_white_noise(sample_rate: int, duration: float) -> np.ndarray:
    """
    Generate white noise with uniform spectral density.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    
    Returns:
        Normalized signal array [-1, 1]
    """
    num_samples = int(sample_rate * duration)
    signal = np.random.randn(num_samples)

    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    return signal


def generate_pink_noise(sample_rate: int, duration: float) -> np.ndarray:
    """
    Generate pink noise with 1/f spectral density using FFT filtering.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    
    Returns:
        Normalized signal array [-1, 1]
    """
    num_samples = int(sample_rate * duration)
    

    white = np.random.randn(num_samples)
    

    fft = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(num_samples, 1.0 / sample_rate)
    


    frequencies[0] = 1e-10
    pink_filter = 1.0 / np.sqrt(frequencies)
    

    fft_filtered = fft * pink_filter
    signal = np.fft.irfft(fft_filtered, n=num_samples)
    

    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    return signal


def generate_impulse(sample_rate: int, duration: float) -> np.ndarray:
    """
    Generate impulse clicks at 1-second intervals.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    
    Returns:
        Signal array with impulses at [-1, 1]
    """
    num_samples = int(sample_rate * duration)
    signal = np.zeros(num_samples)
    

    impulse_interval_samples = int(sample_rate * IMPULSE_INTERVAL)
    

    offset_samples = int(sample_rate * 0.1)  # 100ms offset
    
    for i in range(int(duration / IMPULSE_INTERVAL)):
        idx = offset_samples + i * impulse_interval_samples
        if idx < num_samples:
            signal[idx] = 1.0  # Positive impulse at unity (scaled by peak level later)
    
    return signal


def apply_fade(signal: np.ndarray, sample_rate: int, fade_ms: float = FADE_DURATION_MS) -> np.ndarray:
    """
    Apply cosine fade-in and fade-out to signal.
    
    Args:
        signal: Input signal array
        sample_rate: Sample rate in Hz
        fade_ms: Fade duration in milliseconds
    
    Returns:
        Signal with fades applied
    """
    fade_samples = int(sample_rate * fade_ms / 1000)
    
    if fade_samples * 2 >= len(signal):
        # Signal too short for fades
        return signal
    

    fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
    fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))
    

    signal = signal.copy()
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out
    
    return signal


def to_stereo(signal: np.ndarray) -> np.ndarray:
    """
    Convert mono signal to stereo by duplicating to both channels.
    
    Args:
        signal: Mono signal array (1D)
    
    Returns:
        Stereo signal array (2D, shape: samples x 2)
    """
    return np.column_stack([signal, signal])


def to_float32(signal: np.ndarray) -> np.ndarray:
    """
    Convert signal to 32-bit float format for WAV output.
    
    Args:
        signal: Normalized signal array [-1, 1]
    
    Returns:
        Signal as float32 array
    """
    return signal.astype(np.float32)


def dbfs_to_linear(dbfs: float) -> float:
    """
    Convert dBFS to linear amplitude.
    
    Args:
        dbfs: Level in dBFS (0 = full scale, negative values = quieter)
    
    Returns:
        Linear amplitude multiplier (0 to 1)
    """
    return 10 ** (dbfs / 20)


def generate_filename(signal_type: str, sample_rate: int, duration: float, channels: str,
                      f_start: float = None, f_end: float = None) -> str:
    """
    Generate descriptive filename encoding all parameters.
    
    Args:
        signal_type: Type of signal (log_sweep, white_noise, etc.)
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        channels: 'mono' or 'stereo'
        f_start: Start frequency for sweeps (optional)
        f_end: End frequency for sweeps (optional)
    
    Returns:
        Filename string (without path)
    """

    if duration == int(duration):
        duration_str = f"{int(duration)}s"
    else:
        duration_str = f"{duration:.1f}s"
    

    if signal_type == 'log_sweep' and f_start is not None and f_end is not None:
        f_start_str = f"{int(f_start)}" if f_start >= 10 else f"{f_start:.1f}"
        f_end_str = f"{int(f_end)}" if f_end >= 1000 else f"{f_end:.0f}"
        return f"{signal_type}_{f_start_str}hz-{f_end_str}hz_{sample_rate}_{duration_str}_{channels}.wav"
    
    return f"{signal_type}_{sample_rate}_{duration_str}_{channels}.wav"


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference signals for acoustic measurement and analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Signal Types:
  log_sweep    - Logarithmic sine sweep (default: 10Hz to Nyquist-500Hz)
  white_noise  - White noise burst (broadband verification)
  pink_noise   - Pink noise burst with 1/f spectrum
  impulse      - Impulse clicks at 1-second intervals (phase/group delay)

Examples:
  %(prog)s                                    # Default: 10s log sweep, 48kHz, mono, -18dBFS
  %(prog)s -t impulse -d 5                    # 5s impulse clicks
  %(prog)s -t pink_noise -r 96000 -c stereo   # 96kHz stereo pink noise
  %(prog)s -t white_noise -o ./test_signals/  # White noise to specific folder
  %(prog)s -t log_sweep -p -12                # Log sweep at -12 dBFS peak
  %(prog)s -p 0                               # Full scale (0 dBFS) output
  %(prog)s --f-start 20 --f-end 20000         # Classic 20Hz-20kHz sweep
  %(prog)s -r 96000 --f-end 47500             # 96kHz sweep to near-Nyquist
        """
    )
    
    parser.add_argument(
        '-r', '--sample-rate',
        type=int,
        default=48000,
        metavar='RATE',
        help='Sample rate in Hz (default: 48000)'
    )
    
    parser.add_argument(
        '-c', '--channels',
        type=str,
        choices=['mono', 'stereo'],
        default='mono',
        help='Output channels (default: mono)'
    )
    
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=10.0,
        metavar='SECONDS',
        help='Signal duration in seconds (default: 10)'
    )
    
    parser.add_argument(
        '-t', '--type',
        type=str,
        choices=['log_sweep', 'white_noise', 'pink_noise', 'impulse'],
        default='log_sweep',
        dest='signal_type',
        help='Signal type (default: log_sweep)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='.',
        metavar='FOLDER',
        help='Output folder (default: current directory)'
    )
    
    parser.add_argument(
        '-p', '--peak-level',
        type=float,
        default=-18.0,
        metavar='DBFS',
        help='Peak level in dBFS (default: -18, range: -60 to 0)'
    )
    
    parser.add_argument(
        '--f-start',
        type=float,
        default=None,
        metavar='HZ',
        help='Sweep start frequency in Hz (default: 10 Hz)'
    )
    
    parser.add_argument(
        '--f-end',
        type=float,
        default=None,
        metavar='HZ',
        help='Sweep end frequency in Hz (default: Nyquist - 500 Hz)'
    )
    
    args = parser.parse_args()
    

    if args.sample_rate < 8000 or args.sample_rate > 384000:
        print(f"Error: Sample rate {args.sample_rate} out of reasonable range (8000-384000 Hz)", 
              file=sys.stderr)
        sys.exit(1)
    

    if args.duration <= 0 or args.duration > 3600:
        print(f"Error: Duration {args.duration} out of range (0-3600 seconds)", 
              file=sys.stderr)
        sys.exit(1)
    
    if args.peak_level < -60 or args.peak_level > 0:
        print(f"Error: Peak level {args.peak_level} out of range (-60 to 0 dBFS)", 
              file=sys.stderr)
        sys.exit(1)
    

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    


    f_start = args.f_start if args.f_start is not None else SWEEP_FREQ_START
    f_end = args.f_end if args.f_end is not None else (args.sample_rate / 2 - SWEEP_FREQ_END_OFFSET)
    

    nyquist = args.sample_rate / 2
    if f_end > nyquist:
        print(f"Warning: f_end ({f_end} Hz) exceeds Nyquist ({nyquist} Hz), clamping to {nyquist * 0.99:.0f} Hz",
              file=sys.stderr)
        f_end = nyquist * 0.99
    
    if f_start >= f_end:
        print(f"Error: f_start ({f_start} Hz) must be less than f_end ({f_end} Hz)",
              file=sys.stderr)
        sys.exit(1)

    print(f"Generating {args.signal_type}...")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Duration: {args.duration} s")
    print(f"  Channels: {args.channels}")
    print(f"  Peak level: {args.peak_level} dBFS")
    
    if args.signal_type == 'log_sweep':
        print(f"  Frequency range: {f_start:.1f} Hz - {f_end:.1f} Hz")
        signal = generate_log_sweep(args.sample_rate, args.duration, f_start, f_end)
    elif args.signal_type == 'white_noise':
        signal = generate_white_noise(args.sample_rate, args.duration)
    elif args.signal_type == 'pink_noise':
        signal = generate_pink_noise(args.sample_rate, args.duration)
    elif args.signal_type == 'impulse':
        signal = generate_impulse(args.sample_rate, args.duration)
    else:
        print(f"Error: Unknown signal type {args.signal_type}", file=sys.stderr)
        sys.exit(1)
    

    peak_amplitude = dbfs_to_linear(args.peak_level)
    signal = signal * peak_amplitude
    

    if args.signal_type != 'impulse':
        signal = apply_fade(signal, args.sample_rate)
    

    if args.channels == 'stereo':
        signal = to_stereo(signal)
    

    signal_out = to_float32(signal)
    

    if args.signal_type == 'log_sweep':
        filename = generate_filename(args.signal_type, args.sample_rate, args.duration, 
                                     args.channels, f_start, f_end)
    else:
        filename = generate_filename(args.signal_type, args.sample_rate, args.duration, args.channels)
    filepath = output_path / filename
    

    wavfile.write(str(filepath), args.sample_rate, signal_out)
    
    print(f"  Output: {filepath}")
    print("Done.")


if __name__ == '__main__':
    main()
