#!/usr/bin/env python3
"""
Signal Generator for Reference Signal Capture

Generates test signals for acoustic measurement and analysis:
- Log sine sweep (20Hz-20kHz): Frequency response measurement
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
from scipy.signal import chirp


# Constants
FADE_DURATION_MS = 20  # Fade in/out duration in milliseconds
SWEEP_FREQ_START = 20  # Hz
SWEEP_FREQ_END = 20000  # Hz
IMPULSE_INTERVAL = 1.0  # Seconds between impulses


def generate_log_sweep(sample_rate: int, duration: float) -> np.ndarray:
    """
    Generate logarithmic sine sweep from 20Hz to 20kHz.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    
    Returns:
        Normalized signal array [-1, 1]
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = chirp(t, f0=SWEEP_FREQ_START, f1=SWEEP_FREQ_END, t1=duration, method='logarithmic')

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


def generate_filename(signal_type: str, sample_rate: int, duration: float, channels: str) -> str:
    """
    Generate descriptive filename encoding all parameters.
    
    Args:
        signal_type: Type of signal (log_sweep, white_noise, etc.)
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        channels: 'mono' or 'stereo'
    
    Returns:
        Filename string (without path)
    """

    if duration == int(duration):
        duration_str = f"{int(duration)}s"
    else:
        duration_str = f"{duration:.1f}s"
    
    return f"{signal_type}_{sample_rate}_{duration_str}_{channels}.wav"


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference signals for acoustic measurement and analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Signal Types:
  log_sweep    - Logarithmic sine sweep 20Hz-20kHz (frequency response)
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
    

    generators = {
        'log_sweep': generate_log_sweep,
        'white_noise': generate_white_noise,
        'pink_noise': generate_pink_noise,
        'impulse': generate_impulse,
    }
    
    generator = generators[args.signal_type]
    

    print(f"Generating {args.signal_type}...")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Duration: {args.duration} s")
    print(f"  Channels: {args.channels}")
    print(f"  Peak level: {args.peak_level} dBFS")
    
    signal = generator(args.sample_rate, args.duration)
    

    peak_amplitude = dbfs_to_linear(args.peak_level)
    signal = signal * peak_amplitude
    

    if args.signal_type != 'impulse':
        signal = apply_fade(signal, args.sample_rate)
    

    if args.channels == 'stereo':
        signal = to_stereo(signal)
    

    signal_out = to_float32(signal)
    

    filename = generate_filename(args.signal_type, args.sample_rate, args.duration, args.channels)
    filepath = output_path / filename
    

    wavfile.write(str(filepath), args.sample_rate, signal_out)
    
    print(f"  Output: {filepath}")
    print("Done.")


if __name__ == '__main__':
    main()
