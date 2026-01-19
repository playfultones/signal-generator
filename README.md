# Signal Generator

Python CLI tool for generating reference audio signals for acoustic measurement and analysis.

## Signal Types

| Type | Purpose | Recommended Duration |
|------|---------|---------------------|
| `log_sweep` | Logarithmic sine sweep for frequency response measurement | 15–20s |
| `white_noise` | Broadband verification, FFT analysis | 10s |
| `pink_noise` | Broadband verification with 1/f spectrum | 10s |
| `impulse` | Impulse clicks (1/sec) for phase response / group delay check | 5s |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python signal_generator.py [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-t, --type` | `log_sweep` | Signal type: `log_sweep`, `white_noise`, `pink_noise`, `impulse` |
| `-r, --sample-rate` | `48000` | Sample rate in Hz |
| `-d, --duration` | `10` | Duration in seconds |
| `-c, --channels` | `mono` | Output channels: `mono` or `stereo` |
| `-p, --peak-level` | `-18` | Peak level in dBFS (range: -60 to 0) |
| `-o, --output` | `.` | Output folder |
| `--f-start` | `10` | Sweep start frequency in Hz |
| `--f-end` | `Nyquist-500` | Sweep end frequency in Hz |

### Examples

```bash
# Default: 10s log sweep at 48kHz mono, -18 dBFS peak, 10Hz to 23500Hz
python signal_generator.py

# Full bandwidth sweep for 48kHz (10Hz to 23500Hz)
python signal_generator.py -t log_sweep -d 20

# Classic 20Hz-20kHz sweep
python signal_generator.py -t log_sweep --f-start 20 --f-end 20000

# 96kHz sweep with extended high frequency range
python signal_generator.py -r 96000 -t log_sweep --f-end 47500

# 5-second impulse clicks
python signal_generator.py -t impulse -d 5

# 96kHz stereo pink noise
python signal_generator.py -t pink_noise -r 96000 -c stereo

# White noise to specific folder
python signal_generator.py -t white_noise -o ./test_signals/

# Log sweep at -12 dBFS peak level
python signal_generator.py -t log_sweep -p -12

# Full scale (0 dBFS) output
python signal_generator.py -p 0

# Full measurement set for 48kHz
python signal_generator.py -t log_sweep -d 20 -o ./signals/
python signal_generator.py -t pink_noise -d 10 -o ./signals/
python signal_generator.py -t white_noise -d 10 -o ./signals/
python signal_generator.py -t impulse -d 5 -o ./signals/
```

## Output

- **Format**: 32-bit float WAV
- **Peak level**: -18 dBFS by default (configurable via `-p`)
- **Fades**: 10ms cosine fade-in/fade-out (except impulse)
- **Filename format**:
  - Sweeps: `log_sweep_{f_start}hz-{f_end}hz_{sample_rate}_{duration}s_{channels}.wav`
  - Others: `{type}_{sample_rate}_{duration}s_{channels}.wav`

Examples:
- `log_sweep_10hz-23500hz_48000_20s_mono.wav`
- `log_sweep_20hz-20000hz_48000_20s_mono.wav`
- `pink_noise_96000_10s_stereo.wav`
- `impulse_48000_5s_mono.wav`

## Technical Details

### Log Sweep

Exponential/logarithmic chirp using direct phase accumulation for accurate frequency coverage from start to end frequency. Default range is 10 Hz to (Nyquist - 500 Hz), providing:

- **Sub-bass coverage**: Starts at 10 Hz to capture behavior below 20 Hz
- **Near-Nyquist coverage**: Extends to within 500 Hz of Nyquist for full bandwidth measurement
- **No aliasing artifacts**: 500 Hz headroom below Nyquist prevents aliasing issues

For a 48kHz sample rate, the default sweep covers **10 Hz to 23,500 Hz**.

### White Noise

Gaussian white noise with flat spectral density, normalized to [-1, 1] with headroom.

### Pink Noise

FFT-filtered noise with 1/f power spectral density (3dB/octave rolloff). More representative of real-world acoustic environments.

### Impulse

Single-sample impulses at 1-second intervals with 100ms initial offset. Clean transients without fades for accurate impulse response capture.

## License

MIT License - see [LICENSE](LICENSE) for details.
