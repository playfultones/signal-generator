# Signal Generator

Python CLI tool for generating reference audio signals for acoustic measurement and analysis.

## Signal Types

| Type | Purpose | Recommended Duration |
|------|---------|---------------------|
| `log_sweep` | Logarithmic sine sweep 20Hz–20kHz for frequency response measurement | 15–20s |
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

### Examples

```bash
# Default: 10s log sweep at 48kHz mono, -18 dBFS peak
python signal_generator.py

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

# Full measurement set
python signal_generator.py -t log_sweep -d 20
python signal_generator.py -t pink_noise -d 10
python signal_generator.py -t impulse -d 5
```

## Output

- **Format**: 32-bit float WAV
- **Peak level**: -18 dBFS by default (configurable via `-p`)
- **Fades**: 20ms cosine fade-in/fade-out (except impulse)
- **Filename**: `{type}_{sample_rate}_{duration}s_{channels}.wav`

Examples:
- `log_sweep_48000_20s_mono.wav`
- `pink_noise_96000_10s_stereo.wav`
- `impulse_48000_5s_mono.wav`

## Technical Details

### Log Sweep
Exponential chirp from 20Hz to 20kHz using scipy's `chirp()` function. Ideal for measuring frequency response via deconvolution.

### White Noise
Gaussian white noise with flat spectral density, normalized to [-1, 1] with headroom.

### Pink Noise
FFT-filtered noise with 1/f power spectral density (3dB/octave rolloff). More representative of real-world acoustic environments.

### Impulse
Single-sample impulses at 1-second intervals with 100ms initial offset. Clean transients without fades for accurate impulse response capture.

## License

MIT License - see [LICENSE](LICENSE) for details.
