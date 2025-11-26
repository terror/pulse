# Pulse

A terminal-based audio visualizer for macOS written in Rust.

## Features

- **Real-time FFT Spectrum Analysis**: Uses `realfft` to compute frequency bands.
- **Smooth Visualization**: Exponential smoothing for a fluid visual experience.
- **TUI Interface**: Built with `ratatui` and `crossterm`.

## Requirements

- **macOS**
- **Rust** (stable)
- **BlackHole** (Virtual Audio Driver) - [Download Here](https://github.com/ExistentialAudio/BlackHole)

## Setup

1.  **Install BlackHole** (2ch version is sufficient).
2.  **Configure Audio**:
    - Open **Audio MIDI Setup** (`Cmd+Space` -> Audio MIDI Setup).
    - Create a **Multi-Output Device** (click `+` in bottom left).
    - Check your main output (e.g., Headphones/Speakers) AND **BlackHole 2ch**.
    - Set this Multi-Output Device as your **system output** in System Settings -> Sound.

## Running

```bash
cargo run
```

The app looks for an input device with "BlackHole" in its name. If found, it will visualize the frequency spectrum.

## Controls

- `q` or `Esc`: Quit

## Troubleshooting

- **No audio/bars moving?**
    - Ensure your system output is set to the Multi-Output Device.
    - Ensure the app found the "BlackHole" device (check the terminal output).
    - Check volume levels in Audio MIDI Setup.
