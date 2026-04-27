# hantek1008c-linux-gui

An open-source oscilloscope GUI for the **Hantek 1008C** USB scope on Linux.

The Hantek 1008C is an 8-channel, 12-bit USB oscilloscope. The vendor software is
Windows-only. This project gives you a scope UI on Linux with no proprietary
software required.

## Demo

https://github.com/user-attachments/assets/15a113a6-694e-49ce-842e-19c3c2059689

## Features

- Up to 8 simultaneous channels, toggled on/off individually
- Adjustable V/div per channel (10 mV – 5 V)
- Adjustable time/div
- Hardware trigger with selectable channel, level, and horizontal position
- Draggable channel offset handles in a left margin strip
- Channels stagger vertically on first enable so they don't stack on zero

## Requirements

- Linux (udev USB access — see setup below)
- Python ≥ 3.10
- `pip install -r requirements.txt`

Dependencies: `PyQt6`, `pyqtgraph`, `pyusb`, `overrides`

## USB device access

The device needs a udev rule so it's accessible without root:

1. Create `/etc/udev/rules.d/99-hantek1008.rules` with:
   ```
   ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="0783", ATTR{idProduct}=="5725", MODE="0666"
   ```
2. `sudo udevadm control -R`
3. Replug the device

## Setup & running

```bash
git clone https://github.com/LyndonTate/hantek1008c-linux-gui.git
cd hantek1008c-linux-gui
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

## Known limitations

- Rising edge trigger only (falling edge not yet implemented)
- No roll mode (continuous scrolling for slow signals)
- No automatic measurements (Vpp, frequency, RMS, etc.)
- No measurement cursors
- No waveform export (CSV, image, etc.)
- Trigger mode is normal only (no auto-trigger or single-shot)

## Project structure

```
gui/            PyQt6/pyqtgraph oscilloscope application
vendor/         Modified copy of the upstream hantek1008py driver (see NOTICE)
requirements.txt
```

## Credits & license

The USB driver in `vendor/hantek1008.py` is derived from
[hantek1008py](https://github.com/mfg92/hantek1008py) by Mathias Graßmann (mfg92),
licensed under the Apache License 2.0.

Modifications to the upstream driver:
- Replaced assert checks on init responses with `log.debug()` for compatibility with devices that have slightly different firmware responses
- Removed `0xc2` command from the burst loop
- Replaced hardcoded `0xac` init payload with a dynamically computed value
- Added thread-safe queuing of trigger level and pre-trigger depth changes between burst cycles
- Added `reconfigure()` to switch channels and trigger settings without a full USB reinit
- Added `_hw_trigger_ac_payload()` computing the correct pre-trigger buffer depth per channel count
- Extended `Hantek1008.__init__` with `trigger_channel`, `trigger_slope`, `trigger_level` parameters

This project is also licensed under the **Apache License 2.0** — see `LICENSE`.
See `NOTICE` for the full upstream attribution.
