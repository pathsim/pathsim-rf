
<p align="center">
  <img src="https://raw.githubusercontent.com/pathsim/pathsim-rf/main/docs/source/logos/rf_logo.png" width="300" alt="PathSim-RF Logo" />
</p>

<p align="center">
  <strong>RF engineering toolbox for PathSim</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pathsim-rf"><img src="https://img.shields.io/pypi/v/pathsim-rf" alt="PyPI"></a>
  <img src="https://img.shields.io/github/license/pathsim/pathsim-rf" alt="License">
</p>

<p align="center">
  <a href="https://docs.pathsim.org/rf">Documentation</a> &bull;
  <a href="https://pathsim.org">PathSim Homepage</a> &bull;
  <a href="https://github.com/pathsim/pathsim-rf">GitHub</a>
</p>

---

PathSim-RF extends the [PathSim](https://github.com/pathsim/pathsim) simulation framework with blocks for RF and microwave engineering. All blocks follow the standard PathSim block interface and can be connected into simulation diagrams.

## Blocks

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| `RFNetwork` | N-port network from S-parameter data (Touchstone) via vector fitting | `ntwk`, `auto_fit` |
| `TransmissionLine` | Lossy delay-based transmission line (scattering domain) | `length`, `er`, `attenuation`, `Z0` |
| `RFAmplifier` | Amplifier with optional IP3 nonlinearity | `gain` [dB], `IIP3` [dBm], `P1dB` [dBm], `Z0` |
| `RFMixer` | Ideal frequency converter (time-domain multiplication) | `conversion_gain` [dB], `Z0` |

## Install

```bash
pip install pathsim-rf
```

## License

MIT
