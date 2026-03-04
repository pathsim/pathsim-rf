#########################################################################################
##
##                            Transmission Line Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks._block import Block
from pathsim.utils.adaptivebuffer import AdaptiveBuffer

# CONSTANTS =============================================================================

C0 = 299792458.0  # speed of light [m/s]

# BLOCKS ================================================================================

class TransmissionLine(Block):
    """Lossy transmission line modeled as a delayed scattering two-port.

    In the scattering (wave) domain, the transmission line crosses incident
    waves from one port to the other with a propagation delay and attenuation:

    .. math::

        b_1(t) = T \\cdot a_2(t - \\tau)

    .. math::

        b_2(t) = T \\cdot a_1(t - \\tau)

    where :math:`\\tau = L / v_p` is the one-way propagation delay,
    :math:`v_p = c_0 / \\sqrt{\\varepsilon_r}` is the phase velocity,
    and :math:`T = 10^{-\\alpha L / 20}` is the voltage transmission
    coefficient for attenuation :math:`\\alpha` in dB/m.

    The block uses a single vector-valued adaptive interpolating buffer
    to delay both wave directions simultaneously.

    Parameters
    ----------
    length : float
        Physical length of the line [m].
    er : float
        Effective relative permittivity [-]. Default 1.0 (free space).
    attenuation : float
        Attenuation constant [dB/m]. Default 0.0 (lossless).
    Z0 : float
        Characteristic impedance [Ohm]. Stored for reference, does not
        affect the scattering computation (matched-line assumption).
    """

    input_port_labels = {
        "a1": 0,
        "a2": 1,
    }

    output_port_labels = {
        "b1": 0,
        "b2": 1,
    }

    def __init__(self, length=1.0, er=1.0, attenuation=0.0, Z0=50.0):

        super().__init__()

        # input validation
        if length <= 0:
            raise ValueError(f"'length' must be positive but is {length}")
        if er <= 0:
            raise ValueError(f"'er' must be positive but is {er}")
        if attenuation < 0:
            raise ValueError(f"'attenuation' must be non-negative but is {attenuation}")

        # store parameters
        self.length = length
        self.er = er
        self.attenuation = attenuation
        self.Z0 = Z0

        # derived quantities
        self.vp = C0 / np.sqrt(er)
        self.tau = length / self.vp
        self.T = 10.0 ** (-attenuation * length / 20.0)

        # single vector-valued buffer for [a1, a2]
        self._buffer = AdaptiveBuffer(self.tau)

    def __len__(self):
        # no algebraic passthrough — output depends on past input only
        return 0

    def reset(self):
        super().reset()
        self._buffer.clear()

    def sample(self, t, dt):
        """Store current incident waves into the delay buffer."""
        self._buffer.add(t, np.array([self.inputs[0], self.inputs[1]]))

    def update(self, t):
        """Read delayed waves, cross and scale."""
        delayed = self._buffer.get(t)

        if np.isscalar(delayed):
            # buffer not yet filled (t < tau)
            self.outputs[0] = 0.0
            self.outputs[1] = 0.0
        else:
            # b1 = T * a2(t-tau), b2 = T * a1(t-tau)
            self.outputs[0] = self.T * delayed[1]
            self.outputs[1] = self.T * delayed[0]
