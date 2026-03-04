#########################################################################################
##
##                              RF Amplifier Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function


# BLOCKS ================================================================================

class RFAmplifier(Function):
    """Ideal RF amplifier with optional output saturation.

    In the linear regime the amplifier simply scales the input signal:

    .. math::

        y(t) = G \\cdot x(t)

    When a saturation level is specified, soft compression is modelled
    with a hyperbolic tangent:

    .. math::

        y(t) = V_{\\mathrm{sat}} \\tanh\\!\\left(\\frac{G \\cdot x(t)}{V_{\\mathrm{sat}}}\\right)

    Parameters
    ----------
    gain : float
        Linear voltage gain (dimensionless). Default 10.0.
    saturation : float or None
        Output saturation amplitude. If *None* (default) the amplifier
        operates in purely linear mode.
    """

    input_port_labels = {
        "rf_in": 0,
    }

    output_port_labels = {
        "rf_out": 0,
    }

    def __init__(self, gain=10.0, saturation=None):

        # input validation
        if gain <= 0:
            raise ValueError(f"'gain' must be positive but is {gain}")
        if saturation is not None and saturation <= 0:
            raise ValueError(f"'saturation' must be positive but is {saturation}")

        self.gain = gain
        self.saturation = saturation

        super().__init__(func=self._eval)

    def _eval(self, rf_in):
        if self.saturation is None:
            return self.gain * rf_in
        return self.saturation * np.tanh(self.gain * rf_in / self.saturation)
