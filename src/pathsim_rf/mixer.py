#########################################################################################
##
##                                RF Mixer Block
##
#########################################################################################

# IMPORTS ===============================================================================

from pathsim.blocks.function import Function


# BLOCKS ================================================================================

class RFMixer(Function):
    """Ideal RF mixer (frequency converter).

    Performs time-domain multiplication of the RF and local-oscillator
    (LO) signals, which corresponds to frequency translation:

    .. math::

        y(t) = G_{\\mathrm{conv}} \\cdot x_{\\mathrm{RF}}(t) \\cdot x_{\\mathrm{LO}}(t)

    Parameters
    ----------
    conversion_gain : float
        Conversion gain [dB]. Default 0.0.  Negative values represent
        conversion loss (typical for passive mixers).
    Z0 : float
        Reference impedance [Ohm]. Default 50.0.
    """

    input_port_labels = {
        "rf": 0,
        "lo": 1,
    }

    output_port_labels = {
        "if_out": 0,
    }

    def __init__(self, conversion_gain=0.0, Z0=50.0):

        if Z0 <= 0:
            raise ValueError(f"'Z0' must be positive but is {Z0}")

        self.conversion_gain = conversion_gain
        self.Z0 = Z0

        # linear voltage gain (can be < 1 for conversion loss)
        self._gain_linear = 10.0 ** (conversion_gain / 20.0)

        super().__init__(func=self._eval)

    def _eval(self, rf, lo):
        return self._gain_linear * rf * lo
