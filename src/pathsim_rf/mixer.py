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
        Linear conversion gain (dimensionless). Default 1.0.
    """

    input_port_labels = {
        "rf": 0,
        "lo": 1,
    }

    output_port_labels = {
        "if_out": 0,
    }

    def __init__(self, conversion_gain=1.0):

        if conversion_gain <= 0:
            raise ValueError(
                f"'conversion_gain' must be positive but is {conversion_gain}"
            )

        self.conversion_gain = conversion_gain

        super().__init__(func=self._eval)

    def _eval(self, rf, lo):
        return self.conversion_gain * rf * lo
