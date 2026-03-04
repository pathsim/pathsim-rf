#########################################################################################
##
##                              RF Amplifier Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function


# HELPERS ===============================================================================

def _dbm_to_vpeak(p_dbm, z0):
    """Convert power in dBm to peak voltage amplitude."""
    p_watts = 10.0 ** (p_dbm / 10.0) * 1e-3
    return np.sqrt(2.0 * z0 * p_watts)


# BLOCKS ================================================================================

class RFAmplifier(Function):
    """RF amplifier with optional nonlinearity (IP3 / P1dB compression).

    In the linear regime the amplifier scales the input signal by the
    voltage gain derived from the specified gain in dB:

    .. math::

        y(t) = a_1 \\cdot x(t)

    When nonlinearity is specified via IIP3 or P1dB, a third-order
    polynomial model is used:

    .. math::

        y(t) = a_1 x(t) + a_3 x^3(t)

    where :math:`a_3 = -a_1 / A_{\\mathrm{IIP3}}^2` and
    :math:`A_{\\mathrm{IIP3}}` is the input-referred IP3 voltage
    amplitude.  The output is hard-clipped at the gain compression
    peak to prevent unphysical sign reversal.

    Parameters
    ----------
    gain : float
        Small-signal voltage gain [dB]. Default 20.0.
    P1dB : float or None
        Input-referred 1 dB compression point [dBm]. If given without
        *IIP3*, the intercept is estimated as IIP3 = P1dB + 9.6 dB.
    IIP3 : float or None
        Input-referred third-order intercept point [dBm].  Takes
        precedence over *P1dB* if both are given.
    Z0 : float
        Reference impedance [Ohm]. Default 50.0.
    """

    input_port_labels = {
        "rf_in": 0,
    }

    output_port_labels = {
        "rf_out": 0,
    }

    def __init__(self, gain=20.0, P1dB=None, IIP3=None, Z0=50.0):

        # input validation
        if Z0 <= 0:
            raise ValueError(f"'Z0' must be positive but is {Z0}")

        # store user-facing parameters
        self.gain = gain
        self.Z0 = Z0

        # linear voltage gain
        self._a1 = 10.0 ** (gain / 20.0)

        # resolve nonlinearity specification
        if IIP3 is not None:
            self.IIP3 = float(IIP3)
            self.P1dB = self.IIP3 - 9.6
        elif P1dB is not None:
            self.P1dB = float(P1dB)
            self.IIP3 = self.P1dB + 9.6
        else:
            self.IIP3 = None
            self.P1dB = None

        # derive polynomial coefficients
        if self.IIP3 is not None:
            A_iip3 = _dbm_to_vpeak(self.IIP3, Z0)
            self._a3 = -self._a1 / A_iip3 ** 2
            # clip at gain compression peak (dy/dx = 0)
            self._x_sat = A_iip3 / np.sqrt(3.0)
            self._y_sat = 2.0 * self._a1 * A_iip3 / (3.0 * np.sqrt(3.0))
        else:
            self._a3 = 0.0
            self._x_sat = None
            self._y_sat = None

        super().__init__(func=self._eval)

    def _eval(self, rf_in):
        x = rf_in
        if self._x_sat is not None and abs(x) > self._x_sat:
            return np.copysign(self._y_sat, x)
        return self._a1 * x + self._a3 * x ** 3
