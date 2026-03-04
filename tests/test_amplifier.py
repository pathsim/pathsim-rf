########################################################################################
##
##                                  TESTS FOR
##                              'amplifier.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_rf import RFAmplifier
from pathsim_rf.amplifier import _dbm_to_vpeak


# TESTS ================================================================================

class TestRFAmplifier(unittest.TestCase):
    """Test the RFAmplifier block."""

    # -- initialisation ----------------------------------------------------------------

    def test_init_default(self):
        """Test default initialization."""
        amp = RFAmplifier()
        self.assertEqual(amp.gain, 20.0)
        self.assertIsNone(amp.IIP3)
        self.assertIsNone(amp.P1dB)
        self.assertEqual(amp.Z0, 50.0)

    def test_init_custom(self):
        """Test custom initialization with IIP3."""
        amp = RFAmplifier(gain=15.0, IIP3=10.0, Z0=75.0)
        self.assertEqual(amp.gain, 15.0)
        self.assertEqual(amp.IIP3, 10.0)
        self.assertAlmostEqual(amp.P1dB, 10.0 - 9.6)
        self.assertEqual(amp.Z0, 75.0)

    def test_init_P1dB_derives_IIP3(self):
        """P1dB without IIP3 derives IIP3 = P1dB + 9.6."""
        amp = RFAmplifier(P1dB=0.0)
        self.assertAlmostEqual(amp.IIP3, 9.6)
        self.assertAlmostEqual(amp.P1dB, 0.0)

    def test_IIP3_takes_precedence(self):
        """IIP3 takes precedence over P1dB when both given."""
        amp = RFAmplifier(P1dB=0.0, IIP3=15.0)
        self.assertEqual(amp.IIP3, 15.0)
        self.assertAlmostEqual(amp.P1dB, 15.0 - 9.6)

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            RFAmplifier(Z0=0)
        with self.assertRaises(ValueError):
            RFAmplifier(Z0=-50)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(RFAmplifier.input_port_labels["rf_in"], 0)
        self.assertEqual(RFAmplifier.output_port_labels["rf_out"], 0)

    # -- linear mode -------------------------------------------------------------------

    def test_linear_gain_dB(self):
        """20 dB gain = voltage factor of 10."""
        amp = RFAmplifier(gain=20.0)
        amp.inputs[0] = 0.1
        amp.update(None)
        self.assertAlmostEqual(amp.outputs[0], 1.0)

    def test_linear_6dB(self):
        """6 dB gain ≈ voltage factor of ~2."""
        amp = RFAmplifier(gain=6.0)
        amp.inputs[0] = 1.0
        amp.update(None)
        expected = 10.0 ** (6.0 / 20.0)  # 1.9953
        self.assertAlmostEqual(amp.outputs[0], expected, places=4)

    def test_linear_negative_input(self):
        """Linear mode works with negative inputs."""
        amp = RFAmplifier(gain=20.0)
        amp.inputs[0] = -0.05
        amp.update(None)
        self.assertAlmostEqual(amp.outputs[0], -0.5)

    def test_linear_zero_input(self):
        """Zero input produces zero output."""
        amp = RFAmplifier(gain=20.0)
        amp.inputs[0] = 0.0
        amp.update(None)
        self.assertAlmostEqual(amp.outputs[0], 0.0)

    # -- nonlinear (IP3) mode ----------------------------------------------------------

    def test_ip3_small_signal_linear(self):
        """Small signals are approximately linear even with IP3."""
        amp = RFAmplifier(gain=20.0, IIP3=10.0)
        # tiny input well below compression
        amp.inputs[0] = 1e-6
        amp.update(None)
        expected_linear = amp._a1 * 1e-6
        self.assertAlmostEqual(amp.outputs[0], expected_linear, places=10)

    def test_ip3_compression(self):
        """Near IP3 the output compresses below linear gain."""
        amp = RFAmplifier(gain=20.0, IIP3=10.0)
        A_iip3 = _dbm_to_vpeak(10.0, 50.0)
        # drive at half the IIP3 voltage — should see compression
        x_in = A_iip3 * 0.5
        amp.inputs[0] = x_in
        amp.update(None)
        linear_out = amp._a1 * x_in
        self.assertLess(amp.outputs[0], linear_out)

    def test_ip3_saturation_clip(self):
        """Output is clipped at the gain compression peak for large signals."""
        amp = RFAmplifier(gain=20.0, IIP3=10.0)
        amp.inputs[0] = 1e3  # way beyond saturation
        amp.update(None)
        self.assertAlmostEqual(amp.outputs[0], amp._y_sat)

    def test_ip3_symmetry(self):
        """Nonlinear response is odd-symmetric."""
        amp = RFAmplifier(gain=20.0, IIP3=10.0)

        amp.inputs[0] = 1e3
        amp.update(None)
        pos = amp.outputs[0]

        amp.inputs[0] = -1e3
        amp.update(None)
        neg = amp.outputs[0]

        self.assertAlmostEqual(pos, -neg)

    def test_ip3_zero(self):
        """Zero input gives zero output with IP3."""
        amp = RFAmplifier(gain=20.0, IIP3=10.0)
        amp.inputs[0] = 0.0
        amp.update(None)
        self.assertAlmostEqual(amp.outputs[0], 0.0)

    # -- helper ------------------------------------------------------------------------

    def test_dbm_to_vpeak(self):
        """Verify dBm to peak voltage conversion."""
        # 0 dBm = 1 mW into 50 Ohm -> V_rms = sqrt(0.001*50) = 0.2236
        # V_peak = V_rms * sqrt(2) = 0.3162
        v = _dbm_to_vpeak(0.0, 50.0)
        self.assertAlmostEqual(v, np.sqrt(2.0 * 50.0 * 1e-3), places=6)

    def test_dbm_to_vpeak_30dBm(self):
        """30 dBm = 1 W -> V_peak = sqrt(2*50*1) = 10.0 V."""
        v = _dbm_to_vpeak(30.0, 50.0)
        self.assertAlmostEqual(v, 10.0, places=4)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
