########################################################################################
##
##                                  TESTS FOR
##                                'mixer.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_rf import RFMixer


# TESTS ================================================================================

class TestRFMixer(unittest.TestCase):
    """Test the RFMixer block."""

    # -- initialisation ----------------------------------------------------------------

    def test_init_default(self):
        """Test default initialization."""
        mx = RFMixer()
        self.assertEqual(mx.conversion_gain, 0.0)
        self.assertEqual(mx.Z0, 50.0)
        self.assertAlmostEqual(mx._gain_linear, 1.0)

    def test_init_custom(self):
        """Test custom initialization."""
        mx = RFMixer(conversion_gain=-6.0, Z0=75.0)
        self.assertEqual(mx.conversion_gain, -6.0)
        self.assertEqual(mx.Z0, 75.0)

    def test_init_negative_gain(self):
        """Negative conversion gain (loss) is valid."""
        mx = RFMixer(conversion_gain=-10.0)
        expected = 10.0 ** (-10.0 / 20.0)
        self.assertAlmostEqual(mx._gain_linear, expected)

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            RFMixer(Z0=0)
        with self.assertRaises(ValueError):
            RFMixer(Z0=-50)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(RFMixer.input_port_labels["rf"], 0)
        self.assertEqual(RFMixer.input_port_labels["lo"], 1)
        self.assertEqual(RFMixer.output_port_labels["if_out"], 0)

    # -- functionality -----------------------------------------------------------------

    def test_unity_gain_multiplication(self):
        """0 dB conversion gain: output is product of RF and LO."""
        mx = RFMixer(conversion_gain=0.0)
        mx.inputs[0] = 3.0
        mx.inputs[1] = 4.0
        mx.update(None)
        self.assertAlmostEqual(mx.outputs[0], 12.0)

    def test_conversion_gain_dB(self):
        """Conversion gain in dB scales the product."""
        mx = RFMixer(conversion_gain=20.0)  # 10x voltage
        mx.inputs[0] = 1.0
        mx.inputs[1] = 2.0
        mx.update(None)
        self.assertAlmostEqual(mx.outputs[0], 10.0 * 1.0 * 2.0)

    def test_conversion_loss(self):
        """Negative dB represents conversion loss."""
        mx = RFMixer(conversion_gain=-20.0)  # 0.1x voltage
        mx.inputs[0] = 5.0
        mx.inputs[1] = 2.0
        mx.update(None)
        self.assertAlmostEqual(mx.outputs[0], 0.1 * 5.0 * 2.0, places=4)

    def test_zero_rf(self):
        """Zero RF input gives zero output."""
        mx = RFMixer()
        mx.inputs[0] = 0.0
        mx.inputs[1] = 5.0
        mx.update(None)
        self.assertAlmostEqual(mx.outputs[0], 0.0)

    def test_zero_lo(self):
        """Zero LO input gives zero output."""
        mx = RFMixer()
        mx.inputs[0] = 3.0
        mx.inputs[1] = 0.0
        mx.update(None)
        self.assertAlmostEqual(mx.outputs[0], 0.0)

    def test_negative_signals(self):
        """Mixer handles negative signals correctly."""
        mx = RFMixer()
        mx.inputs[0] = -2.0
        mx.inputs[1] = 3.0
        mx.update(None)
        self.assertAlmostEqual(mx.outputs[0], -6.0)

    def test_sinusoidal_mixing(self):
        """Verify frequency mixing with sinusoids."""
        mx = RFMixer()
        f_rf = 1e9
        f_lo = 0.9e9
        t = 1e-10

        rf_val = np.cos(2 * np.pi * f_rf * t)
        lo_val = np.cos(2 * np.pi * f_lo * t)

        mx.inputs[0] = rf_val
        mx.inputs[1] = lo_val
        mx.update(None)

        expected = rf_val * lo_val
        self.assertAlmostEqual(mx.outputs[0], expected)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
