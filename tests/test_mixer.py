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

    def test_init_default(self):
        """Test default initialization."""
        mx = RFMixer()
        self.assertEqual(mx.conversion_gain, 1.0)

    def test_init_custom(self):
        """Test custom initialization."""
        mx = RFMixer(conversion_gain=0.5)
        self.assertEqual(mx.conversion_gain, 0.5)

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            RFMixer(conversion_gain=0)
        with self.assertRaises(ValueError):
            RFMixer(conversion_gain=-1)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(RFMixer.input_port_labels["rf"], 0)
        self.assertEqual(RFMixer.input_port_labels["lo"], 1)
        self.assertEqual(RFMixer.output_port_labels["if_out"], 0)

    def test_multiplication(self):
        """Output is product of RF and LO signals."""
        mx = RFMixer()
        mx.inputs[0] = 3.0  # rf
        mx.inputs[1] = 4.0  # lo
        mx.update(None)
        self.assertAlmostEqual(mx.outputs[0], 12.0)

    def test_conversion_gain(self):
        """Conversion gain scales the output."""
        mx = RFMixer(conversion_gain=2.0)
        mx.inputs[0] = 3.0
        mx.inputs[1] = 5.0
        mx.update(None)
        self.assertAlmostEqual(mx.outputs[0], 30.0)

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
        """Verify frequency mixing with sinusoids (trig identity)."""
        mx = RFMixer()
        f_rf = 1e9   # 1 GHz
        f_lo = 0.9e9  # 900 MHz
        t = 1e-10  # sample time

        rf_val = np.cos(2 * np.pi * f_rf * t)
        lo_val = np.cos(2 * np.pi * f_lo * t)

        mx.inputs[0] = rf_val
        mx.inputs[1] = lo_val
        mx.update(None)

        # cos(a)*cos(b) = 0.5*[cos(a-b) + cos(a+b)]
        expected = rf_val * lo_val
        self.assertAlmostEqual(mx.outputs[0], expected)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
