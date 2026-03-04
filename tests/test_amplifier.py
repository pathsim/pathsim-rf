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


# TESTS ================================================================================

class TestRFAmplifier(unittest.TestCase):
    """Test the RFAmplifier block."""

    def test_init_default(self):
        """Test default initialization."""
        amp = RFAmplifier()
        self.assertEqual(amp.gain, 10.0)
        self.assertIsNone(amp.saturation)

    def test_init_custom(self):
        """Test custom initialization."""
        amp = RFAmplifier(gain=20.0, saturation=5.0)
        self.assertEqual(amp.gain, 20.0)
        self.assertEqual(amp.saturation, 5.0)

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            RFAmplifier(gain=0)
        with self.assertRaises(ValueError):
            RFAmplifier(gain=-1)
        with self.assertRaises(ValueError):
            RFAmplifier(saturation=0)
        with self.assertRaises(ValueError):
            RFAmplifier(saturation=-1)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(RFAmplifier.input_port_labels["rf_in"], 0)
        self.assertEqual(RFAmplifier.output_port_labels["rf_out"], 0)

    def test_linear_gain(self):
        """Linear mode: output = gain * input."""
        amp = RFAmplifier(gain=5.0)
        amp.inputs[0] = 2.0
        amp.update(None)
        self.assertAlmostEqual(amp.outputs[0], 10.0)

    def test_linear_negative_input(self):
        """Linear mode works with negative inputs."""
        amp = RFAmplifier(gain=3.0)
        amp.inputs[0] = -4.0
        amp.update(None)
        self.assertAlmostEqual(amp.outputs[0], -12.0)

    def test_linear_zero_input(self):
        """Zero input produces zero output."""
        amp = RFAmplifier(gain=10.0)
        amp.inputs[0] = 0.0
        amp.update(None)
        self.assertAlmostEqual(amp.outputs[0], 0.0)

    def test_saturation_small_signal(self):
        """With saturation, small signals are approximately linear."""
        amp = RFAmplifier(gain=10.0, saturation=100.0)
        amp.inputs[0] = 0.01  # small signal: gain*input/sat = 0.001
        amp.update(None)
        # tanh(x) ≈ x for small x, so output ≈ gain * input
        self.assertAlmostEqual(amp.outputs[0], 10.0 * 0.01, places=3)

    def test_saturation_large_signal(self):
        """With saturation, large signals are clipped to saturation level."""
        amp = RFAmplifier(gain=100.0, saturation=5.0)
        amp.inputs[0] = 1000.0  # heavily driven
        amp.update(None)
        # tanh(large) ≈ 1, so output ≈ saturation
        self.assertAlmostEqual(amp.outputs[0], 5.0, places=3)

    def test_saturation_symmetry(self):
        """Saturation is symmetric for positive and negative inputs."""
        amp = RFAmplifier(gain=100.0, saturation=5.0)

        amp.inputs[0] = 1000.0
        amp.update(None)
        pos = amp.outputs[0]

        amp.inputs[0] = -1000.0
        amp.update(None)
        neg = amp.outputs[0]

        self.assertAlmostEqual(pos, -neg)

    def test_saturation_zero(self):
        """Zero input gives zero output even with saturation."""
        amp = RFAmplifier(gain=10.0, saturation=5.0)
        amp.inputs[0] = 0.0
        amp.update(None)
        self.assertAlmostEqual(amp.outputs[0], 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
