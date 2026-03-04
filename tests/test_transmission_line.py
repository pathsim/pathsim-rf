########################################################################################
##
##                                  TESTS FOR
##                          'transmission_line.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_rf import TransmissionLine
from pathsim_rf.transmission_line import C0


# TESTS ================================================================================

class TestTransmissionLine(unittest.TestCase):
    """Test the TransmissionLine block."""

    def test_init_default(self):
        """Test default initialization."""
        tl = TransmissionLine()
        self.assertEqual(tl.length, 1.0)
        self.assertEqual(tl.er, 1.0)
        self.assertEqual(tl.attenuation, 0.0)
        self.assertEqual(tl.Z0, 50.0)

    def test_init_custom(self):
        """Test custom initialization."""
        tl = TransmissionLine(length=0.5, er=4.0, attenuation=0.1, Z0=75.0)
        self.assertEqual(tl.length, 0.5)
        self.assertEqual(tl.er, 4.0)
        self.assertEqual(tl.attenuation, 0.1)
        self.assertEqual(tl.Z0, 75.0)

    def test_derived_quantities(self):
        """Verify propagation velocity, delay, and transmission coefficient."""
        tl = TransmissionLine(length=2.0, er=4.0, attenuation=0.5)

        expected_vp = C0 / np.sqrt(4.0)
        expected_tau = 2.0 / expected_vp
        expected_T = 10.0 ** (-0.5 * 2.0 / 20.0)

        self.assertAlmostEqual(tl.vp, expected_vp)
        self.assertAlmostEqual(tl.tau, expected_tau)
        self.assertAlmostEqual(tl.T, expected_T)

    def test_lossless_transmission(self):
        """Lossless line has T = 1."""
        tl = TransmissionLine(attenuation=0.0)
        self.assertAlmostEqual(tl.T, 1.0)

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            TransmissionLine(length=0)
        with self.assertRaises(ValueError):
            TransmissionLine(length=-1)
        with self.assertRaises(ValueError):
            TransmissionLine(er=0)
        with self.assertRaises(ValueError):
            TransmissionLine(attenuation=-0.1)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(TransmissionLine.input_port_labels["a1"], 0)
        self.assertEqual(TransmissionLine.input_port_labels["a2"], 1)
        self.assertEqual(TransmissionLine.output_port_labels["b1"], 0)
        self.assertEqual(TransmissionLine.output_port_labels["b2"], 1)

    def test_no_passthrough(self):
        """Delay block has no algebraic passthrough."""
        tl = TransmissionLine()
        self.assertEqual(len(tl), 0)

    def test_output_zero_before_delay(self):
        """Before the buffer fills, output should be zero."""
        tl = TransmissionLine(length=1.0, er=1.0)

        tl.inputs[0] = 1.0
        tl.inputs[1] = 2.0
        tl.sample(0.0, 0.01)
        tl.update(0.0)

        self.assertAlmostEqual(tl.outputs[0], 0.0)
        self.assertAlmostEqual(tl.outputs[1], 0.0)

    def test_crossing(self):
        """Verify that a1 appears at b2 and a2 appears at b1 after delay."""
        tau = 1e-9  # short line for easy testing
        length = tau * C0  # er=1

        tl = TransmissionLine(length=length, er=1.0, attenuation=0.0)
        self.assertAlmostEqual(tl.tau, tau)

        # Fill buffer with constant input over several samples
        dt = tau / 10
        for i in range(20):
            t = i * dt
            tl.inputs[0] = 3.0  # a1
            tl.inputs[1] = 7.0  # a2
            tl.sample(t, dt)

        # Query at t > tau — should see crossed, unattenuated output
        t_query = 15 * dt
        tl.update(t_query)

        self.assertAlmostEqual(tl.outputs[0], 7.0, places=1)  # b1 = T * a2
        self.assertAlmostEqual(tl.outputs[1], 3.0, places=1)  # b2 = T * a1

    def test_attenuation(self):
        """Verify that attenuation scales the output correctly."""
        tau = 1e-9
        length = tau * C0
        atten_dB_per_m = 3.0  # 3 dB/m

        tl = TransmissionLine(length=length, er=1.0, attenuation=atten_dB_per_m)
        expected_T = 10.0 ** (-atten_dB_per_m * length / 20.0)

        # Fill buffer
        dt = tau / 10
        for i in range(20):
            t = i * dt
            tl.inputs[0] = 1.0
            tl.inputs[1] = 1.0
            tl.sample(t, dt)

        tl.update(15 * dt)

        self.assertAlmostEqual(tl.outputs[0], expected_T, places=1)
        self.assertAlmostEqual(tl.outputs[1], expected_T, places=1)

    def test_reset(self):
        """After reset, buffer should be empty and outputs zero."""
        tl = TransmissionLine()
        tl.inputs[0] = 5.0
        tl.inputs[1] = 5.0
        tl.sample(0.0, 0.01)

        tl.reset()
        tl.update(1.0)

        self.assertAlmostEqual(tl.outputs[0], 0.0)
        self.assertAlmostEqual(tl.outputs[1], 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
