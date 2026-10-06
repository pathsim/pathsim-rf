########################################################################################
##
##                                  TESTS FOR
##                              'netlist_to_statespace.py'
##
########################################################################################

# IMPORTS ==============================================================================
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

import pathsim_rf.netlist_to_statespace
from pathsim_rf.netlist_to_statespace import CircuitModel, parse_netlist, parse_value_with_units, parse_netlist_file

# TESTS ================================================================================

class TestNetlistParser(unittest.TestCase):
    """Test the netlist parser functions and CircuitModel used in NetlistStateSpace block."""

    def test_parse_value_with_si_suffixes(self):
        """Test conversion between spice si suffixes and power of ten."""
        cases = {
            "10f": 10e-15,
            "10p": 10e-12,
            "10n": 10e-9,
            "10u": 10e-6,
            "10µ": 10e-6,
            "10m": 10e-3,
            "10k": 10e3,
            "10K": 10e3,
            "10meg": 10e6,
            "10Meg": 10e6,
            "10M": 10e6,
            "10g": 10e9,
            "10G": 10e9,
            "10t": 10e12,
            "10T": 10e12,
            "100.e-3": 100e-3,
            "2.2kOhm": 2200.0,
            "3mH": 3e-3,
            "4uF": 4e-6,
        }
        for raw, expected in cases.items():
            self.assertAlmostEqual(parse_value_with_units(raw), expected)

    def test_parse_param_reference(self):
        """Test .param in spice list are taken into account"""
        elements = parse_netlist(
            """
            .param RMAIN = 12k
            R1 n1 0 {RMAIN}
            L1 n1 0 1mH
            """
        )
        self.assertEqual(len(elements), 2)
        self.assertAlmostEqual(elements[0].value, 12000.0)
        self.assertAlmostEqual(elements[1].value, 1e-3)

    def test_parse_current_source_waveform_keeps_placeholder(self):
        """Test if current sources with args are correctly parsed"""
        elements = parse_netlist('I1 0 N3 PWL file="SIGNAL.TXT"')
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].kind, "I")
        self.assertIsNone(elements[0].value)

    def test_parse_behavioral_source_placeholder(self):
        """Test behavioral sources with are correctly parsed 
        and segragated between current and voltage sources"""
        elements = parse_netlist("B1 0 N3 I=10*(exp(-t)-exp(-2*t)) \n B2 0 N4 V=10*(exp(-t)-exp(-2*t))")
        self.assertEqual(len(elements), 2)
        self.assertEqual(elements[0].kind, "I")
        self.assertIsNone(elements[0].value)
        self.assertEqual(elements[1].kind, "V")
        self.assertIsNone(elements[0].value)

    def test_unsupported_element_raises(self):
        """Test unsupported netlist elements raising an error"""
        with self.assertRaises(ValueError):
            parse_netlist("X1 n1 0 some_subckt")

    def test_get_system_without_outputs_has_zero_rows(self):
        """Test system with zero output correctly formed"""
        model = CircuitModel(
            parse_netlist(
                """
                V1 n1 0 1
                R1 n1 n2 1k
                C1 n2 0 1u
                """
            )
        )
        A, B, C, D = model.get_system()
        self.assertEqual(C.shape, (0, A.shape[0]))
        self.assertEqual(D.shape, (0, B.shape[1]))
        self.assertEqual(model.output_labels, [])

    def test_outputs_accumulate_for_all_supported_dipoles(self):
        """Test output currents can be selected as StateSpace output for all dipole types"""
        model = CircuitModel(
            parse_netlist(
                """
                V1 n1 0 1
                R1 n1 n2 10
                C1 n2 0 1u
                L1 n2 0 2m
                I1 n2 0 1
                """
            ),
        )

        model.add_node_voltage_output("n2")
        for name in ["R1", "L1", "C1", "V1", "I1"]:
            model.add_dipole_current_output(name)
        A, B, C, D = model.get_system()

        self.assertEqual(C.shape, (6, A.shape[0]))
        self.assertEqual(D.shape, (6, B.shape[1]))
        self.assertEqual(
            model.output_labels,
            ["V(n2)", "I(R1)", "I(L1)", "I(C1)", "I(V1)", "I(I1)"],
        )
        np.testing.assert_allclose(C[-1], 0.0)
        np.testing.assert_allclose(D[-1], [0.0, 1.0])

    def test_model_without_inputs_raises(self):
        """Test error in case of no input present in netlist (in the form of current or voltage source)"""
        with self.assertRaisesRegex(ValueError, "at least one independent"):
            CircuitModel(parse_netlist("R1 n1 0 1k\nC1 n1 0 1u"))

    def test_unknown_node_and_dipole_raise(self):
        """Test error in case of unknown dipole or node selected as output"""
        model = CircuitModel(parse_netlist("V1 n1 0 1\nR1 n1 n2 1k\nC1 n2 0 1u"))
        with self.assertRaisesRegex(ValueError, "Unknown node"):
            model.add_node_voltage_output("missing")
        with self.assertRaisesRegex(ValueError, "Unknown dipole"):
            model.add_dipole_current_output("R404")

# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
