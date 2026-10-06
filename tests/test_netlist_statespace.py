########################################################################################
##
##                                  TESTS FOR
##                              'netlist_to_statespace.py'
##
########################################################################################

# IMPORTS ==============================================================================
import unittest
from pathlib import Path
import numpy as np
import importlib.util
TEST_DIR = Path(__file__).parent

from pathsim_rf.netlist_to_statespace import CircuitModel, NetlistStateSpace, parse_netlist_file, parse_netlist
from pathsim.blocks.lti import StateSpace


# TESTS ================================================================================

class TestNetlistStateSpace(unittest.TestCase):
    """Test the NetlistStateSpace block"""

    def test_path_string_matches_manual_circuit_model(self):
        """Test CircuitModel is correctly used within NetlistStateSpace"""
        netlist_path = TEST_DIR / "filter.net"
        model = CircuitModel(parse_netlist_file(netlist_path))
        model.add_node_voltage_output("n1")
        model.add_dipole_current_output("Rload")
        A, B, C, D = model.get_system()

        block = NetlistStateSpace(netlist_path, output_voltages=["n1"], output_currents=["Rload"])

        self.assertIsInstance(block, StateSpace)
        np.testing.assert_allclose(block.A, A)
        np.testing.assert_allclose(block.B, B)
        np.testing.assert_allclose(block.C, C)
        np.testing.assert_allclose(block.D, D)
        self.assertEqual(block.state_labels, model.state_labels)
        self.assertEqual(block.input_labels, ["V1"])
        self.assertEqual(block.output_labels, ["V(n1)", "I(Rload)"])

    def test_path_object_preserves_feedback_source_input(self):
        """Test current sources feedback used to chain models"""
        block = NetlistStateSpace(TEST_DIR / "filter_without_load.net",
                                  output_voltages=["n1"],
                                  output_currents=["Lfilter"])

        self.assertEqual(block.input_labels, ["V1", "Bload"])
        self.assertEqual(block.output_labels, ["V(n1)", "I(Lfilter)"])
        self.assertEqual(len(block.inputs), 2)
        self.assertEqual(len(block.outputs), 2)

    def test_inline_netlist_string(self):
        """Test direct netlist use to class for instance generation"""
        block = NetlistStateSpace(
            """
            V1 in 0 1
            R1 in out 10
            C1 out 0 1u
            """,
            output_voltages=["out"],
            output_currents=["R1", "V1"])

        self.assertEqual(block.input_labels, ["V1"])
        self.assertEqual(block.output_labels, ["V(out)", "I(R1)", "I(V1)"])
        self.assertEqual(block.C.shape[0], 3)
        self.assertEqual(block.D.shape[0], 3)

    def test_explicit_missing_path_raises(self):
        """Test missing .net file provided"""
        with self.assertRaises(FileNotFoundError):
            NetlistStateSpace(TEST_DIR / "missing.net")
    
    def test_path_str_import(self):
        """Test str netlist_path arg allows file to be imported"""
        netlist_path = TEST_DIR / "filter.net"
        NetlistStateSpace(str(netlist_path))

    def test_invalid_output_names_raise(self):
        """Test invalid output names provided for state space generated"""
        netlist_path = TEST_DIR / "filter.net"
        with self.assertRaisesRegex(ValueError, "Unknown node"):
            NetlistStateSpace(netlist_path, output_voltages=["missing"])
        with self.assertRaisesRegex(ValueError, "Unknown dipole"):
            NetlistStateSpace(netlist_path, output_currents=["R404"])


# RUN TESTS LOCALLY ====================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
