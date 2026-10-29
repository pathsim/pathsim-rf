########################################################################################
##
##                                  TESTS FOR
##                         'NetlistCircuit' in 'blocks/circuit.py'
##
########################################################################################

# IMPORTS ==============================================================================

import os
import tempfile
import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Scope, Amplifier, PulseSource
from pathsim.solvers import BDF2, ESDIRK43, EUB
from pathsim_rf import NetlistCircuit

try:
    import ElecSolver
    HAS_ELECSOLVER = True
except ImportError:
    HAS_ELECSOLVER = False


# HELPERS ==============================================================================

R, C = 1e3, 1e-6
TAU = R * C

RC = """
V1 in 0 1
R1 in out 1k
C1 out 0 1u
"""

LC_LOAD = """
V1 in 0 1
Lfilter in n1 0.5e-3
Cfilter n1 0 50e-6
Bload n1 0 I=V(n1)/50
"""


def simulate(block, wiring, sources, T=5e-3, dt=1e-5, Solver=BDF2, **kw):
    """Simulate a circuit block, 'sources' is a dict input name -> callable and
    'wiring' a list of output names that are recorded."""
    srcs = [Source(f) for f in sources.values()]
    sco = Scope()
    conns = [Connection(s, block[name]) for s, name in zip(srcs, sources)]
    conns += [Connection(block[name], sco[i]) for i, name in enumerate(wiring)]
    sim = Simulation([block, sco] + srcs, conns, Solver=Solver, dt=dt, log=False, **kw)
    sim.run(T)
    return sco.read()


# TESTS ================================================================================

@unittest.skipUnless(HAS_ELECSOLVER, "ElecSolver not installed")
class TestNetlistCircuitPorts(unittest.TestCase):

    def test_default_ports(self):
        blk = NetlistCircuit(RC)
        self.assertEqual(blk.input_names, ["V1"])
        self.assertEqual(
            blk.output_names, ["V(in)", "V(out)", "I(R1)", "I(C1)", "I(V1)"]
            )
        self.assertEqual(len(blk.inputs), 1)
        self.assertEqual(len(blk.outputs), 5)

    def test_input_order_voltage_then_current_sources(self):
        net = RC + "I1 0 out 1m\nV2 x 0 0\nRx x out 1\n"
        blk = NetlistCircuit(net)
        self.assertEqual(blk.input_names, ["V1", "V2", "I1"])

    def test_restricted_outputs(self):
        blk = NetlistCircuit(RC, output_voltages=["out"], output_currents=["C1"])
        self.assertEqual(blk.output_names, ["V(out)", "I(C1)"])

        none = NetlistCircuit(RC, output_voltages=[], output_currents=["R1"])
        self.assertEqual(none.output_names, ["I(R1)"])

    def test_ports_addressable_by_name(self):
        blk = NetlistCircuit(RC)
        blk["V1"]
        blk["V(out)"]
        blk["I(R1)"]

    def test_netlist_from_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rc.cir")
            with open(path, "w") as f:
                f.write(RC)
            for arg in (path, __import__("pathlib").Path(path)):
                blk = NetlistCircuit(arg)
                self.assertEqual(blk.input_names, ["V1"])

    def test_params_and_si_prefixes(self):
        net = ".param Rv = {2*500}\nV1 in 0 1\nR1 in out {Rv}\nC1 out 0 1u\n"
        t, y = simulate(NetlistCircuit(net, output_voltages=["out"], output_currents=[]),
                        ["V(out)"], {"V1": lambda t: 1.0})
        self.assertLess(np.abs(y[0] - (1 - np.exp(-t / TAU))).max(), 1e-4)


@unittest.skipUnless(HAS_ELECSOLVER, "ElecSolver not installed")
class TestNetlistCircuitResponse(unittest.TestCase):

    def test_rc_step_response_and_currents(self):
        blk = NetlistCircuit(RC)
        names = ["V(in)", "V(out)", "I(R1)", "I(C1)", "I(V1)"]
        t, y = simulate(blk, names, {"V1": lambda t: 1.0}, Solver=ESDIRK43,
                        tolerance_lte_rel=1e-9, tolerance_lte_abs=1e-12)
        v = 1.0 - np.exp(-t / TAU)
        i = np.exp(-t / TAU) / R
        np.testing.assert_allclose(y[0], 1.0, atol=1e-9)
        np.testing.assert_allclose(y[1], v, atol=1e-6)
        #currents are positive from n1 to n2 of each element
        np.testing.assert_allclose(y[2], i, atol=1e-9)
        np.testing.assert_allclose(y[3], i, atol=1e-9)
        #the source current flows through the source from n1 to n2, so the circuit is fed with -I(V1)
        np.testing.assert_allclose(y[4], -i, atol=1e-9)

    def test_current_source_sign_and_output(self):
        #'I1 0 n 2m' conducts from node 0 to node n through the source, injecting into n
        net = "I1 0 n 1\nR1 n 0 10\nC1 n 0 1u\n"
        blk = NetlistCircuit(net)
        self.assertEqual(blk.input_names, ["I1"])
        t, y = simulate(blk, ["V(n)", "I(I1)", "I(R1)"], {"I1": lambda t: 2.0},
                        Solver=ESDIRK43, T=1e-3, dt=1e-6,
                        tolerance_lte_rel=1e-9, tolerance_lte_abs=1e-12)
        np.testing.assert_allclose(y[1], 2.0, atol=1e-12)                 #I(I1) is the input
        np.testing.assert_allclose(y[0][-1], 20.0, rtol=1e-3)             #V(n) -> I R
        np.testing.assert_allclose(y[2], y[0] / 10.0, atol=1e-9)         #R1 n -> 0

    def test_reversed_node_order_signs(self):
        #elements declared from the higher node index to the lower one
        net = "V1 a 0 1\nR1 b a 1k\nC1 0 b 1u\n"     #'R1 b a': current b -> a, 'C1 0 b': 0 -> b
        blk = NetlistCircuit(net)
        t, y = simulate(blk, ["V(b)", "I(R1)", "I(C1)"], {"V1": lambda t: 1.0},
                        Solver=ESDIRK43, tolerance_lte_rel=1e-9, tolerance_lte_abs=1e-12)
        i = np.exp(-t / TAU) / R
        np.testing.assert_allclose(y[0], 1 - np.exp(-t / TAU), atol=1e-6)
        #current charges the cap from a through R: R1 declared b -> a is negative
        np.testing.assert_allclose(y[1], -i, atol=1e-9)
        #C1 declared 0 -> b, the charging current flows b -> 0 through it: negative
        np.testing.assert_allclose(y[2], -i, atol=1e-9)

    def test_initial_value_by_name(self):
        net = "V1 in 0 0\nR1 in out 1k\nC1 out 0 1u\n"
        blk = NetlistCircuit(net, output_voltages=["out"], output_currents=[],
                             initial_value={"V(out)": 2.0})
        t, y = simulate(blk, ["V(out)"], {"V1": lambda t: 0.0})
        self.assertLess(np.abs(y[0] - 2.0 * np.exp(-t / TAU)).max(), 1e-3)

        #default is all zeros
        t, y = simulate(NetlistCircuit(net, output_voltages=["out"], output_currents=[]),
                        ["V(out)"], {"V1": lambda t: 0.0})
        self.assertEqual(np.abs(y[0]).max(), 0.0)

    def test_inductor_initial_current(self):
        net = "V1 in 0 0\nR1 in n 10\nL1 n 0 10m\n"      #tau = L/R = 1ms
        blk = NetlistCircuit(net, output_voltages=[], output_currents=["L1"],
                             initial_value={"I(L1)": 3.0})
        t, y = simulate(blk, ["I(L1)"], {"V1": lambda t: 0.0}, T=3e-3, dt=1e-6)
        self.assertLess(np.abs(y[0] - 3.0 * np.exp(-t / 1e-3)).max(), 1e-3)

    def test_full_x0_vector(self):
        net = "V1 in 0 0\nR1 in out 1k\nC1 out 0 1u\n"
        blk = NetlistCircuit(net)
        x0 = np.zeros(blk.n)
        x0[blk.builder.number_intensities + blk.parser.node_map["out"]] = 1.0
        blk2 = NetlistCircuit(net, x0=x0)
        np.testing.assert_allclose(blk2.initial_value, NetlistCircuit(
            net, initial_value={"V(out)": 1.0}).initial_value)

    def test_load_loop_wired_by_name(self):
        #behavioral load source fed back from the measured node voltage, same circuit as
        #an explicit load resistor
        explicit = LC_LOAD.replace("Bload n1 0 I=V(n1)/50", "Rload n1 0 50")

        def pulse():
            return PulseSource(T=1 / 9000, amplitude=1, duty=0.5, t_rise=1e-8, t_fall=1e-8)

        #reference: explicit resistor
        ref = NetlistCircuit(explicit, output_voltages=["n1"], output_currents=["Lfilter"])
        src, sco = pulse(), Scope()
        sim = Simulation(
            [src, ref, sco],
            [Connection(src, ref["V1"]), Connection(ref["V(n1)"], sco[0]),
             Connection(ref["I(Lfilter)"], sco[1])],
            Solver=EUB, dt=5e-7, log=False
            )
        sim.run(1e-3)
        _, y_ref = sco.read()

        #behavioral source closed through an amplifier with the algebraic loop
        blk = NetlistCircuit(LC_LOAD)
        src, load, sco = pulse(), Amplifier(gain=1 / 50), Scope()
        sim = Simulation(
            [src, blk, load, sco],
            [Connection(src, blk["V1"]),
             Connection(blk["V(n1)"], load, sco[0]),
             Connection(load, blk["Bload"]),
             Connection(blk["I(Lfilter)"], sco[1])],
            Solver=EUB, dt=5e-7, log=False
            )
        sim.run(1e-3)
        _, y = sco.read()

        np.testing.assert_allclose(y[0], y_ref[0], atol=1e-9)
        np.testing.assert_allclose(y[1], y_ref[1], atol=1e-9)

    def test_mutual_coupling_transient(self):
        #magnetically coupled inductors, the second one is excited through the coupling
        a = "V1 in 0 0\nR1 in p 1\nL1 p 0 1m\nL2 q 0 1m\nR2 q 0 1\nK1 L1 L2 0.5\n"
        blk = NetlistCircuit(a, output_voltages=[], output_currents=["L1", "L2"])
        t, y = simulate(blk, ["I(L1)", "I(L2)"], {"V1": lambda t: 1.0}, T=30e-3, dt=1e-4)
        #steady state: L1 is a DC short, so i1 = V / R1, the mutual coupling only drives
        #a transient in the second inductor, which decays to zero
        self.assertAlmostEqual(y[0][-1], 1.0, places=3)
        self.assertAlmostEqual(y[1][-1], 0.0, places=3)
        self.assertGreater(np.abs(y[1]).max(), 1e-3)


@unittest.skipUnless(HAS_ELECSOLVER, "ElecSolver not installed")
class TestNetlistCircuitErrors(unittest.TestCase):

    def test_no_sources(self):
        with self.assertRaises(ValueError):
            NetlistCircuit("R1 a 0 1k\nC1 a 0 1u\n")

    def test_gnd_alias_rejected(self):
        with self.assertRaises(ValueError):
            NetlistCircuit("V1 a gnd 1\nR1 a gnd 1k\nC1 a gnd 1u\n")

    def test_no_ground_connection(self):
        with self.assertRaises(ValueError):
            NetlistCircuit("V1 a b 1\nR1 a b 1k\nC1 a b 1u\n")

    def test_unknown_outputs(self):
        with self.assertRaises(ValueError):
            NetlistCircuit(RC, output_voltages=["nope"])
        with self.assertRaises(ValueError):
            NetlistCircuit(RC, output_currents=["Rnope"])
        with self.assertRaises(ValueError):
            NetlistCircuit(RC, output_voltages=["0"])

    def test_unknown_initial_condition(self):
        with self.assertRaises(ValueError):
            NetlistCircuit(RC, initial_value={"V(nope)": 1.0})
        with self.assertRaises(ValueError):
            NetlistCircuit(RC, initial_value={"foo": 1.0})

    def test_parallel_capacitors_rejected(self):
        net = "V1 in 0 1\nR1 in a 1k\nC1 a 0 1u\nC2 a 0 1u\n"
        with self.assertRaises(ValueError):
            NetlistCircuit(net)

    def test_ideal_coupling_rejected(self):
        net = "V1 in 0 1\nR1 in p 1\nL1 p 0 1m\nL2 q 0 1m\nR2 q 0 1\nK1 L1 L2 {k}\n"
        for k in ("1", "-1", "1.2"):
            with self.assertRaises(ValueError):
                NetlistCircuit(net.format(k=k))
        #strong but not ideal coupling works
        NetlistCircuit(net.format(k="0.999999"))

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            NetlistCircuit(42)


class TestNetlistCircuitDependency(unittest.TestCase):

    @unittest.skipIf(HAS_ELECSOLVER, "ElecSolver installed")
    def test_missing_elecsolver(self):
        with self.assertRaises(ImportError):
            NetlistCircuit(RC)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
