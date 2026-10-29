#########################################################################################
##
##                          SPARSE ELECTRICAL CIRCUIT BLOCK
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import scipy.sparse as sp

from pathsim.blocks.dae import SparseLinearDAE


# HELPERS ===============================================================================

def _source_matrix(builder):
    """Superposition matrix 'B' of an ElecSolver 'TemporalSystemBuilder'.

    Column 'k' is the right hand side of source 'k' at unit amplitude (current
    sources first, then voltage sources, each in the order they were added), so
    that 'rhs(t) = B @ u(t)'. The columns are taken from the 'build_second_member'
    method of the builder, so the removed node equations and the ground handling
    stay consistent with the system matrices. The source amplitudes of the
    builder are restored afterwards.

    Parameters
    ----------
    builder : TemporalSystemBuilder
        circuit builder, after 'build_system()'

    Returns
    -------
    B : scipy.sparse.csr_matrix, shape (n, number of sources)
        superposition matrix, one column per source
    """
    n = builder.number_intensities + builder.size
    n_cs = builder.current_source_data.shape[0]
    n_vs = builder.voltage_source_data.shape[0]

    saved = builder.current_source_data.copy(), builder.voltage_source_data.copy()
    cols = []
    try:
        for k in range(n_cs + n_vs):
            cs, vs = np.zeros(n_cs), np.zeros(n_vs)
            if k < n_cs: cs[k] = 1.0
            else:        vs[k - n_cs] = 1.0
            builder.current_source_data, builder.voltage_source_data = cs, vs
            data, (rows,) = builder.build_second_member(check=False)
            col = np.zeros(n)
            np.add.at(col, rows, data)
            cols.append(sp.csc_matrix(col.reshape(-1, 1)))
    finally:
        builder.current_source_data, builder.voltage_source_data = saved
        builder.build_second_member(check=False)

    return sp.hstack(cols, format="csr") if cols else sp.csr_matrix((n, 0))


# BLOCKS ================================================================================

def _import_elecsolver():
    """Import the netlist parser of `ElecSolver`"""
    try:
        from ElecSolver import NetlistParser
    except ImportError as exc:
        raise ImportError(
            "'NetlistCircuit' needs the optional dependency 'ElecSolver', "
            "install it with 'pip install ElecSolver'"
            ) from exc
    return NetlistParser


class NetlistCircuit(SparseLinearDAE):
    """Sparse linear circuit block from a SPICE-like netlist, parsed and solved
    with `ElecSolver`.

    The block behaves like a state space model of the netlist, where all the
    independent sources are inputs and the node voltages and element currents
    are outputs, all of them addressable by their netlist names. The netlist is
    parsed by the ``NetlistParser`` of `ElecSolver` and the resulting circuit is
    integrated as a sparse linear DAE by :class:`.SparseLinearDAE`. Nothing is
    converted to a dense matrix, so large circuits are possible, and the
    integration is done by the engine of the simulation, with any solver.

    Supported elements are resistors ``R``, inductors ``L``, capacitors ``C``,
    magnetic couplings ``K`` and independent voltage ``V`` and current ``I``
    sources (SPICE syntax, SI prefixes and ``.param`` expressions are handled by
    the `ElecSolver` parser). Behavioral sources ``B<name> n1 n2 I=...`` or
    ``V=...`` are inputs as well, their expression is ignored, the signal is
    provided by the simulation. Everything else, such as ``.tran``, is ignored.
    Node ``0`` is the ground.

    **Inputs** are all the independent sources, named like in the netlist, all
    voltage sources first, then all current sources, each in netlist order. For
    a voltage source ``V1 a b`` the input is :math:`V_a - V_b` in V and for a
    current source ``I1 a b`` it is the current in A that flows from ``a`` to
    ``b`` through the source (SPICE convention).

    **Outputs** are all node voltages (without ground) named ``V(node)``,
    followed by all element currents named ``I(name)`` for all resistors,
    inductors, capacitors, voltage sources and current sources. The currents
    are positive when they flow from ``n1`` to ``n2`` as in the declaration of
    the element. The outputs can be restricted with `output_voltages` and
    `output_currents`.

    So the block is wired by name, for example ``circuit["V1"]`` for an input
    and ``circuit["V(n1)"]`` or ``circuit["I(Rload)"]`` for an output, see the
    example.

    Note
    ----
    The initial condition is chosen by the user and zero by default (no current
    in the inductors, no charge in the capacitors), see `initial_value`. The
    initial condition solver of `ElecSolver` is not used.

    The reduction of the DAE needs an index-1 circuit. Capacitor loops, such as
    parallel capacitors or a capacitor directly across an ideal voltage source,
    and inductor cut-sets, such as inductors in series without a resistive
    path, are rejected with a ``ValueError``, as are ideal magnetic couplings
    with :math:`|k| \\geq 1`, which make the inductance matrix singular
    (:math:`|k| < 1`, even very close to 1, is fine). Every part of the circuit needs a
    connection to ground ``0``.

    Example
    -------
    An LC filter with a resistive load, driven by a pulse source. The load is
    wired as a behavioral current source from the measured node voltage (an
    algebraic loop that the simulation resolves):

    .. code-block:: python

        from pathsim import Simulation, Connection
        from pathsim.blocks import NetlistCircuit, PulseSource, Amplifier, Scope

        netlist = \"\"\"
        V1 in 0 DC 0
        Lfilter in n1 0.5e-3
        Cfilter n1 0 50e-6
        Bload n1 0 I=V(n1)/50
        \"\"\"

        circuit = NetlistCircuit(netlist)

        src  = PulseSource(T=1/9000, amplitude=1, duty=0.5)
        load = Amplifier(gain=1/50)
        sco  = Scope(labels=["V(n1)", "I(Lfilter)"])

        sim = Simulation(
            [src, circuit, load, sco],
            [
                Connection(src, circuit["V1"]),
                Connection(circuit["V(n1)"], load, sco[0]),
                Connection(load, circuit["Bload"]),
                Connection(circuit["I(Lfilter)"], sco[1]),
            ]
        )

    Parameters
    ----------
    netlist : str, Path
        path of a netlist file (``.net``, ``.cir``, ``.sp``) or the netlist as
        text, a `str` that is an existing file or has one of these extensions
        is read as a file
    output_voltages : list[str], None
        nodes exposed as ``V(node)`` outputs, all nodes except ground if `None`
    output_currents : list[str], None
        elements exposed as ``I(name)`` outputs, all resistors, inductors,
        capacitors and sources if `None`
    initial_value : dict[str, float], None
        initial condition by name, for example ``{"V(n1)": 1.0, "I(Lfilter)":
        0.5}`` for a precharged capacitor and a current in the inductor. The
        keys are ``V(node)`` and ``I(name)`` of inductors, capacitors, resistors
        and voltage sources, everything that is not given is zero. Only the
        quantities that are states of the circuit have an effect (capacitor
        voltages through the node potentials, inductor currents). Zero if `None`
    x0 : array[float], None
        initial condition as the full `ElecSolver` vector (all branch currents,
        followed by all node potentials), takes precedence over `initial_value`

    Attributes
    ----------
    parser : NetlistParser
        the `ElecSolver` parser with the mapped netlist
    builder : TemporalSystemBuilder
        the `ElecSolver` circuit builder
    input_names : list[str]
        names of the inputs
    output_names : list[str]
        names of the outputs
    """

    def __init__(
        self,
        netlist,
        output_voltages=None,
        output_currents=None,
        initial_value=None,
        x0=None
        ):

        NetlistParser = _import_elecsolver()

        parser = NetlistParser(netlist)
        parser.map_netlist()
        self.parser = parser

        node_map = parser.node_map
        dipoles = parser.dipole_map

        for node in node_map:
            if node.lower() == "gnd":
                raise ValueError(
                    f"node '{node}' is not the ground, use node '0' as ground"
                    )

        cur = list(parser.current)
        vol = list(parser.voltage)

        if not cur and not vol:
            raise ValueError(
                "circuit must contain at least one independent voltage or current source"
                )

        #ideal coupling |k| >= 1 makes the inductance matrix singular
        for name, c in parser.coupling_map.items():
            i, j = c["L_coords"]
            Li = dipoles[list(parser.inductors)[i]]["value"]
            Lj = dipoles[list(parser.inductors)[j]]["value"]
            k = c["value"] / np.sqrt(Li * Lj)
            if abs(k) >= 1.0:
                raise ValueError(
                    f"coupling '{name}' has |k| = {abs(k):.6g} >= 1, the inductance "
                    f"matrix is singular, use |k| < 1"
                    )

        if not any(0 in d["nodes"] for d in dipoles.values()):
            raise ValueError("circuit has no connection to the ground node '0'")

        #circuit with sources, SPICE conventions: 'I a b' conducts from a to b
        #through the source, so it injects into b, and 'V a b' is V(a) - V(b)
        builder = parser.generate_temporal_system()
        for name in cur:
            a, b = dipoles[name]["nodes"]
            builder.add_current_source(1.0, b, a)
        for name in vol:
            a, b = dipoles[name]["nodes"]
            builder.add_voltage_source(1.0, a, b)
        builder.set_ground(0)
        builder.build_system()
        self.builder = builder

        S1, S2, _ = builder.get_system()
        n = S1.shape[0]

        #inputs in the order voltage sources, then current sources, the builder
        #superposition matrix has them as current sources, then voltage sources
        B = _source_matrix(builder)
        ncs = len(cur)
        B = sp.hstack([B[:, ncs:], B[:, :ncs]], format="csr")
        self.input_names = vol + cur
        col_in = {name: i for i, name in enumerate(self.input_names)}

        #layout of the solution vector
        nI = builder.number_intensities
        nL, nR, nC = len(parser.inductors), len(parser.resistors), len(parser.capacitors)
        sign = np.sign(builder.all_coords[1] - builder.all_coords[0]).astype(float)

        #index and sign of the branch currents in the solution vector
        branch = {}
        for kind, names, offset in (
            ("L", parser.inductors, 0),
            ("R", parser.resistors, nL),
            ("C", parser.capacitors, nL + nR),
            ("V", parser.voltage, nL + nR + nC)
            ):
            for k, name in enumerate(names):
                branch[name] = (offset + k, sign[offset + k])

        #outputs
        if output_voltages is None:
            output_voltages = [node for node in node_map if node != "0"]
        if output_currents is None:
            output_currents = (
                list(parser.resistors) + list(parser.inductors) +
                list(parser.capacitors) + vol + cur
                )

        rows, cols, vals = [], [], []
        drows, dcols = [], []
        labels = []

        for node in output_voltages:
            if node == "0":
                raise ValueError("the ground voltage 'V(0)' is not an output")
            if node not in node_map:
                raise ValueError(f"unknown node '{node}'")
            rows.append(len(labels)); cols.append(nI + node_map[node]); vals.append(1.0)
            labels.append(f"V({node})")

        for name in output_currents:
            if name in branch:
                idx, sgn = branch[name]
                rows.append(len(labels)); cols.append(idx); vals.append(sgn)
            elif name in parser.current:
                drows.append(len(labels)); dcols.append(col_in[name])
            else:
                raise ValueError(f"unknown element '{name}'")
            labels.append(f"I({name})")

        ny = len(labels)
        C = sp.csr_matrix((vals, (rows, cols)), shape=(ny, n))
        D = sp.csr_matrix((np.ones(len(drows)), (drows, dcols)), shape=(ny, len(self.input_names)))
        self.output_names = labels

        #initial condition by name
        if x0 is None and initial_value is not None:
            x0 = np.zeros(n)
            for key, val in dict(initial_value).items():
                if key.startswith("V(") and key.endswith(")") and key[2:-1] in node_map:
                    x0[nI + node_map[key[2:-1]]] = val
                elif key.startswith("I(") and key.endswith(")") and key[2:-1] in branch:
                    idx, sgn = branch[key[2:-1]]
                    x0[idx] = val * sgn
                else:
                    raise ValueError(f"unknown initial condition '{key}'")

        super().__init__(
            S1, S2, B, C, D,
            x0=x0,
            input_labels=self.input_names,
            output_labels=labels
            )
