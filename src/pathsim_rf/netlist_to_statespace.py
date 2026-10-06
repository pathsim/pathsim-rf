"""
netlist_to_statespace.py

Turns a small SPICE-like netlist of R, L, C, K (mutual inductance / coupling
coefficient), V (independent voltage source) and I (independent current
source) elements into an explicit LTI state-space model

    xdot = A x + B u
    y    = C x + D u

via Modified Nodal Analysis (MNA) + elimination of the algebraic
(non-storage) unknowns. This is the standard "state-variable method" used
to derive circuit ODEs by hand, automated in code so it scales to
messy topologies and handles coupled inductors correctly.

Only LINEAR elements are supported (no diodes/switches/etc - PathSim itself
handles those fine via its algebraic-loop solver and Function blocks, see
the accompanying explanation).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

from pathsim.blocks.lti import StateSpace


# --------------------------------------------------------------------------
# Netlist parsing
# --------------------------------------------------------------------------

@dataclass
class Element:
    kind: str      # 'R','L','C','V','I','K'
    name: str
    n1: str = None
    n2: str = None
    value: float = None
    # for K elements:
    l1: str = None
    l2: str = None


SI_MULTIPLIERS = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "K": 1e3,
    "meg": 1e6,
    "Meg": 1e6,
    "M": 1e6,
    "g": 1e9,
    "G": 1e9,
    "t": 1e12,
    "T": 1e12,
}

_PARAM_PATTERN = re.compile(r"^\.param\s+([A-Za-z_]\w*)\s*=\s*(.+)$")
_VALUE_PATTERN = re.compile(
    r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)([A-Za-zµ]*)\s*$"
)
_VALID_KINDS = {"R", "L", "C", "V", "I", "K"}
_IGNORED_LINE_STARTS = (".", "*", '"')


def _normalize_node(node: str) -> str:
    """Normalize ground aliases to canonical node '0'."""
    return "0" if node.strip().lower() in {"0", "gnd"} else node


def _get_si_multiplier(suffix: str) -> float:
    """Return numeric multiplier for an SI prefix/suffix string."""
    if suffix in SI_MULTIPLIERS:
        return SI_MULTIPLIERS[suffix]
    if suffix.lower() in SI_MULTIPLIERS:
        return SI_MULTIPLIERS[suffix.lower()]

    candidates = sorted(SI_MULTIPLIERS.keys(), key=len, reverse=True)
    for key in candidates:
        if not key:
            continue
        if suffix.startswith(key):
            return SI_MULTIPLIERS[key]
        low_key = key.lower()
        if suffix.lower().startswith(low_key):
            return SI_MULTIPLIERS[low_key]
    raise ValueError(f"Unknown SI prefix: '{suffix}'")


def parse_value_with_units(value_str: str, params: dict[str, str] | None = None) -> float:
    """
    Parse SPICE-like numeric tokens with SI prefixes and optional unit tails.

    Supported forms include:
    - plain/scientific floats: ``1``, ``100.e-3``, ``2.2e6``
    - SI-prefixed tokens: ``4u``, ``10k``, ``3Meg``, ``2.2kOhm``, ``1mH``
    - parameter references: ``{RMAIN}`` when ``params`` is provided
    """
    if value_str is None:
        raise ValueError("Missing value")

    value = value_str.strip()
    try:
        return float(value)
    except ValueError:
        pass

    if value.startswith("{") and value.endswith("}"):
        if params is None:
            raise ValueError(f"Unknown parameter reference: '{value}'")
        name = value[1:-1].strip()
        if name not in params:
            raise ValueError(f"Unknown parameter: '{name}'")
        return parse_value_with_units(params[name], params)

    match = _VALUE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid value format: '{value_str}'")

    number, suffix = match.groups()
    return float(number) * _get_si_multiplier(suffix)


def _parse_optional_source_value(raw: str, params: dict[str, str]) -> float | None:
    """Parse source literal values, returning None for waveform expressions."""
    try:
        return parse_value_with_units(raw, params)
    except ValueError:
        # Many source statements use waveforms (PWL, SIN, EXP, etc.); those
        # are runtime waveforms and do not affect linearization here.
        return None


def _parse_behavioral_source(parts: list[str], line_number: int, line: str) -> Element:
    """
    Parse LTspice behavioral source shorthand:
      Bx n+ n- I=<expr>  -> modeled as current source input placeholder
      Bx n+ n- V=<expr>  -> modeled as voltage source input placeholder
    """
    if len(parts) < 4:
        raise ValueError(f"Malformed behavioral source at line {line_number}: '{line}'")

    name = parts[0]
    n1, n2 = _normalize_node(parts[1]), _normalize_node(parts[2])
    expr = " ".join(parts[3:]).strip()
    expr_upper = expr.upper()
    if expr_upper.startswith("I="):
        return Element(kind="I", name=name, n1=n1, n2=n2, value=None)
    if expr_upper.startswith("V="):
        return Element(kind="V", name=name, n1=n1, n2=n2, value=None)
    raise ValueError(
        f"Unsupported behavioral source expression at line {line_number}: '{line}'"
    )


def parse_netlist(text: str) -> list[Element]:
    """
    Parse netlist text into linear element records.

    One element per line, whitespace separated; '#' or ';' starts a comment.
      R<name> n1 n2 value
      L<name> n1 n2 value
      C<name> n1 n2 value
      V<name> n+ n- value      (value is a placeholder; real waveform comes
                                 from the PathSim Source block at sim time)
      I<name> n+ n- value      (same)
      B<name> n+ n- I=<expr>   (behavioral current source placeholder)
      B<name> n+ n- V=<expr>   (behavioral voltage source placeholder)
      K<name> Lname1 Lname2 k  (coupling coefficient, -1<=k<=1)
    Node '0' (or 'gnd') is ground.
    Deck/meta lines beginning with '.', '*', or a quoted header are ignored.
    """
    params: dict[str, str] = {}
    raw_element_lines: list[tuple[int, str]] = []

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#")[0].split(";")[0].strip()
        if not line:
            continue
        match = _PARAM_PATTERN.fullmatch(line)
        if match is not None:
            params[match.group(1)] = match.group(2).strip()
            continue
        raw_element_lines.append((line_number, line))

    elements: list[Element] = []
    for line_number, line in raw_element_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith(_IGNORED_LINE_STARTS):
            continue

        name = parts[0]
        kind = name[0].upper()
        if kind == "B":
            elements.append(_parse_behavioral_source(parts, line_number, line))
            continue

        if kind not in _VALID_KINDS:
            raise ValueError(
                f"Unsupported element '{name}' at line {line_number}: '{line}'"
            )

        if kind == 'K':
            if len(parts) < 4:
                raise ValueError(f"Malformed K element at line {line_number}: '{line}'")
            elements.append(Element(kind='K', name=name, l1=parts[1], l2=parts[2],
                                     value=parse_value_with_units(parts[3], params)))
        else:
            if len(parts) < 3:
                raise ValueError(f"Malformed element at line {line_number}: '{line}'")
            n1, n2 = _normalize_node(parts[1]), _normalize_node(parts[2])
            val_token = " ".join(parts[3:]) if len(parts) > 3 else None
            if kind in {"R", "L", "C"}:
                if val_token is None:
                    raise ValueError(f"Missing value for element '{name}' at line {line_number}")
                val = parse_value_with_units(val_token, params)
            else:
                val = _parse_optional_source_value(val_token, params) if val_token else None
            elements.append(Element(kind=kind, name=name, n1=n1, n2=n2, value=val))
    return elements


def parse_netlist_file(path: str | Path) -> list[Element]:
    """Load and parse a netlist file."""
    with open(path, "r", encoding="utf-8") as handle:
        return parse_netlist(handle.read())


GND = {'0', 'gnd', 'GND'}


# --------------------------------------------------------------------------
# MNA DAE -> explicit state space reduction
# --------------------------------------------------------------------------

class CircuitModel:
    """
    Build an explicit state-space model from a linear circuit netlist.

    The circuit is first stamped as the modified nodal analysis (MNA) system

    ``E * dz/dt + G * z = B * u``,

    where ``z`` contains node voltages, inductor currents, and ideal voltage
    source currents. The MNA unknowns are partitioned into differential states
    ``x_d`` and algebraic unknowns ``x_a``. Eliminating ``x_a`` gives

    ``x_a = Phi * x_d + Psi * u``

    and the final explicit model

    ``dx_d/dt = A * x_d + B_ss * u``.

    The MNA matrices are stamped directly into floating-point arrays and
    reduced using SciPy sparse LU solves. Singular solves are retried with
    bounded diagonal regularization and emit :class:`RuntimeWarning` when
    regularization is used.

    Parameters
    ----------
    elements:
        Output of :func:`parse_netlist` / :func:`parse_netlist_file`.

    Attributes
    ----------
    A:
        State matrix of the reduced explicit system.
    B_ss:
        Input matrix of the reduced explicit system.
    Phi:
        Map from differential states to eliminated algebraic variables.
    Psi:
        Map from inputs to eliminated algebraic variables.
    state_labels:
        Labels matching the row and column ordering of the state matrices.
    """

    def __init__(self, elements: list[Element]):

        self.elements = elements

        self.R = [e for e in elements if e.kind == 'R']
        self.L = [e for e in elements if e.kind == 'L']
        self.C = [e for e in elements if e.kind == 'C']
        self.V = [e for e in elements if e.kind == 'V']
        self.I = [e for e in elements if e.kind == 'I']
        self.K = [e for e in elements if e.kind == 'K']
        self.dipoles_by_name = {
            e.name: e for e in elements if e.kind in {"R", "L", "C", "V", "I"}
        }

        # ---- nodes ----
        nodes = []
        for e in elements:
            if e.kind == 'K':
                continue
            for n in (e.n1, e.n2):
                if n not in GND and n not in nodes:
                    nodes.append(n)
        self.nodes = nodes                      # ordered list of non-ground node names
        self.node_idx = {n: i for i, n in enumerate(nodes)}
        self.n_nodes = len(nodes)

        self.L_names = [e.name for e in self.L]
        self.L_idx = {name: i for i, name in enumerate(self.L_names)}
        self.n_L = len(self.L)

        self.V_names = [e.name for e in self.V]
        self.V_idx = {name: i for i, name in enumerate(self.V_names)}
        self.n_V = len(self.V)

        # unknown ordering: [node voltages] + [inductor currents] + [V-source currents]
        self.n_total = self.n_nodes + self.n_L + self.n_V

        def col_node(n):
            return None if n in GND else self.node_idx[n]

        def col_iL(name):
            return self.n_nodes + self.L_idx[name]

        def col_iV(name):
            return self.n_nodes + self.n_L + self.V_idx[name]

        self._col_node, self._col_iL, self._col_iV = col_node, col_iL, col_iV

        # ---- inductance matrix (with mutual terms) ----
        Lmat = np.zeros((self.n_L, self.n_L), dtype=float)
        l_values = {e.name: e.value for e in self.L}
        for e in self.L:
            i = self.L_idx[e.name]
            Lmat[i, i] = float(e.value)
        for k in self.K:
            i, j = self.L_idx[k.l1], self.L_idx[k.l2]
            Li = l_values[k.l1]
            Lj = l_values[k.l2]
            M = float(k.value) * np.sqrt(float(Li) * float(Lj))
            Lmat[i, j] += M
            Lmat[j, i] += M
        self.Lmat = Lmat

        # ---- inputs: one column per V source, then one column per I source ----
        self.input_names = self.V_names + [e.name for e in self.I]
        self.n_u = len(self.input_names)
        if self.n_u == 0:
            raise ValueError(
                "Circuit must contain at least one independent voltage or current source."
            )

        # ---- build E (dynamic) and G (algebraic) matrices, and B (input map) ----
        n = self.n_total
        E = np.zeros((n, n), dtype=float)
        G = np.zeros((n, n), dtype=float)
        B = np.zeros((n, self.n_u), dtype=float)

        def stamp_G(row, col, val):
            if row is not None and col is not None:
                G[row, col] += val

        # Resistors: contribute to node KCL rows only
        for e in self.R:
            a, b = col_node(e.n1), col_node(e.n2)
            g = 1.0 / float(e.value)
            stamp_G(a, a, g); stamp_G(b, b, g)
            stamp_G(a, b, -g); stamp_G(b, a, -g)

        # Capacitors: contribute dv/dt terms to node KCL rows (the E matrix)
        for e in self.C:
            a, b = col_node(e.n1), col_node(e.n2)
            c = float(e.value)
            if a is not None: E[a, a] += c
            if b is not None: E[b, b] += c
            if a is not None and b is not None:
                E[a, b] -= c
                E[b, a] -= c

        # Inductors: KCL stamp (current unknown enters/leaves nodes) +
        # dedicated branch row  v_na - v_nb - sum_j Lij * d(iLj)/dt = 0
        for e in self.L:
            a, b = col_node(e.n1), col_node(e.n2)
            iL_col = col_iL(e.name)
            row = iL_col  # branch row shares index with its current unknown
            stamp_G(a, iL_col, 1)
            stamp_G(b, iL_col, -1)
            if a is not None: G[row, a] += 1
            if b is not None: G[row, b] -= 1
            # branch eqn: v_na - v_nb - L*d(iL)/dt - sum_j M_ij*d(iLj)/dt = 0
            # => E[row, iLj] = -Lij  (note the minus sign!)
            i = self.L_idx[e.name]
            for j, name_j in enumerate(self.L_names):
                Lij = self.Lmat[i, j]
                if Lij != 0:
                    E[row, col_iL(name_j)] -= Lij

        # Voltage sources: KCL stamp + branch row v_na - v_nb = u(t)
        for e in self.V:
            a, b = col_node(e.n1), col_node(e.n2)
            iV_col = col_iV(e.name)
            row = iV_col
            stamp_G(a, iV_col, 1)
            stamp_G(b, iV_col, -1)
            if a is not None: G[row, a] += 1
            if b is not None: G[row, b] -= 1
            u_col = self.input_names.index(e.name)
            B[row, u_col] = 1

        # Current sources: pure RHS injection into node KCL rows.
        # Convention: positive I flows from n1 -> n2 *through the source*,
        # i.e. it delivers current INTO n2 and draws it OUT of n1 from the
        # external circuit's point of view.
        for e in self.I:
            a, b = col_node(e.n1), col_node(e.n2)
            u_col = self.input_names.index(e.name)
            if a is not None: B[a, u_col] -= 1
            if b is not None: B[b, u_col] += 1

        self.E, self.G, self.B = E, G, B

        # ---- differential / algebraic partition ----
        diff_rows = [r for r in range(n) if any(E[r, c] != 0 for c in range(n))]
        alg_rows = [r for r in range(n) if r not in diff_rows]
        self.diff_rows, self.alg_rows = diff_rows, alg_rows

        # sanity: the state columns should be exactly node-voltages that own a
        # nonzero E column plus all inductor currents; algebraic columns = rest
        diff_cols = sorted(set(c for r in diff_rows for c in range(n) if E[r, c] != 0))
        alg_cols = [c for c in range(n) if c not in diff_cols]
        if len(diff_cols) != len(diff_rows) or len(alg_cols) != len(alg_rows):
            raise ValueError(
                "Circuit is degenerate for this reduction (e.g. an all-capacitor "
                "loop, an all-inductor cutset, or a floating node). "
                f"diff_rows={len(diff_rows)} diff_cols={len(diff_cols)} "
                f"alg_rows={len(alg_rows)} alg_cols={len(alg_cols)}"
            )
        self.diff_cols, self.alg_cols = diff_cols, alg_cols

        self._set_state_labels()
        self._reduce()
        self._selected_output_rows: list[tuple[np.ndarray, np.ndarray]] = []
        self.output_labels: list[str] = []

    def _set_state_labels(self):
        """Set labels for the differential state vector in `self.diff_cols` order."""
        dc = self.diff_cols
        labels = []
        for c in dc:
            if c < self.n_nodes:
                labels.append(f"v_{self.nodes[c]}")
            else:
                labels.append(f"i_{self.L_names[c - self.n_nodes]}")
        self.state_labels = labels
        self.state_label_idx = {label: i for i, label in enumerate(labels)}

    def _raise_algebraic_singular(self):
        raise ValueError(
            "Algebraic subsystem is singular. Usually means a loop made "
            "purely of ideal voltage sources (and/or 0-ohm shorts), or a "
            "node with no DC path to ground."
        )

    def _raise_state_singular(self):
        raise ValueError(
            "State/mass matrix is singular: this circuit's capacitor "
            "voltages (or inductor currents) aren't independent, so the "
            "naive 'one state per capacitor-touched node' selection "
            "over-counted states. Classic cause: a capacitor whose *both* "
            "terminals only reach the rest of the circuit through that "
            "same capacitor (no other cap ties either node down "
            "independently) -- e.g. 'R1 in a / C1 a b / R2 b 0' with "
            "nothing else at a or b. Only one of v_a, v_b is really an "
            "independent state there; the other is pinned by KCL. This "
            "reduction doesn't do full tree/cotree state selection, so it "
            "can't detect that automatically yet. Workarounds: (1) add a "
            "TINY STRAY CAPACITANCE FROM ONE OF THE FLOATING NODES TO "
            "GROUND (not a resistor -- the degeneracy lives in the "
            "capacitor/mass matrix, a parallel resistor doesn't touch it "
            "at all). e.g. 'Cstray b 0 1e-15' -- this gives that node's "
            "own row independent rank; it adds one extra, extremely fast "
            "eigenvalue (a numerical artifact, orders of magnitude faster "
            "than your real dynamics) alongside the correct physical "
            "pole(s), or (2) hand-pick the true independent capacitor "
            "voltage as the state and eliminate the redundant node "
            "yourself before building the netlist."
        )

    @staticmethod
    def _safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiply without relying on NumPy BLAS-backed matmul.
        This avoids environment-specific native linear algebra crashes.
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: {a.shape} and {b.shape}")
        out = np.zeros((a.shape[0], b.shape[1]), dtype=float)
        for i in range(a.shape[0]):
            for k in range(a.shape[1]):
                aik = a[i, k]
                if aik != 0.0:
                    out[i, :] += aik * b[k, :]
        return out

    @staticmethod
    def _solve_with_scipy(mat: np.ndarray, rhs: np.ndarray, context: str) -> np.ndarray:
        """Solve right-hand sides with SciPy sparse LU and bounded regularization."""
        import scipy.sparse as sps
        from scipy.sparse.linalg import splu

        rhs_2d = rhs if rhs.ndim == 2 else rhs.reshape((-1, 1))
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"{context} matrix must be square, got {mat.shape}")
        if rhs_2d.shape[0] != mat.shape[0]:
            raise ValueError(
                f"{context} rhs has incompatible shape {rhs_2d.shape} for matrix {mat.shape}"
            )

        last_error: Exception | None = None
        for eps in (0.0, 1e-15, 1e-12, 1e-9, 1e-6):
            try:
                matrix = mat if eps == 0.0 else mat + eps * np.eye(mat.shape[0])
                factor = splu(sps.csc_matrix(matrix))
                columns = [
                    np.asarray(factor.solve(np.asarray(column, dtype=float)), dtype=float)
                    for column in rhs_2d.T
                ]
                solution = np.column_stack(columns)
                if eps > 0.0:
                    warnings.warn(
                        f"Circuit reduction regularized singular {context} with eps={eps:.1e}",
                        RuntimeWarning,
                    )
                return solution if rhs.ndim == 2 else solution[:, 0]
            except (RuntimeError, ValueError) as exc:
                last_error = exc

        raise ValueError(f"Failed to solve {context} with SciPy sparse LU.") from last_error

    def _reduce(self):
        """Eliminate algebraic variables and solve the mass system with SciPy."""
        E, G, B = self.E, self.G, self.B
        dr, ar = self.diff_rows, self.alg_rows
        dc, ac = self.diff_cols, self.alg_cols

        E_dd = E[np.ix_(dr, dc)]
        G_alg_d = G[np.ix_(ar, dc)]
        G_alg_a = G[np.ix_(ar, ac)]
        G_diff_d = G[np.ix_(dr, dc)]
        G_diff_a = G[np.ix_(dr, ac)]
        B_alg = B[ar, :]
        B_diff = B[dr, :]

        try:
            Phi = -self._solve_with_scipy(G_alg_a, G_alg_d, "algebraic subsystem")
            Psi = self._solve_with_scipy(G_alg_a, B_alg, "algebraic subsystem")
        except ValueError:
            self._raise_algebraic_singular()

        try:
            A = -self._solve_with_scipy(
                E_dd,
                G_diff_d + self._safe_matmul(G_diff_a, Phi),
                "state/mass subsystem",
            )
            Bmat = self._solve_with_scipy(
                E_dd,
                B_diff - self._safe_matmul(G_diff_a, Psi),
                "state/mass subsystem",
            )
        except ValueError:
            self._raise_state_singular()

        self.A = A
        self.B_ss = Bmat
        self.Phi = Phi
        self.Psi = Psi

    def _build_output_rows(
        self,
        coefficient_rows: list[dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project raw MNA output expressions to state-space rows.

        Each coefficient mapping uses keys of the form ``node:name``,
        ``iL:name``, or ``iV:name``. Algebraic unknowns are substituted through
        ``Phi`` and ``Psi`` so each result satisfies ``y = C*x + D*u``.
        """
        if not coefficient_rows:
            return (
                np.empty((0, len(self.state_labels)), dtype=float),
                np.empty((0, self.n_u), dtype=float),
            )

        vec = np.zeros((len(coefficient_rows), self.n_total), dtype=float)
        for row, coeffs in enumerate(coefficient_rows):
            for key, val in coeffs.items():
                try:
                    typ, name = key.split(":", 1)
                except ValueError as exc:
                    raise ValueError(f"Invalid output coefficient key '{key}'") from exc

                if typ == "node":
                    col = self._col_node(name)
                elif typ == "iL":
                    col = self._col_iL(name)
                elif typ == "iV":
                    col = self._col_iV(name)
                else:
                    raise ValueError(f"Unknown output coefficient type '{typ}'")
                if col is not None:
                    vec[row, col] += float(val)

        vec_d = vec[:, self.diff_cols]
        vec_a = vec[:, self.alg_cols]
        C = vec_d + self._safe_matmul(vec_a, self.Phi)
        D = self._safe_matmul(vec_a, self.Psi)
        return C, D

    def _build_output_row(
        self,
        coefficients: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project one raw MNA expression to one state-space output row."""
        C, D = self._build_output_rows([coefficients])
        return C[0], D[0]

    def _state_derivative_row(
        self,
        node: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return rows representing ``dV(node)/dt = C*x + D*u``."""
        n_states = len(self.state_labels)
        if node in GND:
            return np.zeros(n_states), np.zeros(self.n_u)
        label = f"v_{node}"
        if label not in self.state_labels:
            raise ValueError(
                f"node '{node}' has no capacitor attached, so its voltage isn't "
                f"a state and dv/dt isn't defined this way. States: {self.state_labels}"
            )
        idx = self.state_label_idx[label]
        return self.A[idx, :], self.B_ss[idx, :]

    def add_node_voltage_output(self, node_name: str) -> None:
        """
        Add a node-to-ground voltage to the selected system outputs.

        Parameters
        ----------
        node_name:
            Netlist node whose voltage is measured relative to ground.

        Raises
        ------
        ValueError
            If ``node_name`` is not present in the circuit.
        """
        node = _normalize_node(node_name)
        if node not in GND and node not in self.node_idx:
            raise ValueError(f"Unknown node '{node_name}'")

        coefficients = {} if node in GND else {f"node:{node}": 1.0}
        self._selected_output_rows.append(self._build_output_row(coefficients))
        self.output_labels.append(f"V({node})")

    def add_dipole_current_output(self, dipole_name: str) -> None:
        """
        Add the current through a two-terminal netlist element.

        Parameters
        ----------
        dipole_name:
            Name of a resistor, inductor, capacitor, voltage source, or current
            source. Positive current follows the element declaration from
            ``n1`` to ``n2``.

        Raises
        ------
        ValueError
            If no supported dipole has the requested name.

        Notes
        -----
        Resistor current is derived from Ohm's law. Inductor and voltage-source
        currents are MNA branch unknowns. Capacitor current is computed from its
        voltage derivative. Current-source current is its corresponding input
        signal directly.
        """
        try:
            element = self.dipoles_by_name[dipole_name]
        except KeyError as exc:
            raise ValueError(f"Unknown dipole '{dipole_name}'") from exc

        if element.kind == "R":
            coefficients = {}
            if element.n1 not in GND:
                coefficients[f"node:{element.n1}"] = 1.0 / element.value
            if element.n2 not in GND:
                key = f"node:{element.n2}"
                coefficients[key] = coefficients.get(key, 0.0) - 1.0 / element.value
            row = self._build_output_row(coefficients)
        elif element.kind == "L":
            row = self._build_output_row({f"iL:{element.name}": 1.0})
        elif element.kind == "C":
            A_n1, B_n1 = self._state_derivative_row(element.n1)
            A_n2, B_n2 = self._state_derivative_row(element.n2)
            row = (
                element.value * (A_n1 - A_n2),
                element.value * (B_n1 - B_n2),
            )
        elif element.kind == "V":
            row = self._build_output_row({f"iV:{element.name}": 1.0})
        elif element.kind == "I":
            C_row = np.zeros(len(self.state_labels), dtype=float)
            D_row = np.zeros(self.n_u, dtype=float)
            D_row[self.input_names.index(element.name)] = 1.0
            row = C_row, D_row
        else:
            raise ValueError(
                f"Element '{dipole_name}' of type '{element.kind}' is not a dipole"
            )

        self._selected_output_rows.append(row)
        self.output_labels.append(f"I({element.name})")

    def get_system(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the reduced system with all selected outputs.

        Returns
        -------
        A, B, C, D:
            Matrices satisfying ``dx/dt = A*x + B*u`` and
            ``y = C*x + D*u``. Rows of ``C`` and ``D`` follow the order in
            which outputs were added. With no selected outputs, ``C`` and ``D``
            have zero rows and retain the correct state/input column counts.
        """
        if not self._selected_output_rows:
            return (
                self.A,
                self.B_ss,
                np.empty((0, len(self.state_labels)), dtype=float),
                np.empty((0, self.n_u), dtype=float),
            )
        C = np.vstack([row[0] for row in self._selected_output_rows])
        D = np.vstack([row[1] for row in self._selected_output_rows])
        return self.A, self.B_ss, C, D


class NetlistStateSpace(StateSpace):
    """
    PathSim state-space block constructed directly from a linear netlist.

    Parameters
    ----------
    netlist:
        Existing netlist path or inline netlist text. A :class:`str` is treated
        as a path when it names an existing file and as netlist text otherwise.
        A :class:`Path` is always treated as an explicit file path.
    output_voltages:
        Node names exposed as node-to-ground voltage outputs.
    output_currents:
        Names of R, L, C, voltage-source, or current-source dipoles exposed as
        current outputs. Positive current follows each netlist ``n1 -> n2``
        declaration.
    initial_value:
        Initial differential state. Defaults to zero for every state.

    Notes
    -----
    Output ports are ordered with all requested voltages first, followed by all
    requested currents. Their labels are ``V(node)`` and ``I(dipole)``.
    Input ports retain the voltage-source-then-current-source ordering of the
    netlist model. The underlying :class:`CircuitModel` is available as
    :attr:`model`.

    Examples
    --------
    >>> block = NetlistStateSpace(
    ...     "filter.net",
    ...     output_voltages=["n1"],
    ...     output_currents=["Rload"],
    ... )
    """

    def __init__(
        self,
        netlist: str | Path,
        output_voltages: list[str] | None = None,
        output_currents: list[str] | None = None,
        initial_value: np.ndarray | None = None,
    ):
        if isinstance(netlist, Path):
            elements = parse_netlist_file(netlist)
        elif isinstance(netlist, str):
            candidate = Path(netlist)
            try:
                is_file = candidate.is_file()
            except OSError:
                is_file = False
            elements = parse_netlist_file(candidate) if is_file else parse_netlist(netlist)
        else:
            raise TypeError("netlist must be a string or pathlib.Path")

        self.model = CircuitModel(elements)
        for node_name in output_voltages or []:
            self.model.add_node_voltage_output(node_name)
        for dipole_name in output_currents or []:
            self.model.add_dipole_current_output(dipole_name)

        A, B, C, D = self.model.get_system()
        super().__init__(
            A=A,
            B=B,
            C=C,
            D=D,
            initial_value=initial_value,
            state_labels=list(self.model.state_labels),
            input_labels=list(self.model.input_names),
            output_labels=list(self.model.output_labels),
        )
