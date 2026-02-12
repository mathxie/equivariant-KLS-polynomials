"""
tn.sage
-------

Utility script for the thagomizer matroid `T_n` (the graphic matroid of
`K_{1,1,n}`), including equivariant Kazhdan-Lusztig computations under both
`S_n` and `B_n`, with partition/bipartition labeling.

Example::

    HOME=$PWD SAGE_CACHE_DIR=.sage_cache sage -c "load('families/tn.sage'); analyse_thagomizer(3, verbose=False)"

Dependency note
---------------
Loading this file also loads the required core modules:
`core/bases.sage`, `core/ekl.sage`, and `core/rep2part.sage`.

Notation
--------
- `W_V`: action on graph vertices.
- `W_E`: induced action on edges / matroid ground set.
- `chi : W_V -> W_E`: induced homomorphism; `pullback_classfunction(cf, chi)`
  pulls eKL coefficients from `W_E` back to `W_V` (and then to standard `S_n`).
"""

import logging
import warnings

# Some Sage installs configure dot2tex logging at INFO; silence it for clean output.
logging.getLogger("dot2tex").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import inspect
import os
import sys
from pathlib import Path


def _this_sage_file_path():
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and argv0.endswith(".sage") and os.path.exists(argv0):
        return os.path.abspath(argv0)
    for fr in inspect.stack():
        for key in ("fpath", "filename"):
            val = fr.frame.f_locals.get(key)
            if val is None:
                continue
            try:
                val = os.fspath(val)
            except TypeError:
                continue
            if isinstance(val, str) and val.endswith(".sage") and os.path.exists(val):
                return os.path.abspath(val)
    return None


# Allow `sage -c "load('.../ekl_new/families/this_file.sage')"` from any CWD.
_THIS_SAGE_FILE = _this_sage_file_path()
if _THIS_SAGE_FILE:
    ekl_root = Path(_THIS_SAGE_FILE).resolve().parent.parent
    from sage.repl.attach import load_attach_path

    paths = load_attach_path()
    try:
        existing = {Path(p).resolve() for p in paths}
    except Exception:
        existing = set()
    if ekl_root not in existing:
        paths.insert(0, ekl_root)

load("core/ekl.sage")
load("core/rep2part.sage")


def build_thagomizer_matroid(n):
    """
    Build the thagomizer graphic matroid of `K_{1,1,n}` and relabel the
    ground set as `{0, ±1, ..., ±n}` for compatibility with `B_n`.
    """

    g = graphs.CompleteMultipartiteGraph([1, 1, n])
    base = Matroid(g)
    m = Matroid(base.bases())

    mapping = {(0, 1): 0}
    mapping.update({(0, j): j - 1 for j in range(2, n + 2)})
    mapping.update({(1, j): -(j - 1) for j in range(2, n + 2)})

    return m.relabel(mapping)


def _groundset_order(n):
    """
    Fixed ground-set order used by `PermutationGroup` constructors.
    """

    positives = [0] + list(range(1, n + 1))
    negatives = [-i for i in range(1, n + 1)]
    return positives + negatives


def _apply_sigma_to_point(point, sigma):
    """
    Action of `sigma in SymmetricGroup(n)` on the thagomizer ground set.
    """

    if point == 0:
        return 0
    if point > 0:
        return int(sigma(point))
    return -int(sigma(-point))


def build_symmetric_subgroup(m, n, *, return_hom=False):
    """
    Build the subgroup induced by `SymmetricGroup(n)` on the thagomizer ground set.

    Returns
    -------
    G_sn : PermutationGroup
        Group acting on the ground set, isomorphic to `S_n`.
    Sn : SymmetricGroup(n)
        Standard symmetric group.
    to_sn : callable
        Convert an element of `G_sn` to an element of `SymmetricGroup(n)`.

    Optional
    --------
    If `return_hom=True`, also return `psi : S_n -> G_sn`.
    """

    Sn = SymmetricGroup(n)
    domain = _groundset_order(n)
    images = [
        [_apply_sigma_to_point(point, sigma) for point in domain]
        for sigma in Sn.gens()
    ]
    G_sn = PermutationGroup(images, domain=domain)

    def to_sn(element):
        """Convert a ground-action element to `SymmetricGroup(n)`."""

        image = [int(element(i)) for i in range(1, n + 1)]
        return Sn(image)

    # Build ψ : S_n → G_sn for pullback workflows.
    images_as_elements = [G_sn(img) for img in images]
    psi = Sn.hom(images_as_elements, G_sn)

    if return_hom:
        return G_sn, Sn, to_sn, psi
    return G_sn, Sn, to_sn


def summarize_decomposition(decomposition, n, class_keys, labels, char_vectors, G, signed_group):
    """
    Convert `ClassFunction.decompose()` output into a readable structured form.
    """

    result = []
    for multiplicity, character in decomposition:
        class_values = {}
        for gen in G.conjugacy_classes_representatives():
            key = signed_cycle_type(gen, n, signed_group)
            class_values[key] = character(gen)
        label_key = identify_irreps_from_values(class_values, char_vectors, class_keys)
        label = labels.get(label_key, label_key)
        result.append(
            {
                "multiplicity": multiplicity,
                "degree": character.degree(),
                "values": class_values,
                "label": label,
            }
        )
    return result


def summarize_classfunction_sn(cf, class_data, characters, Sn, G, to_sn):
    """Decompose a class function for an `S_n` action into irreducibles."""

    group_order = Sn.cardinality()
    class_keys = [entry["key"] for entry in class_data]

    values = {}
    for gen in G.conjugacy_classes_representatives():
        key = tuple(to_sn(gen).cycle_type())
        values[key] = cf(gen)
    cf_vector = [complex(values.get(key, 0)) for key in class_keys]

    result = []
    for label, data in characters.items():
        irrep_vals = data["values"]
        inner = sum(
            class_entry["size"] * cf_val * ir_val.conjugate()
            for class_entry, cf_val, ir_val in zip(class_data, cf_vector, irrep_vals)
        ) / group_order
        multiplicity = int(round(inner.real))
        if multiplicity:
            result.append(
                {
                    "multiplicity": multiplicity,
                    "degree": data["degree"],
                    "values": dict(zip(class_keys, cf_vector)),
                    "label": data["partition"],
                }
            )

    return result


def analyse_thagomizer(n, *, verbose=False, return_data=False, compute_inverse=False, print_schur=False):
    """
    Compute eKL (and optionally inverse eKL) for `T_n` under `B_n`.

    By default, output is partition/bipartition labeled decomposition.
    Set `print_schur=True` to also print Schur expansions.
    """

    n = ZZ(n)
    if n < 0:
        raise ValueError("n must be >= 0")

    m = build_thagomizer_matroid(n)
    Bn = standard_bn_permutation_group(n)
    bn_data = bn_character_data(n)

    coeffs = ekl(m, Bn)
    decomposed = [
        decompose_representation(cf, group_kind="bn", n=n, bn_data=bn_data)
        for cf in coeffs
    ]

    inv_coeffs = None
    inv_decomposed = None
    if compute_inverse:
        inv_coeffs = equivariant_inverse_kl(m, Bn)
        inv_decomposed = [
            decompose_representation(cf, group_kind="bn", n=n, bn_data=bn_data)
            for cf in inv_coeffs
        ]

    data = {
        "n": n,
        "matroid": m,
        "group": Bn,
        "ekl_coeffs": coeffs,
        "decomposition": decomposed,
        "ikl_coeffs": inv_coeffs,
        "inverse_decomposition": inv_decomposed,
    }

    if verbose:
        _print_analysis_bn(data, print_schur=print_schur)

    if return_data:
        return QuietDict(data, summary=f"analyse_thagomizer(B_{n})")
    return None


def _print_analysis_bn(data, *, print_schur=False):
    m = data["matroid"]
    n = data["n"]
    print(f"T_{n} under B_{n}")
    print(m.lattice_of_flats().kazhdan_lusztig_polynomial())
    for i, res in enumerate(data["decomposition"]):
        print(f"t^{i}: {res['formatted']}")
        if print_schur:
            print(f"  schur: {res['schur']}")
    if data.get("inverse_decomposition") is not None:
        print("-- inverse --")
        for i, res in enumerate(data["inverse_decomposition"]):
            print(f"Q t^{i}: {res['formatted']}")
            if print_schur:
                print(f"  schur: {res['schur']}")


def _sn_vertex_action_hom_on_k11n(graph, n):
    """
    Build the natural vertex action of the *standard* S_n on K_{1,1,n}.

    Returns
    -------
    Sn, W_V, psi
        - Sn: SymmetricGroup(n), acting on {1..n}
        - W_V: permutation group on the vertex set
        - psi: homomorphism Sn -> W_V realizing the action on vertices
    """

    vertices = sorted(graph.vertices())
    if len(vertices) != n + 2:
        raise ValueError("graph must have n+2 vertices for K_{1,1,n}.")

    # The last n vertices form the size-n part; the first two vertices are fixed.
    block = vertices[2:]
    block_index = {v: i for i, v in enumerate(block)}
    Sn = SymmetricGroup(n)  # standard action on {1..n}

    gens_images = []
    for sigma in Sn.gens():
        image = []
        for v in vertices:
            if v in block_index:
                image.append(block[int(sigma(block_index[v] + 1)) - 1])
            else:
                image.append(v)
        gens_images.append(image)

    W_V = PermutationGroup(gens=gens_images, domain=vertices)
    images_as_elements = [W_V(img) for img in gens_images]
    psi = Sn.hom(images_as_elements, W_V)
    return Sn, W_V, psi


def analyse_thagomizer_bySn(
    n, *, verbose=False, return_data=False, compute_inverse=False, print_schur=False
):
    """
    Compute eKL using the standard `S_n` action on the size-`n` vertex block
    of `K_{1,1,n}`.

    Workflow:
      `S_n --psi--> W_V --chi--> W_E`
    Compute on `W_E`, then pull back along `phi = chi * psi` and decompose on
    standard `S_n`.
    """

    n = ZZ(n)
    if n < 0:
        raise ValueError("n must be >= 0")

    graph = graphs.CompleteMultipartiteGraph([1, 1, n])
    Sn, W_V, psi = _sn_vertex_action_hom_on_k11n(graph, n)
    m, W_E, chi = induced_action_on_graph_matroid(graph, W_V, return_hom=True)
    phi = chi * psi  # Sn -> W_E (edge/matroid action)

    coeffs_E = ekl(m, W_E)
    coeffs = [pullback_classfunction(cf, phi) for cf in coeffs_E]

    inv_coeffs = None
    inv = None
    if compute_inverse:
        inv_coeffs_E = equivariant_inverse_kl(m, W_E)
        inv_coeffs = [pullback_classfunction(cf, phi) for cf in inv_coeffs_E]

    detailed = []
    sn_data = sn_character_data(n)
    for idx, cf in enumerate(coeffs):
        res = decompose_representation(cf, group_kind="sn", n=n, sn_data=sn_data)
        detailed.append(
            {
                "degree": idx,
                "inflated": [
                    {"label": part, "multiplicity": mult}
                    for part, mult in res["decomposition"]
                ],
                "result": res,
            }
        )

    data = {
        "n": n,
        "matroid": m,
        "graph": graph,
        "symmetric_group": Sn,
        "vertex_group": W_V,
        "edge_group": W_E,
        "chi": chi,
        "phi": phi,
        "ekl_coeffs": coeffs,
        "ikl_coeffs": inv_coeffs,
        "decomposition": detailed,
    }

    if verbose:
        _print_analysis_sn(data, print_schur=print_schur)

    if return_data:
        return QuietDict(data, summary=f"analyse_thagomizer_bySn(S_{n})")

    return None


def _print_analysis_sn(data, *, print_schur=False):
    """Print analysis output in the compact `S_n` format."""

    n = data["n"]
    print(f"T_{n} under S_{n}")
    print(data["matroid"].lattice_of_flats().kazhdan_lusztig_polynomial())
    for entry in data["decomposition"]:
        deg = entry["degree"]
        res = entry.get("result")
        if res is None:
            continue
        print(f"t^{deg} dim={res['dimension']} :: {res['formatted']}")
        if print_schur:
            print(f"  schur: {res['schur']}")
    if data.get("ikl_coeffs") is not None:
        print("-- inverse --")
        sn_data = sn_character_data(n)
        for i, cf in enumerate(data["ikl_coeffs"]):
            res = decompose_representation(cf, group_kind="sn", n=n, sn_data=sn_data)
            print(f"Q t^{i} dim={res['dimension']} :: {res['formatted']}")
            if print_schur:
                print(f"  schur: {res['schur']}")
