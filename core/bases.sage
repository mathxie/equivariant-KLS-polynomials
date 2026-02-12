"""
bases.sage
----------

Purpose
-------
Shared helpers for the equivariant KL / inverse-KL / μ / characteristic / Z stacks.

This file collects reusable group/poset utilities so that the feature modules
(``ekl.sage``, ``equivariant_characteristic.sage``, ``ez.sage``) stay focused.

Public helpers (selected)
-------------------------
- ``_ensure_gap_backend()`` (libgap fallback when PTYs are unavailable)
- ``now()``
- ``induced_action_on_flats_group(M, ground_group=None)``
- ``cover_edges(G)``, ``digraph_to_poset(G)``
- ``inflate_classfunction(cf_H, hom_G_to_H, G=None, H=None)``
- ``zero_classfunction(group)``, ``classfunction_pointwise_product(cf1, cf2)``
- ``coerce_classfunction_to_group(cf, target_group)``
- ``stable_orbit_rep_key(x)``
- ``restrict_group_to_subset(G, subset)``
- ``shrink_group_to_poset(W, P)``, ``canonicalize_action(P, W)``, ``canonical_setup(P, W)``
- ``peel_cover_of_minimals_k(P, k, use_covers=False)``

Example::

    sage -c "load('core/bases.sage'); load('tests/helpers_examples.sage')"

SageMath examples
-----------------

EXAMPLES::

    sage: load('core/bases.sage')
    sage: P = Poset(([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)]))
    sage: W = P.hasse_diagram().automorphism_group()
    sage: setup = canonical_setup(P, W)
    sage: (setup["P_can"].cardinality(), setup["W_can"].order())
    (4, 2)
"""

import logging
import warnings
import time

# Some Sage installs configure dot2tex logging at INFO; silence it for clean output.
logging.getLogger("dot2tex").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class QuietDict(dict):
    """
    Dict with a compact ``repr`` for interactive sessions.

    Sage (like Python) echoes the return value of an expression, which can be
    extremely noisy when a helper returns a large nested dict containing groups,
    class functions, etc.  This wrapper keeps the mapping behavior of a dict,
    but renders as a short one-line summary.
    """

    def __init__(self, *args, summary=None, max_keys=8, **kwargs):
        super().__init__(*args, **kwargs)
        self._quiet_summary = summary
        self._quiet_max_keys = max_keys

    def __repr__(self):
        summary = self._quiet_summary or self.__class__.__name__
        try:
            keys = sorted(self.keys())
        except TypeError:
            keys = list(self.keys())
        preview = keys[: self._quiet_max_keys]
        more = "" if len(keys) <= len(preview) else ", ..."
        key_str = ", ".join(map(str, preview)) + more
        return f"<{summary} keys=[{key_str}]>"

    def as_dict(self):
        """Return a plain dict copy (useful when you want the full repr)."""

        return dict(self)


def _ensure_gap_backend(*, verbose=False):
    """
    Configure the global ``gap`` interface so it works when PTYs are unavailable.

    Some environments run out of PTYs (``OSError('out of pty devices')``), which
    breaks Sage's default pexpect-based GAP interface.  When that happens we
    transparently replace the global ``gap`` object by a small wrapper around
    ``libgap`` implementing the subset of the API used in this repo:

    - ``gap(expr)``  -> GAP object
    - ``gap.eval(expr)`` -> string (compatible with ``sage_eval``)

    Parameters
    ----------
    verbose : bool (default: False)
        If True, print a one-line message when switching to libgap.

    Returns
    -------
    gap_backend or None
        - Returns the (possibly replaced) global ``gap`` object.
        - Returns None if ``gap`` is not available in this Sage session.

    Notes
    -----
    - This function is idempotent: it only configures the backend once per process.
    - When switching to libgap, the replacement object has attribute ``_is_libgap = True``.
    """

    if globals().get("_GAP_BACKEND_CONFIGURED", False):
        return globals().get("gap")
    globals()["_GAP_BACKEND_CONFIGURED"] = True

    if "gap" not in globals():
        return None

    try:
        import os
        import pty

        master, slave = pty.openpty()
        os.close(master)
        os.close(slave)
        return globals().get("gap")
    except Exception:
        pass

    try:
        from sage.libs.gap.libgap import libgap as _libgap
    except Exception as exc:
        raise RuntimeError(
            "GAP backend requires PTYs, but PTYs are unavailable and libgap could not be imported."
        ) from exc

    class _LibGapCompat:
        _is_libgap = True

        def __call__(self, expr):
            return _libgap.eval(expr)

        def eval(self, expr):
            text = str(expr).strip()
            if not text:
                return ""

            parts = [p.strip() for p in text.split(";;") if p.strip()]
            val = None
            for part in parts:
                while part.endswith(";"):
                    part = part[:-1].rstrip()
                if part:
                    val = _libgap.eval(part)
            if val is None:
                return ""
            return str(val)

    globals()["gap"] = _LibGapCompat()
    if verbose:
        print("[gap] using libgap backend (pty unavailable)")
    return globals()["gap"]


_ensure_gap_backend(verbose=False)


def now():
    """Return a timestamp string for lightweight log messages."""

    return time.strftime("%H:%M:%S %Y-%m-%d ", time.localtime())


def require_base_functions(names, context):
    """
    Raise a clear error if any of the requested helper names are missing.
    Useful for dependency checks in scripts that assume ``bases.sage`` is
    loaded.
    """

    missing = [name for name in names if name not in globals()]
    if missing:
        raise RuntimeError(
            "Missing helpers: "
            + ", ".join(missing)
            + f"; load bases.sage before {context}."
        )


def induced_action_on_flats_group(M, ground_group=None, flats=None):
    """
    Given a matroid ``M`` and optionally its ground-set automorphism group
    ``ground_group``, induce the action on the lattice of flats.

    Note: the lattice-of-flats iterator can depend on Python hash iteration
    order; we sort flats deterministically to keep the induced permutation
    representation reproducible across processes.

    Example::

        M = Matroid(graph=graphs.PathGraph(3))
        induced_action_on_flats_group(M)

    INPUT:

    - ``M`` -- a matroid.
    - ``ground_group`` -- (default: ``None``) a permutation group acting on the ground set.
      If ``None``, use ``M.automorphism_group()``.
    - ``flats`` -- (default: ``None``) an explicit list/iterator of flats. When provided,
      we use it (after coercing to a list) instead of recomputing ``M.lattice_of_flats()``.

    OUTPUT:

    A permutation group acting on the (sorted) list of flats, with domain equal to that list.

    EXAMPLES::

        sage: load('core/bases.sage')
        sage: M = matroids.Uniform(2, 4)
        sage: W = induced_action_on_flats_group(M)
        sage: W.order()  # induced from Aut(M) on flats
        24
    """

    if ground_group is None:
        ground_group = M.automorphism_group()

    if flats is None:
        flats = list(M.lattice_of_flats())
    else:
        flats = list(flats)
    flats = sorted(flats, key=lambda F: (len(F), tuple(sorted(repr(e) for e in F))))
    flats_set = set(flats)
    gens = []
    for sigma in ground_group.gens():
        perm = []
        for F in flats:
            Fimg = frozenset(sigma(e) for e in F)
            assert Fimg in flats_set, f"flat image check failed: {F} → {Fimg}"
            perm.append(Fimg)
        gens.append(perm)

    return PermutationGroup(gens=gens, domain=flats)


def cover_edges(G):
    """Return the cover edges of a DAG, or raise if a directed cycle exists."""

    if not G.is_directed_acyclic():
        raise ValueError("Input digraph has a directed cycle; cannot convert to a poset.")
    H = G.transitive_reduction()
    return [(u, v) for (u, v) in H.edges(labels=False)]


def digraph_to_poset(G):
    """Convert a DAG into a ``Poset`` with the same vertex set."""

    return Poset((G.vertices(), cover_edges(G)))


def inflate_classfunction(cf_H, hom_G_to_H, G=None, H=None):
    """
    Inflate a class function ``cf_H`` along a surjective homomorphism
    ``phi : G → H``.

    Example::

        P = Poset(([0, 1, 2], [(0, 1), (1, 2)]))
        G = PermutationGroup([[0, 2, 1]], domain=[0, 1, 2])
        W_small, pi = shrink_group_to_poset(G, P)
        cf_small = ClassFunction(W_small, [1] * len(W_small.conjugacy_classes()))
        cf_big = inflate_classfunction(cf_small, pi, G, W_small)
        [cf_big(g) for g in G.conjugacy_classes_representatives()]  # -> [1, 1]

    INPUT:

    - ``cf_H`` -- a class function on ``H``.
    - ``hom_G_to_H`` -- a (typically surjective) group homomorphism ``G -> H``.
    - ``G`` -- (optional) the domain group. If omitted, inferred from the homomorphism.
    - ``H`` -- (optional) the codomain group. If omitted, inferred from the homomorphism.

    OUTPUT:

    A class function on ``G`` given by ``cf_H(hom_G_to_H(g))``.

    EXAMPLES::

        sage: load('core/bases.sage')
        sage: P = Poset(([0, 1, 2], [(0, 1), (1, 2)]))
        sage: G = PermutationGroup([[0, 2, 1]], domain=[0, 1, 2])
        sage: W_small, pi = shrink_group_to_poset(G, P)
        sage: cf_small = ClassFunction(W_small, [1] * len(W_small.conjugacy_classes()))
        sage: cf_big = inflate_classfunction(cf_small, pi, G, W_small)
        sage: [cf_big(g) for g in G.conjugacy_classes_representatives()]
        [1, 1]
    """

    if G is None:
        try:
            G = hom_G_to_H.domain()
        except AttributeError as exc:
            raise ValueError("Please provide the domain group G explicitly.") from exc
    if H is None:
        try:
            H = hom_G_to_H.codomain()
        except AttributeError as exc:
            raise ValueError("Please provide the codomain group H explicitly.") from exc

    assert H == cf_H.domain(), "chi_H must be defined on the codomain H of phi."

    vals = []
    for conj_class in G.conjugacy_classes():
        g = conj_class.representative()
        h = hom_G_to_H(g)
        assert h in H, f"Image phi(g)={h} is not an element of H."
        vals.append(cf_H(h))

    return ClassFunction(G, vals)


def zero_classfunction(group):
    """Return the zero class function on ``group``."""

    return ClassFunction(group, [0] * len(group.conjugacy_classes()))


def classfunction_pointwise_product(cf1, cf2):
    """
    Multiply class functions by evaluation on conjugacy class representatives.

    We avoid using ``cf1 * cf2`` directly because Sage/GAP may attach different
    character-table objects to class functions even when the underlying group
    is the same, which can trigger “no product of class functions of different
    tables”.
    """

    group = cf1.domain()
    values = []
    for conj_class in group.conjugacy_classes():
        g = conj_class.representative()
        values.append(cf1(g) * cf2(g))
    return ClassFunction(group, values)


def coerce_classfunction_to_group(cf, target_group):
    """
    Rebuild a class function on ``target_group`` by evaluation.

    This avoids subtle parent/coercion issues when Sage returns a stabilizer
    subgroup object that is equal (same underlying permutations, same order)
    but not identical to the ambient group.
    """

    source_group = cf.domain()
    values = []
    for conj_class in target_group.conjugacy_classes():
        g = conj_class.representative()
        if source_group is target_group or source_group == target_group:
            values.append(cf(g))
            continue
        domain = list(source_group.domain())
        g_src = source_group([g(x) for x in domain])
        values.append(cf(g_src))
    return ClassFunction(target_group, values)


def stable_orbit_rep_key(x):
    """Deterministic key for choosing orbit representatives."""

    if isinstance(x, (set, frozenset)):
        return (len(x), tuple(sorted(repr(e) for e in x)))
    return (repr(x),)


def restrict_group_to_subset(G, subset):
    """
    Restrict a permutation group ``G`` to the given subset of its domain.

    Example::

        G = PermutationGroup([[0, 1, 3, 2]], domain=[0, 1, 2, 3])
        G_small = restrict_group_to_subset(G, [0, 1, 2])
        [list(g) for g in G_small.gens()]  # -> [[0, 1, 2]]
    """

    imgs = [[g(x) for x in subset] for g in G.gens()]
    return PermutationGroup(imgs, domain=subset)


def shrink_group_to_poset(W, P):
    """
    Restrict a group action ``W ↷ P`` to the actual elements of the poset.

    Returns ``(W_small, pi)`` where ``W_small`` acts on ``P.list()`` and
    ``pi : W → W_small`` is the induced surjection.

    Example::

        P = Poset(([0, 1, 2], [(0, 1), (1, 2)]))
        G = PermutationGroup([[0, 2, 1]], domain=[0, 1, 2])
        W_small, pi = shrink_group_to_poset(G, P)
    """

    S = list(P)
    bad = []
    for g in W.gens():
        image = [g(x) for x in S]
        if any(y not in S for y in image):
            bad.append((g, image))
    if bad:
        msg = "Some generators move points outside P.list():\n"
        for g, img in bad:
            msg += f"  * g = {g},  g(S) = {img}\n"
        raise ValueError(msg)

    assert {g(x) for x in S for g in W.gens()} == set(P)

    gens_on_S = [[g(x) for x in S] for g in W.gens()]
    W_small = PermutationGroup(gens_on_S, domain=S)
    images = [W_small(g_img) for g_img in gens_on_S]

    if W.order() == 1:
        # `PermutationGroup([])` can be incompatible across different domains;
        # force conversion to a generic model.
        W = PermutationGroup_generic([], domain=W.domain())
        W_small = PermutationGroup_generic([], domain=S)
        gens = list(W.gens())
        imgs = [W_small.identity()] * len(gens)
        pi = W.hom(imgs, W_small)
        return W_small, pi

    pi = W.hom(images, W_small)
    return W_small, pi


def canonicalize_action(poset, group):
    """
    Canonicalise the pair ``(poset, group)`` for caching.

    Returns ``(P_can, W_can, relabel, phi)`` where ``phi`` is the homomorphism
    from ``group`` to ``W_can`` induced by the canonical relabelling.

    Example::

        P = Poset(([0, 1, 2], [(0, 1), (1, 2)]))
        G_big = PermutationGroup([[0, 2, 1, 3]], domain=[0, 1, 2, 3])  # moves a point outside P
        W_small, _ = shrink_group_to_poset(G_big, P)  # drops the extra point
        P_can, W_can, relabel, phi = canonicalize_action(P, W_small)
        list(P_can)  # -> [0, 1, 2]

    INPUT:

    - ``poset`` -- a finite poset.
    - ``group`` -- a permutation group acting on the poset elements (by poset automorphisms).

    OUTPUT:

    A 4-tuple ``(P_can, W_can, relabel, phi)`` where:

    - ``P_can`` is a canonically relabelled copy of ``poset`` (with vertex set
      ``[0, 1, ..., n-1]`` in a deterministic order);
    - ``W_can`` is the relabelled permutation group acting on ``P_can``;
    - ``relabel`` is the relabelling certificate from Sage's canonical label;
    - ``phi`` is the group homomorphism ``group -> W_can`` induced by relabelling.

    EXAMPLES::

        sage: load('core/bases.sage')
        sage: P = Poset(([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)]))
        sage: W = P.hasse_diagram().automorphism_group()
        sage: P_can, W_can, relabel, phi = canonicalize_action(P, W)
        sage: (P_can.cardinality(), W_can.order())
        (4, 2)
    """

    dg = poset.hasse_diagram()
    dg_can, relabel = dg.canonical_label(algorithm="sage", certificate=True)

    # Make the canonical permutation representation deterministic.
    #
    # Iteration order of `poset` can vary (depending on internal dict/set
    # insertion order), so constructing the canonical group action by iterating
    # over `poset` directly may lead to different `W_can` domains across runs.
    #
    # Instead we use the relabeling certificate to identify the vertex
    # corresponding to each canonical label i, and build permutations on the
    # sorted canonical domain [0,1,...,n-1].
    new_dom = sorted(relabel.values())
    inv = {relabel[v]: v for v in poset}

    gens = list(group.gens())
    new_gens = []
    for g in gens:
        perm = []
        for i in new_dom:
            v = inv[i]
            perm.append(relabel[g(v)])
        new_gens.append(perm)

    W_can = PermutationGroup(new_gens, domain=new_dom)
    images = [W_can(g_img) for g_img in new_gens]
    phi = group.hom(images, W_can)
    return digraph_to_poset(dg_can), W_can, relabel, phi


def canonical_setup(P, W):
    """
    Convenience wrapper that performs shrink → canonicalise in one call and
    returns all relevant maps.

    Parameters
    ----------
    P : Poset (or matroid with ``lattice_of_flats``)
    W : permutation group acting on P

    Returns
    -------
    dict with keys:
      - P_can, W_can, relabel, phi  (from canonicalize_action)
      - W_small, pi                 (from shrink_group_to_poset)
      - to_canonical                (composed homomorphism W → W_can)

    Example::

        P = Poset(([0, 1, 2], [(0, 1), (1, 2)]))
        G_big = PermutationGroup([[0, 2, 1, 3]], domain=[0, 1, 2, 3])
        setup = canonical_setup(P, G_big)
        cf_can = ClassFunction(setup["W_can"], [1] * len(setup["W_can"].conjugacy_classes()))
        cf_small = inflate_classfunction(cf_can, setup["phi"])
        cf_big = inflate_classfunction(cf_small, setup["pi"])
    """

    if hasattr(P, "lattice_of_flats"):
        P = P.lattice_of_flats()
    W_small, pi = shrink_group_to_poset(W, P)
    P_can, W_can, relabel, phi = canonicalize_action(P, W_small)

    def to_canonical(g):
        return phi(pi(g))

    return {
        "P_can": P_can,
        "W_can": W_can,
        "relabel": relabel,
        "phi": phi,
        "W_small": W_small,
        "pi": pi,
        "to_canonical": to_canonical,
    }


def peel_cover_of_minimals_k(P, k, *, use_covers=False):
    """
    Remove (at most) ``k`` layers of elements that cover current minimal elements.

    This helper is used by the “peeled” poset families (Boolean / uniform).

    Parameters
    ----------
    P : Poset
    k : int
        Number of peeling steps (k >= 0).
    use_covers : bool (default: False)
        - False: build the induced subposet via ``P.subposet(keep)`` (recommended).
        - True : rebuild using only the original cover relations restricted to ``keep``.

    Returns
    -------
    Poset
        The peeled subposet.

    Example::

        P = Poset(([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)]))
        P2 = peel_cover_of_minimals_k(P, 2)
        list(P2)  # -> [0, 3]
        # Also exercised in tests/helpers_examples.sage::test_peel_covers
    """

    if k < 0:
        raise ValueError("k must be non-negative")

    current_P = P
    for _ in range(k):
        mins = set(current_P.minimal_elements())
        if not mins:
            break

        covering = set().union(*[current_P.upper_covers(x) for x in mins])
        if not covering:
            break

        keep = [e for e in current_P if e not in covering]

        if not use_covers:
            current_P = current_P.subposet(keep)
        else:
            cov = [
                (x, y)
                for (x, y) in current_P.cover_relations()
                if x in keep and y in keep
            ]
            current_P = Poset((keep, cov))

    return current_P
