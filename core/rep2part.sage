"""
rep2part.sage
-------------

Unified workflow for decomposing class functions/characters into irreducibles
of S_n and B_n, labeled by partitions/bipartitions.

This file merges:
- the old `core/irr2partition.sage` (character data + order-agnostic labeling),
- the old `core/rep_decompose.sage` (workflow wrappers: inference, pullback,
  induced action on graphic matroids).

Backwards compatibility: `core/irr2partition.sage` and `core/rep_decompose.sage`
were merged here; the separate wrapper entry points are removed.

Provided utilities:

- ``sn_character_data(n)``: GAP-backed class/character data for S_n
  (cycle-type partitions, sizes, irreducible character values).
- ``sn_decompose(cf, cycle_type_fn=None, data=None)``: decompose a class
  function ``cf`` on a group isomorphic to S_n into ``(partition,
  multiplicity)`` using explicit inner products. By default, the cycle
  type is read from the permutation structure of the domain elements.
- ``bn_character_data(n)``: bipartition class/character lookup for B_n
  (via ``wreath_symmetric_character``).
- ``wreath_symmetric_character(n, classpara, charpara)``: character values in
  the wreath-symmetric group ``C2 ≀ S_n`` (used for B_n tables).
- ``bn_decompose(cf, signed_cycle_type_fn, data=None)``: label
  multiplicities for B_n using ``cf.decompose()`` and bipartition
  matching (no reliance on irrep ordering).
- ``format_sn_decomposition`` / ``format_bn_decomposition``: compact
  string renderings.
- ``decompose_representation(cf)``: infer S_n/B_n and print labeled decomposition.
- ``pullback_classfunction``: pull back along a homomorphism.
- ``induced_action_on_graph_matroid``: vertex action -> edge/matroid action (+ χ).

Example (S_3 regular character should decompose to (3):1, (2,1):2,
(1,1,1):1)::

    HOME=$PWD SAGE_CACHE_DIR=.sage_cache sage -c "
        load('core/rep2part.sage');
        G = SymmetricGroup(3);
        cf = ClassFunction(G, [6,0,0]);
        res = decompose_representation(cf);
        print(res['formatted']);
    "

SageMath examples
-----------------

EXAMPLES::

    sage: load('core/rep2part.sage')
    sage: G = SymmetricGroup(3)
    sage: cf = ClassFunction(G, [6, 0, 0])  # regular character
    sage: res = decompose_representation(cf, group_kind="sn", n=3)
    sage: res["decomposition"]
    [([1, 1, 1], 1), ([2, 1], 2), ([3], 1)]
    sage: res["schur"]
    s[1, 1, 1] + 2*s[2, 1] + s[3]

Dependencies: uses Sage's ``gap`` interface.
"""

import os
import inspect
import sys


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


_THIS_SAGE_FILE = _this_sage_file_path()
_CORE_DIR = os.path.dirname(_THIS_SAGE_FILE) if _THIS_SAGE_FILE else None
if _CORE_DIR:
    load(os.path.join(_CORE_DIR, "bases.sage"))
else:
    load("core/bases.sage")

def wreath_symmetric_character(n, classpara, charpara):
    """
    Character value χ_{charpara}(classpara) in the wreath-symmetric group C2 ≀ S_n.
    """

    gap.eval(f"n := {n};;")
    gap.eval(f"classpara := {classpara};;")
    gap.eval(f"charpara  := {charpara};;")
    gap.eval("betas := List(charpara, BetaSet);;")
    gap.eval('c2 := CharacterTable(\"Cyclic\", 2);;')
    gap.eval("wr := CharacterTableWreathSymmetric(c2, n);;")
    gap.eval("classpos := Position(ClassParameters(wr), classpara);;")
    gap.eval("charpos  := Position(CharacterParameters(wr), charpara);;")
    return int(gap.eval("Irr(wr)[charpos, classpos];"))


def _gap_is_libgap():
    """
    Return True if the global ``gap`` object looks like a libgap wrapper.
    """

    return bool(getattr(gap, "_is_libgap", False))


def sn_character_cache_path():
    """
    Return the default path of the bundled S_n character-table cache, or None.

    The repository ships `core/sn_character_cache.pkl` (a small pickle file).
    """

    import os

    if _CORE_DIR:
        path = os.path.join(_CORE_DIR, "sn_character_cache.pkl")
        if os.path.exists(path):
            return path

    path = os.path.join(os.getcwd(), "core", "sn_character_cache.pkl")
    if os.path.exists(path):
        return path

    return None


def load_sn_character_cache(path=None):
    """
    Load a cached S_n character table from a pickle file.

    Expected structure:
      cache[n][irrep_partition_tuple][class_partition_tuple] = integer value.

    If `path` is None, tries `sn_character_cache_path()`. Returns `{}` when no
    cache is available.
    """

    import os
    import pickle

    if path is None:
        path = sn_character_cache_path()
    if path is None or not os.path.exists(path):
        return {}

    with open(path, "rb") as f:
        cache = pickle.load(f)

    return {int(k): v for k, v in cache.items()}


def _sn_character_data_from_cache(n, n_cache):
    """
    Build `(Sn, class_data, characters)` in the same format as `sn_character_data`,
    using cached table data for fixed `n`.
    """

    Sn = SymmetricGroup(n)
    if not n_cache:
        raise ValueError("empty cache row for S_n")

    first_irrep = next(iter(n_cache))
    class_keys = sorted(n_cache[first_irrep].keys())

    class_data = []
    fact_n = Sn.cardinality()
    for key in class_keys:
        p = Partition(key)
        size = fact_n // p.centralizer_size()
        class_data.append({"key": tuple(key), "size": int(size)})

    identity_key = tuple([1] * n)
    characters = {}
    for irrep_key, row_dict in n_cache.items():
        irrep_key = tuple(irrep_key)
        values = [complex(row_dict.get(k, 0)) for k in class_keys]
        degree = int(row_dict.get(identity_key, 0))
        characters[irrep_key] = {
            "partition": Partitions(n)(irrep_key),
            "values": values,
            "degree": degree,
        }

    return Sn, class_data, characters


# --------------------------------------------------------------------------- #
#  Symmetric group S_n
# --------------------------------------------------------------------------- #
def sn_character_data(n):
    """
    GAP-backed class/character data for S_n.

    Returns
    -------
    Sn : SymmetricGroup
    class_data : list of dicts
        Each entry has keys ``key`` (cycle-type partition as tuple) and
        ``size`` (class size).
    characters : dict
        Mapping ``partition_tuple -> {partition, values, degree}``, where
        ``values`` are in the same order as ``class_data``.
    """

    n = ZZ(n)
    if n < 0:
        raise ValueError("n must be nonnegative.")
    if n == 0:
        Sn = SymmetricGroup(0)
        class_data = [{"key": (), "size": 1}]
        characters = {
            (): {"partition": Partitions(0)(()), "values": [complex(1)], "degree": 1}
        }
        return Sn, class_data, characters

    Sn = SymmetricGroup(n)

    cache = globals().get("SN_CHARACTER_CACHE")
    if cache is None:
        cache = load_sn_character_cache()
        if cache:
            globals()["SN_CHARACTER_CACHE"] = cache
    if isinstance(cache, dict) and n in cache:
        return _sn_character_data_from_cache(n, cache[n])

    if _gap_is_libgap():
        from sage.libs.gap.libgap import libgap

        tbl = libgap.CharacterTable("Symmetric", n)

        class_params = sage_eval(str(libgap.ClassParameters(tbl)))
        class_partitions = [entry[1] for entry in class_params]
        class_sizes = sage_eval(str(libgap.SizesConjugacyClasses(tbl)))

        char_params = sage_eval(str(libgap.CharacterParameters(tbl)))
        char_partitions = [entry[1] for entry in char_params]

        irr = libgap.Irr(tbl)
        nr_classes = int(libgap.NrConjugacyClasses(tbl))
        nr_chars = int(libgap.Length(irr))
        char_values = []
        for i in range(nr_chars):
            chi = irr[i]
            char_values.append([chi[j] for j in range(nr_classes)])
    else:
        tbl = gap('CharacterTable("Symmetric", %s)' % n)

        class_partitions = sage_eval(
            gap.eval('List(ClassParameters(%s), x -> x[2])' % tbl.name())
        )
        class_sizes = sage_eval(
            gap.eval('SizesConjugacyClasses(%s)' % tbl.name())
        )

        char_partitions = sage_eval(
            gap.eval('List(CharacterParameters(%s), x -> x[2])' % tbl.name())
        )
        char_values = sage_eval(
            gap.eval(
                'List(Irr(%s), chi -> List([1..NrConjugacyClasses(%s)], j -> chi[j]))'
                % (tbl.name(), tbl.name())
            )
        )

    class_data = [
        {"key": tuple(part), "size": size}
        for part, size in zip(class_partitions, class_sizes)
    ]

    characters = {}
    for part, values in zip(char_partitions, char_values):
        key = tuple(part)
        characters[key] = {
            "partition": Partitions(n)(part),
            "values": [complex(v) for v in values],
            "degree": int(values[0]),
        }

    return Sn, class_data, characters


def _cycle_type_partition(gen):
    """
    Extract cycle-type partition of a permutation group element.
    """

    if hasattr(gen, "cycle_type"):
        return tuple(sorted(gen.cycle_type(), reverse=True))

    cycles = getattr(gen, "cycle_tuples", lambda: gen.cycle_decomposition())()
    lengths = [len(c) for c in cycles]
    return tuple(sorted(lengths, reverse=True))


def _cycle_type_partition_on_subset(gen, subset):
    """
    Cycle-type partition of ``gen`` restricted to an invariant subset.

    This is used when a group is isomorphic to ``S_n`` but acts on a larger
    set with global fixed points; we ignore those fixed points and compute
    the cycle type on the moved subset.
    """

    subset = list(subset)
    subset_set = set(subset)
    seen = set()
    lengths = []

    for start in subset:
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            if current not in subset_set:
                raise ValueError("element does not preserve the chosen subset.")
            seen.add(current)
            length += 1
            current = gen(current)
        lengths.append(length)

    return tuple(sorted(lengths, reverse=True))


def _default_sn_cycle_type_fn(group, n):
    """
    Build a default ``cycle_type_fn`` for an S_n-like permutation action.

    If the permutation action is on exactly n points, we use the native
    cycle-type extractor. If the action is on more points but has exactly n
    non-globally-fixed points, we compute the cycle type on that moved subset.

    Raises a ValueError when the action is not compatible with S_n cycle types.
    """

    try:
        domain = list(group.domain())
    except Exception:
        return _cycle_type_partition

    if len(domain) == n:
        return _cycle_type_partition

    moved = set()
    for sigma in group.gens():
        for x in domain:
            if sigma(x) != x:
                moved.add(x)

    if not moved:
        if n <= len(domain):
            subset = domain[:n]

            def cycle_type_fn(gen):
                return _cycle_type_partition_on_subset(gen, subset)

            return cycle_type_fn
        raise ValueError("group domain too small for requested S_n parameter n.")

    if len(moved) != n:
        order = None
        try:
            order = ZZ(group.order())
        except Exception:
            pass
        expected = factorial(n)
        if order == expected:
            raise ValueError(
                "Group order is %s (=|S_%s|), but the permutation domain size is %s "
                "and the number of non-globally-fixed points is %s.\n"
                "Cannot read S_n cycle types directly from this action.\n"
                "Please provide a homomorphism phi: SymmetricGroup(%s) -> W "
                "(where W is the character domain and W ~= S_%s), then decompose "
                "after pullback to the standard S_n action, e.g. "
                "decompose_representation(..., pullback=phi)."
                % (expected, n, len(domain), len(moved), n, n)
            )

        raise ValueError(
            f"cannot infer S_{n} cycle types from a permutation action of degree {len(domain)}; "
            "use pullback to a standard S_n action."
        )

    try:
        subset = sorted(moved)
    except Exception:
        subset = sorted(moved, key=lambda e: repr(e))

    def cycle_type_fn(gen):
        return _cycle_type_partition_on_subset(gen, subset)

    return cycle_type_fn


def signed_cycle_type(gen, n, signed_group=None):
    """
    Signed (B_n) cycle type of a group element acting on {1..n}.

    Parameters
    ----------
    gen : permutation-like
    n : int
    signed_group : SignedPermutations or None
        Optional cache of SignedPermutations(n).
    """

    sg = signed_group or SignedPermutations(n)
    image = [int(gen(k)) for k in range(1, n + 1)]
    lam, mu = sg(image).cycle_type()
    return tuple(lam), tuple(mu)


# --------------------------------------------------------------------------- #
#  Wreath/B_n character tables (from tn.sage)
# --------------------------------------------------------------------------- #
def precompute_wreath_character_table(n):
    """
    Precompute bipartition irreducible characters for the wreath product
    S_n (hyperoctahedral context), keyed by bipartition (lambda, mu).
    """

    biparts = [
        (Partitions()(lam), Partitions()(mu))
        for k in range(n + 1)
        for lam in Partitions(k)
        for mu in Partitions(n - k)
    ]

    labels = {(tuple(lam), tuple(mu)): (lam, mu) for lam, mu in biparts}

    class_keys = list(labels.keys())
    table = {}
    for lam, mu in biparts:
        key = (tuple(lam), tuple(mu))
        row = {}
        for la, mm in biparts:
            class_key = (tuple(la), tuple(mm))
            row[class_key] = wreath_symmetric_character(
                n,
                [la, mm],
                [lam, mu],
            )
        table[key] = row

    char_vectors = {
        tuple(row[class_key] for class_key in class_keys): key
        for key, row in table.items()
    }

    return class_keys, labels, char_vectors


def identify_irreps_from_values(values_by_class, char_vectors, class_keys):
    """
    Identify a bipartition label by matching class values vector.
    """

    for class_key in class_keys:
        values_by_class.setdefault(class_key, 0)
    key = tuple(values_by_class[class_key] for class_key in class_keys)
    return char_vectors.get(key)


def precompute_symmetric_character_table(n):
    """
    GAP-backed class/character data for S_n (cycle-type partitions + irreps).

    Returns (Sn, class_data, characters) where:
      * class_data: list of {key: partition_tuple, size: class_size}
      * characters: map partition_tuple -> {partition, values, degree}
    """

    Sn, class_data, characters = sn_character_data(n)
    return Sn, class_data, characters


def precompute_symmetric_character_table_exact(n):
    """
    GAP-backed class/character data for S_n with exact integer values.

    This is intended for order-agnostic labeling by *value-vector matching*,
    i.e. matching an irreducible character by its values on conjugacy classes,
    without relying on the ordering of ``irreducible_characters`` and without
    using inner products.

    Returns (Sn, class_data, characters) where:
      * class_data: list of {key: cycle-type partition tuple, size: class size}
      * characters: map partition_tuple -> {partition, values, degree}
        with ``values`` a list of integers in the same order as class_data.
    """

    Sn = SymmetricGroup(n)
    if _gap_is_libgap():
        from sage.libs.gap.libgap import libgap

        tbl = libgap.CharacterTable("Symmetric", n)

        class_params = sage_eval(str(libgap.ClassParameters(tbl)))
        class_partitions = [entry[1] for entry in class_params]
        class_sizes = sage_eval(str(libgap.SizesConjugacyClasses(tbl)))

        char_params = sage_eval(str(libgap.CharacterParameters(tbl)))
        char_partitions = [entry[1] for entry in char_params]

        irr = libgap.Irr(tbl)
        nr_classes = int(libgap.NrConjugacyClasses(tbl))
        nr_chars = int(libgap.Length(irr))
        char_values = []
        for i in range(nr_chars):
            chi = irr[i]
            char_values.append([chi[j] for j in range(nr_classes)])
    else:
        tbl = gap('CharacterTable("Symmetric", %s)' % n)

        class_partitions = sage_eval(
            gap.eval('List(ClassParameters(%s), x -> x[2])' % tbl.name())
        )
        class_sizes = sage_eval(gap.eval('SizesConjugacyClasses(%s)' % tbl.name()))

        char_partitions = sage_eval(
            gap.eval('List(CharacterParameters(%s), x -> x[2])' % tbl.name())
        )
        char_values = sage_eval(
            gap.eval(
                'List(Irr(%s), chi -> List([1..NrConjugacyClasses(%s)], j -> chi[j]))'
                % (tbl.name(), tbl.name())
            )
        )

    class_data = [
        {"key": tuple(part), "size": size}
        for part, size in zip(class_partitions, class_sizes)
    ]

    characters = {}
    for part, values in zip(char_partitions, char_values):
        key = tuple(part)
        values_zz = [ZZ(v) for v in values]
        characters[key] = {
            "partition": Partitions(n)(part),
            "values": values_zz,
            "degree": int(values_zz[0]),
        }

    return Sn, class_data, characters


def sn_decompose(cf, *, cycle_type_fn=None, data=None):
    """
    Decompose a class function on a group isomorphic to ``S_n`` into partitions.

    INPUT:

    - ``cf`` -- a Sage ``ClassFunction`` on a group isomorphic to ``S_n``.
    - ``cycle_type_fn`` -- (default: ``None``) a function mapping a group element
      to its cycle-type partition (as a tuple).

      If ``None``, we use a default cycle-type reader based on the element's
      action on the group domain (see ``_default_sn_cycle_type_fn``).

    - ``data`` -- (default: ``None``) precomputed ``(Sn, class_data, characters)``
      from ``sn_character_data(n)``. If ``None``, infer ``n`` from ``|G| = n!`` and
      compute the character data (potentially expensive without cache).

    OUTPUT:

    A list of pairs ``(partition, multiplicity)``, where ``partition`` is a Sage
    ``Partition`` (Young diagram label of the Specht module), and
    ``multiplicity`` is an integer.

    NOTES:

    - The decomposition is computed via explicit inner products against the
      irreducible character table of ``S_n`` (from ``sn_character_data``).
    - When working with a *non-standard* permutation model isomorphic to ``S_n``,
      the only nontrivial input is a correct ``cycle_type_fn``.
    - For repeated calls at fixed ``n``, pass ``data=sn_character_data(n)`` to
      avoid reloading character tables.

    EXAMPLES::

        sage: load('core/rep2part.sage')
        sage: G = SymmetricGroup(3)
        sage: cf = ClassFunction(G, [6, 0, 0])  # regular character
        sage: dec = sn_decompose(cf, data=sn_character_data(3))
        sage: print(format_sn_decomposition(dec))
    """

    if data is None:
        order = cf.domain().order()
        n = 0
        fact = 1
        while fact < order:
            n += 1
            fact *= n
        if fact != order:
            raise ValueError("Cannot infer S_n from group order; provide data explicitly.")
        data = sn_character_data(n)

    Sn, class_data, characters = data
    if cycle_type_fn is None:
        cycle_type_fn = _default_sn_cycle_type_fn(cf.domain(), Sn.degree())
    group_order = Sn.cardinality()
    class_keys = [entry["key"] for entry in class_data]

    values = {}
    for gen in cf.domain().conjugacy_classes_representatives():
        key = cycle_type_fn(gen)
        values[key] = cf(gen)
    cf_vector = [complex(values.get(key, 0)) for key in class_keys]

    result = []
    for label, char_data in characters.items():
        irrep_vals = char_data["values"]
        inner = sum(
            cls["size"] * cf_val * ir_val.conjugate()
            for cls, cf_val, ir_val in zip(class_data, cf_vector, irrep_vals)
        ) / group_order
        if abs(inner.imag) > 1e-8:
            raise ValueError(f"inner product not real: {inner}")
        mult = int(round(inner.real))
        if mult:
            result.append((char_data["partition"], mult))

    return result


def format_sn_decomposition(decomp, *, sep=", "):
    """
    Compact string rendering for S_n decomposition.
    """

    return sep.join(f"{part}: {mult}" for part, mult in decomp)


# --------------------------------------------------------------------------- #
#  Hyperoctahedral group B_n (signed permutations)
# --------------------------------------------------------------------------- #
def bn_character_data(n):
    """
    Bipartition class/character lookup for B_n.

    Returns
    -------
    class_keys : list[tuple(tuple, tuple)]
        Bipartition labels (λ, μ).
    labels : dict
        Map from tuple(bipartition) to Partitions objects.
    char_vectors : dict
        Map from value-vectors (ordered by class_keys) to bipartition key.
    """

    n = ZZ(n)
    if n < 0:
        raise ValueError("n must be nonnegative.")
    if n == 0:
        empty = Partitions()(())
        class_keys = [((), ())]
        labels = {((), ()): (empty, empty)}
        char_vectors = {(1,): ((), ())}
        return class_keys, labels, char_vectors

    biparts = [
        (Partitions()(lam), Partitions()(mu))
        for k in range(n + 1)
        for lam in Partitions(k)
        for mu in Partitions(n - k)
    ]

    labels = {(tuple(lam), tuple(mu)): (lam, mu) for lam, mu in biparts}

    class_keys = list(labels.keys())
    table = {}
    for lam, mu in biparts:
        key = (tuple(lam), tuple(mu))
        row = {}
        for la, mm in biparts:
            class_key = (tuple(la), tuple(mm))
            row[class_key] = wreath_symmetric_character(
                n,
                [la, mm],
                [lam, mu],
            )
        table[key] = row

    char_vectors = {
        tuple(row[class_key] for class_key in class_keys): key
        for key, row in table.items()
    }

    return class_keys, labels, char_vectors


def bn_decompose(cf, *, signed_cycle_type_fn, data=None):
    """
    Decompose a class function on a group isomorphic to ``B_n`` into bipartitions.

    INPUT:

    - ``cf`` -- a Sage ``ClassFunction`` on a group isomorphic to the
      hyperoctahedral group ``B_n``.
    - ``signed_cycle_type_fn`` -- a function mapping a group element to its
      signed cycle type ``(lam, mu)``.

      The output should be a pair of partitions whose sizes sum to ``n``.
      See ``signed_cycle_type(...)`` and ``_default_signed_cycle_type_fn(n)``.

    - ``data`` -- (default: ``None``) optional precomputed
      ``(class_keys, labels, char_vectors)`` from ``bn_character_data(n)``.
      If None, attempts to read ``n`` from ``cf.domain().degree()`` (may fail for
      non-standard models).

    OUTPUT:

    A list of pairs ``((lam, mu), multiplicity)``, where ``(lam, mu)`` is a pair
    of Sage ``Partition`` objects (the bipartition label), and
    ``multiplicity`` is an integer.

    NOTES:

    - Multiplicities are obtained from Sage's ``cf.decompose()``; this routine
      only attaches *bipartition labels* by matching character values.
    - Building ``bn_character_data(n)`` uses ``wreath_symmetric_character`` and
      can be expensive; pass ``data=...`` when decomposing many coefficients.

    EXAMPLES::

        sage: load('core/rep2part.sage')
        sage: B = SignedPermutations(3)
        sage: cf = ClassFunction(B, [1] * len(B.conjugacy_classes()))  # trivial character
        sage: dec = bn_decompose(
        ....:     cf,
        ....:     signed_cycle_type_fn=_default_signed_cycle_type_fn(3),
        ....:     data=bn_character_data(3),
        ....: )
        sage: print(format_bn_decomposition(dec))
    """

    if data is None:
        # Signed permutations of size n act on n letters.
        n = getattr(cf.domain(), "degree", lambda: None)()
        if n is None:
            raise ValueError("Missing B_n rank n; provide data explicitly.")
        data = bn_character_data(n)

    class_keys, labels, char_vectors = data

    # Use built-in decompose for multiplicities; attach labels via matching.
    result = []
    for mult, character in cf.decompose():
        values_by_class = {}
        for gen in cf.domain().conjugacy_classes_representatives():
            lam, mu = signed_cycle_type_fn(gen)
            key = (tuple(lam), tuple(mu))
            values_by_class[key] = character(gen)
        for class_key in class_keys:
            values_by_class.setdefault(class_key, 0)
        vec = tuple(values_by_class[class_key] for class_key in class_keys)
        label_key = char_vectors.get(vec)
        label = labels.get(label_key, label_key)
        if label is None:
            raise ValueError("Failed to match a bipartition label.")
        result.append((label, mult))

    return result


def format_bn_decomposition(decomp, *, sep=", "):
    """
    Compact string rendering for B_n bipartition decomposition.
    """

    formatted = []
    for (lam, mu), mult in decomp:
        lam_str = "(" + ", ".join(map(str, lam)) + ")" if lam else "()"
        mu_str = "(" + ", ".join(map(str, mu)) + ")" if mu else "()"
        formatted.append(f"[ {lam_str} , {mu_str} ]:{mult}")
    return sep.join(formatted)


def sn_decomposition_dict(decomp):
    """
    Convert an S_n decomposition list into a dict keyed by partitions.

    Returns
    -------
    dict[Partition, ZZ]
    """

    out = {}
    for part, mult in decomp:
        out[part] = out.get(part, ZZ(0)) + ZZ(mult)
    return out


def bn_decomposition_dict(decomp):
    """
    Convert a B_n decomposition list into a dict keyed by bipartitions.

    Returns
    -------
    dict[tuple[Partition, Partition], ZZ]
    """

    out = {}
    for (lam, mu), mult in decomp:
        key = (lam, mu)
        out[key] = out.get(key, ZZ(0)) + ZZ(mult)
    return out


def sn_decomposition_schur(decomp):
    """
    Frobenius characteristic of an S_n decomposition, as a Schur symmetric function.

    This returns sum_{λ} m_λ * s_λ in SymmetricFunctions(QQ).schur().
    """

    s = SymmetricFunctions(QQ).schur()
    out = s.zero()
    for part, mult in decomp:
        out += ZZ(mult) * s[tuple(part)]
    return out


def bn_decomposition_schur(decomp):
    """
    Frobenius characteristic of a B_n decomposition, as a Schur tensor.

    Returns
    -------
    Tensor element in Sym ⊗ Sym
        sum_{(λ,μ)} m_{λ,μ} * (s_λ ⊗ s_μ), printed with ``#`` (plethystic style).
    """

    s = SymmetricFunctions(QQ).schur()
    one = tensor([s[[]], s[[]]])
    out = one.parent().zero()
    for (lam, mu), mult in decomp:
        out += ZZ(mult) * tensor([s[tuple(lam)], s[tuple(mu)]])
    return out


# --------------------------------------------------------------------------- #
#  Workflow wrappers (merged from the old core/rep_decompose.sage)
# --------------------------------------------------------------------------- #


def coerce_to_classfunction(rep, group=None):
    """
    Normalize input to a ClassFunction-like object and return (cf, group).

    If ``rep`` is a list/tuple of values, ``group`` is required and values
    must be ordered as ``group.conjugacy_classes_representatives()``.
    """

    if hasattr(rep, "domain") and callable(rep):
        return rep, rep.domain()
    if group is None:
        raise ValueError("rep must be a class function/character or provide group.")
    return ClassFunction(group, rep), group


def _infer_sn_n_from_order(order):
    """
    Infer n from |S_n| = n! using only group order.
    """

    order = ZZ(order)
    n = 0
    fact = ZZ(1)
    while fact < order:
        n += 1
        fact *= n
    return n if fact == order else None


def _infer_bn_n_from_order(order):
    """
    Infer n from |B_n| = 2^n * n! using only group order.
    """

    order = ZZ(order)
    n = 0
    fact = ZZ(1)
    two_pow = ZZ(1)
    while two_pow * fact < order:
        n += 1
        fact *= n
        two_pow *= 2
    return n if two_pow * fact == order else None


_STD_BN_CACHE = {}


def standard_bn_permutation_group(n):
    """
    Return the hyperoctahedral group B_n as a permutation group on ±{1..n}.

    This model is GAP-compatible and works with ``ClassFunction`` / ``cf.decompose()``.
    """

    n = ZZ(n)
    if n < 0:
        raise ValueError("n must be nonnegative.")
    if n in _STD_BN_CACHE:
        return _STD_BN_CACHE[n]

    if n == 0:
        B0 = PermutationGroup(gens=[], domain=[])
        _STD_BN_CACHE[n] = B0
        return B0

    domain = list(range(1, n + 1)) + list(range(-1, -n - 1, -1))

    gens = []
    for i in range(1, n):
        mapping = {e: e for e in domain}
        mapping[i] = i + 1
        mapping[i + 1] = i
        mapping[-i] = -(i + 1)
        mapping[-(i + 1)] = -i
        gens.append([mapping[e] for e in domain])

    mapping = {e: e for e in domain}
    mapping[1] = -1
    mapping[-1] = 1
    gens.append([mapping[e] for e in domain])

    Bn = PermutationGroup(gens=gens, domain=domain)
    _STD_BN_CACHE[n] = Bn
    return Bn


def _standard_group_for_kind(kind, n):
    """
    Return the standard reference group for a given kind and n.

    Parameters
    ----------
    kind : str
        Either ``"sn"`` or ``"bn"``.
    n : int

    Returns
    -------
    PermutationGroup
        - ``SymmetricGroup(n)`` for ``kind="sn"``.
        - A concrete permutation model of ``B_n`` (signed permutations) for ``kind="bn"``.
    """

    if kind == "sn":
        return SymmetricGroup(n)
    if kind == "bn":
        return standard_bn_permutation_group(n)
    raise ValueError("unknown group kind")


def _resolve_group_kind(G, group_kind=None, n=None, *, verify_isomorphic=True):
    """
    Decide whether the group should be treated as S_n or B_n and return (kind, n).

    When ``verify_isomorphic`` is True, we also verify that the domain group is
    actually isomorphic to the inferred standard group; this prevents accidental
    “same-order” false positives that can yield fractional decompositions.
    """

    if group_kind is not None and group_kind not in ("sn", "bn"):
        raise ValueError("group_kind must be 'sn' or 'bn'.")

    order = ZZ(G.order())
    candidates = []

    if group_kind == "sn":
        n = n or _infer_sn_n_from_order(order)
        if n is None or factorial(n) != order:
            raise ValueError("Cannot infer n for S_n; pass n explicitly.")
        candidates = [("sn", n)]
    elif group_kind == "bn":
        n = n or _infer_bn_n_from_order(order)
        if n is None or (ZZ(2) ** n) * factorial(n) != order:
            raise ValueError("Cannot infer n for B_n; pass n explicitly.")
        candidates = [("bn", n)]
    else:
        if n is not None:
            if factorial(n) == order:
                candidates.append(("sn", n))
            if (ZZ(2) ** n) * factorial(n) == order:
                candidates.append(("bn", n))
        else:
            sn_n = _infer_sn_n_from_order(order)
            bn_n = _infer_bn_n_from_order(order)
            if sn_n is not None:
                candidates.append(("sn", sn_n))
            if bn_n is not None:
                candidates.append(("bn", bn_n))

    if not candidates:
        raise ValueError("Unable to infer group kind; pass group_kind and n.")

    if not verify_isomorphic:
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError("Ambiguous group order (matches multiple candidates).")

    verified = []
    for kind, nn in candidates:
        std = _standard_group_for_kind(kind, nn)
        try:
            ok = G.is_isomorphic(std)
        except Exception as exc:
            raise ValueError("Failed to verify group isomorphism.") from exc
        if ok:
            verified.append((kind, nn))

    if len(verified) == 1:
        return verified[0]
    if not verified:
        details = ", ".join(f"{k} n={nn}" for k, nn in candidates)
        raise ValueError(
            f"Group order {order} matches ({details}) by size, but the group is not isomorphic to any of them."
        )
    raise ValueError(
        "Ambiguous group order: group is isomorphic to multiple candidates; pass group_kind and n explicitly."
    )


def pullback_classfunction(cf, hom):
    """
    Pull back a class function along a homomorphism.
    """

    if not hasattr(hom, "domain") or not callable(hom):
        raise ValueError("hom must be a Sage group homomorphism.")

    G_src = hom.domain()
    reps = G_src.conjugacy_classes_representatives()
    values = [cf(hom(rep)) for rep in reps]
    return ClassFunction(G_src, values)


def _default_signed_cycle_type_fn(n):
    """
    Default signed-cycle-type function for B_n using ``SignedPermutations(n)``.
    """

    signed_group = SignedPermutations(n)

    def _signed_cycle_type(gen):
        return signed_cycle_type(gen, n, signed_group)

    return _signed_cycle_type


def decompose_representation(
    rep,
    *,
    group=None,
    group_kind=None,
    n=None,
    pullback=None,
    signed_cycle_type_fn=None,
    sn_data=None,
    bn_data=None,
    verify_isomorphic=True,
):
    """
    Decompose a class function into S_n or B_n irreducibles with labels.

    Parameters
    ----------
    rep : ClassFunction, character, or list
        - If ``rep`` is callable and has ``domain()``, it is treated as a class function.
        - If ``rep`` is a list/tuple, you must also pass ``group=...`` and the values
          must be ordered as ``group.conjugacy_classes_representatives()``.
    group : PermutationGroup or None
        Required when ``rep`` is given as a raw list/tuple of values.
    group_kind : {"sn", "bn"} or None
        If None, infer from the group order (and optionally verify isomorphism).
    n : int or None
        Rank parameter for S_n / B_n. If None, infer from group order when possible.
    pullback : group homomorphism or None
        If provided, first pull back the class function along this homomorphism.
    signed_cycle_type_fn : callable or None
        For B_n: a function mapping a group element to a signed cycle type, used
        to label conjugacy classes consistently across isomorphic models.
    sn_data, bn_data : dict or None
        Precomputed character-table data from ``sn_character_data`` / ``bn_character_data``.
    verify_isomorphic : bool (default: True)
        If True, verify that the domain group is isomorphic to the inferred standard group
        (prevents same-order false positives that can yield nonsense decompositions).

    Returns
    -------
    dict
        Keys include:
        - ``kind``: "sn" or "bn"
        - ``n``: inferred/provided parameter
        - ``dimension``: value at the identity
        - ``decomposition``: list of (partition/bipartition, multiplicity)
        - ``formatted``: dict keyed by partitions / bipartitions
        - ``schur``: Frobenius characteristic in the appropriate symmetric-function ring

    EXAMPLES::

        sage: load('core/rep2part.sage')
        sage: G = SymmetricGroup(3)
        sage: cf = ClassFunction(G, [6, 0, 0])
        sage: decompose_representation(cf, group_kind="sn", n=3)["schur"]
        s[1, 1, 1] + 2*s[2, 1] + s[3]
    """

    cf, G = coerce_to_classfunction(rep, group=group)
    if pullback is not None:
        cf = pullback_classfunction(cf, pullback)
        G = cf.domain()

    kind, n = _resolve_group_kind(
        G, group_kind=group_kind, n=n, verify_isomorphic=verify_isomorphic
    )
    dim = cf(G.identity())

    if kind == "sn":
        sn_data = sn_data or sn_character_data(n)
        decomp = sn_decompose(cf, data=sn_data)
        formatted_dict = sn_decomposition_dict(decomp)
        schur = sn_decomposition_schur(decomp)
        return {
            "kind": "sn",
            "n": n,
            "dimension": dim,
            "decomposition": decomp,
            "formatted": formatted_dict,
            "schur": schur,
        }

    bn_data = bn_data or bn_character_data(n)
    signed_cycle_type_fn = signed_cycle_type_fn or _default_signed_cycle_type_fn(n)
    decomp = bn_decompose(cf, signed_cycle_type_fn=signed_cycle_type_fn, data=bn_data)
    formatted_dict = bn_decomposition_dict(decomp)
    schur = bn_decomposition_schur(decomp)
    return {
        "kind": "bn",
        "n": n,
        "dimension": dim,
        "decomposition": decomp,
        "formatted": formatted_dict,
        "schur": schur,
    }


def summarize_decomposition(result):
    """
    Build a short summary string from ``decompose_representation`` output.
    """

    kind = result.get("kind")
    n = result.get("n")
    dim = result.get("dimension")
    decomp = result.get("decomposition") or []
    if kind == "sn":
        formatted = format_sn_decomposition(decomp) or "0"
    elif kind == "bn":
        formatted = format_bn_decomposition(decomp) or "0"
    else:
        formatted = str(result.get("formatted")) or "0"
    return f"{kind} n={n} dim={dim} :: {formatted}"


def induced_action_on_graph_matroid(graph, W, *, check=True, return_hom=False):
    """
    Given a graph and a vertex action W, return the induced action on M(graph).
    """

    M = Matroid(graph)
    vertices = list(graph.vertices())
    edges = list(M.groundset())
    edges = sorted(edges, key=lambda e: tuple(sorted(repr(v) for v in e)))
    edge_label = {edge: idx + 1 for idx, edge in enumerate(edges)}
    bases_labeled = [[edge_label[e] for e in B] for B in M.bases()]
    M = Matroid(bases_labeled)

    def _edge_endpoints(edge):
        try:
            u, v = edge
        except Exception:
            endpoints = list(edge)
            if len(endpoints) != 2:
                raise ValueError(f"edge must have 2 endpoints: {edge}")
            u, v = endpoints
        return u, v

    try:
        domain = list(W.domain())
    except Exception as exc:
        raise ValueError("W must be a permutation group with a domain.") from exc
    domain_set = set(domain)
    vertices_set = set(vertices)

    edges_set = set(edge_label.values())
    edges_lookup = {frozenset(edge): label for edge, label in edge_label.items()}

    def _act_vertex(vertex, sigma):
        if vertex in domain_set:
            return sigma(vertex)
        return vertex

    def _edge_image(edge, sigma):
        u, v = _edge_endpoints(edge)
        e1 = frozenset((_act_vertex(u, sigma), _act_vertex(v, sigma)))
        if e1 in edges_lookup:
            return edges_lookup[e1]
        raise ValueError(f"edge image not in graph: {edge} -> {e1}")

    if check:
        for sigma in W.gens():
            for v in vertices:
                if v in domain_set:
                    img = sigma(v)
                    if img not in vertices_set:
                        raise ValueError("W does not preserve the vertex set.")

    images = []
    for sigma in W.gens():
        image = [_edge_image(edge, sigma) for edge in edges]
        if check and set(image) != edges_set:
            raise ValueError("W generator does not preserve the graph edges.")
        images.append(image)

    domain_edges = [edge_label[edge] for edge in edges]
    W_edges = PermutationGroup(gens=images, domain=domain_edges)
    if not return_hom:
        return M, W_edges

    images_as_elements = [W_edges(img) for img in images]
    chi = W.hom(images_as_elements, W_edges)
    return M, W_edges, chi
