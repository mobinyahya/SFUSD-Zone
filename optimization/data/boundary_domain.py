"""Restrict a refinement step's free nodes to the zones near them.

A pipeline refinement fixes a zoning's interior and frees a band along its
boundaries. By default every free block may still take any zone within
``max_distance``: some 30 on the real Block_0 graphs, although a boundary
block can only sensibly move to a zone it already borders.
:func:`restrict_boundary_domains` keeps, for each free node ``v``,

    candidate_zones(v) & ({source(v)} | {source(u) : u within k hops of v})

where ``source`` is the zoning the step starts from (its hint), and then
removes, to a fixed point, every (node, zone) pair the CP/MIP contiguity rows
can never satisfy: a non-centroid node may take a zone only if one of its
closer neighbours toward that zone's school can take it too
(:func:`prune_unsupported`). The k-hop restriction is a modelling choice; the
pruning is exact -- it removes only pairs no zoning feasible for those rows
uses -- so it changes nothing but model size.

Pairs that must survive are *protected* and never removed, by either part:

* the node's own source zone, when it is a candidate, so the step's hint (the
  source zoning) stays inside every domain;
* the zones :func:`optimization.data.contiguity.relax_unsupported_candidates`
  opens on free nodes along closer-neighbour paths from a source block with
  no same-zone closer neighbour (the relaxation that already widened fixed
  blocks on those paths, replayed on the free ones);

and nodes the relaxation widened (``problem.candidates`` before the call),
fixed nodes and centroids keep their domains unchanged. A protected pair the
contiguity rows cannot satisfy is kept anyway (the solver forbids it) and
counted in ``boundary_domain_unsupported_protected``.

The restriction can make a step infeasible -- the source zoning may break the
closer-neighbour rows (ReCom zonings do) and the nearby zones alone may leave
no repair -- so the pipeline re-solves an ``INFEASIBLE`` restricted step with
its full domains.
"""

from __future__ import annotations

from collections import defaultdict, deque
import time

from optimization.data import contiguity
from optimization.data.closer_neighbors import CLOSER_NEIGHBORS_GRAPH_KEY


def zones_within_hops(G, source, nodes, hops):
    """``{node: source zones of every block within ``hops`` of it}``.

    Propagates zone sets ``hops`` rounds over the whole graph, which is
    linear in edges per round; nodes absent from ``source`` contribute no
    zone but still pass sets along.
    """
    reach = {n: ({source[n]} if n in source else set()) for n in G}
    for _ in range(hops):
        reach = {
            n: reach[n].union(*(reach[u] for u in G.neighbors(n))) for n in G
        }
    return {n: reach[n] for n in nodes}


def prune_unsupported(G, centroids, centroid_school_ids, domains, mutable, protected):
    """Remove unsupported (node, zone) pairs from ``domains`` in place.

    ``domains`` maps every node to its zone set. Only nodes in ``mutable``
    lose zones, and never a pair in ``protected``. A pair ``(v, z)`` with
    ``v`` not ``z``'s centroid goes when no closer neighbour of ``v`` toward
    ``z``'s school has ``z`` in its domain; each removal requeues the pairs
    that leaned on it. Returns the number of pairs removed.

    Exact: by induction over removals, a removed pair is used by no zoning
    in which every non-centroid node has a same-zone closer neighbour.
    """
    relation = G.graph[CLOSER_NEIGHBORS_GRAPH_KEY]
    centroid_zone = {c: z for z, c in enumerate(centroids)}
    schools = [int(s) for s in centroid_school_ids]
    waiting = defaultdict(list)  # (u, z) -> mutable v whose (v, z) u supports
    queue = deque()
    for v in mutable:
        for z in domains[v]:
            if centroid_zone.get(v) == z or (v, z) in protected:
                continue
            queue.append((v, z))
            for u in relation[v][schools[z]]:
                waiting[(u, z)].append(v)
    removed = 0
    while queue:
        v, z = queue.popleft()
        if z not in domains[v]:
            continue
        if any(z in domains[u] for u in relation[v][schools[z]]):
            continue
        domains[v].discard(z)
        removed += 1
        for w in waiting.get((v, z), ()):
            if z in domains[w]:
                queue.append((w, z))
    return removed


def restrict_boundary_domains(problem, source, hops, *, prune=True, paths="shortest"):
    """Restrict ``problem``'s free nodes to nearby zones; returns metadata.

    ``problem.fixed`` and ``problem.candidates`` must already hold the step's
    split (after the pipeline's release of unsupported and unreachable
    assignments). Free nodes -- neither fixed, widened by the relaxation, nor
    a centroid -- get an explicit ``problem.candidates`` entry. ``source`` is
    the zoning the step starts from; ``paths`` the step's ``release_paths``,
    used to replay the relaxation on free nodes. ``prune=False`` skips the
    exact pruning (for a model whose contiguity is not the closer-neighbour
    rows).
    """
    started = time.perf_counter()
    G = problem.G
    fixed = problem.fixed or {}
    widened = dict(problem.candidates or {})
    centroids = set(problem.centroids)
    free = [n for n in G if n not in fixed and n not in widened and n not in centroids]
    domains = {n: set(problem.candidate_zones(n)) for n in G}
    before = {n: len(domains[n]) for n in free}

    # Zones the relaxation opens along closer paths, replayed on free nodes.
    opened = contiguity.relax_unsupported_candidates(
        G,
        source,
        list(problem.centroids),
        problem.centroid_school_ids,
        {n: set() for n in free},
        paths=paths,
    )
    protected = set()
    for n in free:
        keep = set(opened.get(n, ()))
        if n in source:
            keep.add(source[n])
        protected.update((n, z) for z in keep & domains[n])

    near = zones_within_hops(G, source, free, hops)
    unrestricted = 0
    for n in free:
        kept = {z for z in domains[n] if z in near[n] or (n, z) in protected}
        # A block whose nearby zones are all out of its distance range (a
        # source zone freed as out of range, say) keeps its candidates.
        if kept:
            domains[n] = kept
        else:
            unrestricted += 1
    after_hops = sum(len(domains[n]) for n in free)

    removed, infeasible = 0, 0
    if prune:
        hopped = {n: set(domains[n]) for n in free}
        removed = prune_unsupported(
            G, problem.centroids, problem.centroid_school_ids, domains, free, protected
        )
        # An emptied domain proves the step infeasible under the contiguity
        # rows; give the node back its zones so the solver reports it.
        for n in free:
            if not domains[n]:
                domains[n] = hopped[n]
                removed -= len(hopped[n])
                infeasible += 1
    relation = G.graph[CLOSER_NEIGHBORS_GRAPH_KEY]
    schools = problem.centroid_school_ids
    unsupported = sum(
        1
        for n, z in protected
        if not any(z in domains[u] for u in relation[n][int(schools[z])])
    )
    problem.candidates = {**widened, **{n: domains[n] for n in free}} or None
    # The hint is the source zoning, protected above; check it rather than
    # trust it, and say so if a hinted zone fell outside its node's domain.
    hint_outside = [
        n
        for n, z in (problem.hint or {}).items()
        if z not in problem.candidate_zones(n)
    ]
    sizes = [len(domains[n]) for n in free]
    total_before = sum(before.values())
    total_after = sum(sizes)
    return {
        "boundary_domain_hops": hops,
        "boundary_domain_pruned": prune,
        "boundary_domain_free_nodes": len(free),
        "boundary_domain_mean_before": total_before / len(free) if free else 0.0,
        "boundary_domain_max_before": max(before.values(), default=0),
        "boundary_domain_mean_after": total_after / len(free) if free else 0.0,
        "boundary_domain_max_after": max(sizes, default=0),
        "boundary_domain_pairs_before": total_before,
        "boundary_domain_pairs_after_hops": after_hops,
        "boundary_domain_pairs_after": total_after,
        "boundary_domain_pruned_pairs": removed,
        "boundary_domain_single_zone_nodes": sum(1 for s in sizes if s == 1),
        "boundary_domain_protected_pairs": len(protected),
        "boundary_domain_unsupported_protected": unsupported,
        "boundary_domain_unrestricted_nodes": unrestricted,
        "boundary_domain_infeasible_nodes": infeasible,
        "boundary_domain_hint_outside": len(hint_outside),
        "boundary_domain_seconds": time.perf_counter() - started,
    }
