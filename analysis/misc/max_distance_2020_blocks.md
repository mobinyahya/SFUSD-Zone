# Minimum `max_distance` for 2020 Blocks

Calculated using the canonical `Block_0` graph and the active centroid
configurations in `Config/centroids.yaml`.

For each configuration, the required distance is:

```text
max over non-exempt blocks(
    distance to nearest configured centroid
)
```

Values are in miles and rounded up to 0.001 miles, so they can be used
directly as `max_distance` values.

| `centroids_type` | Minimum `max_distance` |
|---|---:|
| `2-zone-1` | 6.694 |
| `1-zone-example` | 7.655 |
| `2-zone-rec-1` | 5.692 |
| `4-zone-rec-1` | 4.954 |
| `4-zone-rec-2` | 4.831 |
| `4-zone-rec-3` | 5.265 |
| `4-zone-rec-4` | 4.831 |
| `7-zone-rec-1` | 6.173 |
| `7-zone-rec-2` | 4.402 |
| `8-zone-rec-1` | 6.307 |
| `4-zone-overlap-1` | 7.679 |
| `7-zone-overlap-1` | 7.243 |
| `14-zone-overlap-1` | 5.858 |
| `10-zone-mcmc` | 2.763 |
| `2-zone-mcmc` | 7.243 |
| `10-zone-mcmc-balance` | 2.285 |
| `2-zone-mcmc-balance` | 7.243 |
| `5-zone-AF` | 5.022 |
| `5-zone-AF-relocated` | 3.089 |
| `5-zone-2` | 5.703 |
| `6-zone-1` | 3.444 |
| `6-zone-2` | 2.726 |
| `6-zone-3` | 2.639 |
| `6-zone-9` | 2.998 |
| `7-zone-14` | 2.726 |
| `7-zone-19` | 2.726 |
| `8-zone-22` | 2.395 |
| `8-zone-24` | 2.639 |
| `8-zone-25` | 2.983 |
| `8-zone-26` | 2.726 |
| `10-zone-3` | 2.405 |
| `10-zone-9` | 3.828 |
| `10-zone-11` | 2.439 |
| `10-zone_MCMC` | 2.439 |
| `13-zone-5` | 2.767 |
| `13-zone-6` | 2.281 |
| `13-zone-7` | 2.639 |
| `18-zone-1` | 1.957 |
| `18-zone-2` | 2.639 |
| `18-zone-3` | 2.261 |
| `18-zone-5` | 2.281 |
| `18-zone-6` | 1.958 |
| `18-zone-7` | 1.935 |
| `18-zone-8` | 2.634 |
| `18-zone-9` | 1.958 |
| `18-zone-10` | 2.047 |
| `56-zone-1` | 1.706 |
| `56-zone-2` | 1.706 |
| `57-zone-1` | 1.706 |
| `59-zone` | 1.706 |

The graph used for this calculation contains 5,948 blocks: 5,893 blocks
constrained by `max_distance` and 55 Treasure/Yerba Buena blocks exempt from
the distance restriction. The results are identical for the
`summer-26-zoning` scenario and the `legacy` scenario with
`geography_vintage: "2020"`.

These values only ensure that every non-exempt block has at least one centroid
candidate. They do not guarantee feasibility under capacity, diversity, or
other optimization constraints.
