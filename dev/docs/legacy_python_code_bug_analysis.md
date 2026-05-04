# Legacy Python Codebase Bugs

Three mathematical bugs were present in Viktor Polak's R-to-Python translation
([ucgmsim/Vs30 at `6db1ad6`](https://github.com/ucgmsim/Vs30/tree/6db1ad6),
first commit 2021-02-04). The original R code
([fostergeotech/Vs30_NZ](https://github.com/fostergeotech/Vs30_NZ)) is correct.
All three bugs are fixed in the refactored codebase.

| Bug | Description | Affected code path | Affects foster_2019_approx or viktor_cpt_clustering? |
|-----|-------------|-------------------|----------------------------------------------|
| 1 | Missing mean-shift term in Bayesian variance update | `posterior()` | No — neither version calls this path |
| 2 | Off-by-one category indexing in Bayesian update loop | `posterior()` | No — same reason |
| 3 | `log(stdv)` instead of `log(vs30)` in model combination | `combine()` | Fixed within 21 days; predates any archived outputs |

## Bug 1: Missing mean-shift term in posterior variance

The R code (`bayes.R`, `priorToPostVar()`) implements Gelman (2013) Eq. 3.9:

```
ν_n σ²_n = ν₀ σ²₀ + σ²_meas + κ₀n/(κ₀+n) · (ȳ - μ₀)²
```

The Python translation omitted the third (mean-shift) term, underestimating
posterior variance when observations disagree with the prior mean.

## Bug 2: Off-by-one category indexing

Category IDs in rasters are 1-indexed (1..15), but the model arrays are
0-indexed (0..14). The Python code used the raster ID directly as an array
index without subtracting 1, so every observation updated the wrong category.

## Bug 3: `log(stdv)` instead of `log(vs30)` in model combination

The combined standard deviation formula (Foster 2019, Eq. 9) requires
`(log(vs30_a) - log(vs30_combined))²`. The Python translation used
`log(stdv_a)` instead of `log(vs30_a)`, producing meaningless results.
Fixed by Viktor Polak 21 days after first commit.

## References

- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd Edition, Section 3.3.
- Original R code: [fostergeotech/Vs30_NZ](https://github.com/fostergeotech/Vs30_NZ)
- Legacy Python code: [ucgmsim/Vs30 at `6db1ad6`](https://github.com/ucgmsim/Vs30/tree/6db1ad6)
- Refactored implementations: `vs30/category.py`, `vs30/utils.py`
