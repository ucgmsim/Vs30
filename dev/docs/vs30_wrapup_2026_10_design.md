# Vs30 repo wrap-up — design

Date: 2026-10-15. Status: approved by Andrew on 2026-10-15.
Archived here after the wrap-up; it was written outside the repo while the work was in progress.

## Goal

Finish all open work on `ucgmsim/Vs30` quickly:

1. `master` contains every beneficial change from every branch.
2. `vs_calc` reports a sane, clearly provisional Vs30 uncertainty instead of the broken values it computes today.
3. CI follows the team's current approach: the shared `ucgmsim/meta-ci-action` workflows, as in `high-frequency`.

Out of scope: publishing to PyPI; the full uncertainty model proposed in `vs30_uncertainty_followup.md` (moving from `VsViewer/docs/` to the developer archive); any change to the Vs30 map science.

## Decisions

Confirmed by Andrew on 2026-10-15:

- No pull requests. Everything reaches `master` in one push once all phases are complete (colleagues agree).
- No PyPI release (little benefit).
- Uncertainty stopgap: Option C with 0.2, defined in Phase 2.
- Adopt the meta-ci-action CI used by `high-frequency`.
- Remove the `GDAL` (osgeo) dependency and replace its two calls (Phase 3a).
- Keep `dev_scripts_archive` on origin as the developer archive, so future developers can see the maintainer scripts. Developer docs go there, under `dev/docs/`, rather than on `master`.
- Branch clean-up covers only branches Andrew created (list in Phase 5). On 2026-10-15 this was checked against commit authors, PR authors and the local branch-creation records. Every other branch on origin is left alone.
- **Notes rule:** every note, README or `.md` file left in `master` must have a clear production purpose. Everything else moves out of the repo and is deleted from `master`; it stays in git history.
- `VsViewer/docs/` is split, as in the Phase 1 table. The NZGD runbook and results go to `nzgd/docs/`, and the vs_calc notes go to the developer archive. The journal PDFs in `reference_papers/` move out of the repo and are cited by DOI instead.
- The up-to-date `CLAUDE.md` goes on the developer archive, never on `master` (Phase 4).
- Merge `andrew-latest`, which contains all of `vs30_refactor` plus the VsViewer work.
- Phase 2b, which fills `vs30_stddev` in the NZGD database (nzgd repo), is in scope.

Repo facts that shape the final push (checked 2026-10-15):

- `master` is protected: it needs 1 approving review, and "Include administrators" (`enforce_admins`) is on, so even an admin push is rejected. Andrew has admin rights.
- `dev_scripts_archive` is locked by the ruleset `protect`, created on 2026-05-13 when the `dev/` scripts were archived. It blocks deletion and updates and has no bypass. The branch stays, locked.
- PR #45 (`vs30_refactor` → `master`) is open. GitHub marks it merged automatically once `master` contains its head commit `5569310`.

## How the work is organised

- **Local only until Phase 5:**
  - Vs30 work happens on local `master` in the main checkout, where `vs30_venv`'s editable install points.
  - Developer-archive work happens on local `dev_scripts_archive`, in the worktree `.worktrees/dev_scripts_archive`.
  - nzgd work happens on local `dev`.
  - Nothing is pushed until Phase 5.
- **Each phase ends with its verification passing locally.**
- **First CI run is the final push.** The shared CI runs only on pull requests and pushes to `master`. Before that, the same checks run locally with `uvx lefthook run pre-push --all-files --no-tty`, which mirrors the shared CI jobs.

## Phase 1 — Integrate locally

On local `master`, run `git merge --no-ff andrew-latest`.

Branch verdicts (deletions happen in Phase 5):

| Branch | Verdict |
|---|---|
| `andrew-latest` | Merge, then delete |
| `vs30_refactor`, `add-jaehwi-effstress`, `cs_comments`, `known-issue-mvn-phase-leak` | Delete. Already inside `andrew-latest`; `cs_comments`' one extra commit is patch-identical to one in `vs30_refactor` |
| `dev_scripts_archive` | Keep: the developer archive of maintainer scripts and notes, locked by its ruleset |
| `geology_rasterize_perf`, `prefilter_perf` (local only) | Delete. Their conclusions were acted on: rasterisation isn't a bottleneck, and the KDTree pre-filter landed in `9645b43` |
| `deployment`, `feedback`, `new_vs30_measurements`, `robin_fixes`, `sung` | Leave alone: created by others. `robin_fixes` is Joel's; its CPT code was superseded by the version merged in 2024, but it stays |

Notes cleanup, per the notes rule:

| File | Verdict |
|---|---|
| `README.md`, `LICENSE` | Keep: installation and licence |
| `wiki/Home.md`, `wiki/Usage.md`, `wiki/images/*` | Keep: the user guide, published to the GitHub wiki. All ten images are referenced |
| `vs30/resources/observations/README.md` | Keep: format and provenance of the observation data the package ships. It is also the only description of the CSV format that custom configs need |
| `VsViewer/frontend/README.md` | Keep: how to run the VsViewer web frontend |
| `VsViewer/vs_calc/examples/` | Keep only if both scripts still run, since they're the only usage examples for vs_calc. Otherwise delete |
| `VsViewer/docs/vs30_database_population.md`, `vs30_population_results_20260919.md` | Move to `nzgd/docs/` in a commit on nzgd `dev`. They document nzgd's population tool and run |
| `VsViewer/docs/vs30_uncertainty_followup.md`, `effective_stress_validation.md`, `validation/` | Move to `dev/docs/vs_calc/` on the archive in a local commit, keeping `validation/` beside the validation note so its links still work |
| `reference_papers/*.pdf` | Move to `/home/arr65/data/temp_vs30_doc_files/reference_papers/` and cite by DOI |

Then fix every reference to a moved file:
- The Worden PDF path in the `vs30/spatial.py` docstring becomes a citation.
- `wiki/Home.md` drops its "PDF in this repo" link.
- The docstrings of `validate_spt_database.py` and nzgd's `batch.py` point at the vs_calc notes' new home on the archive, as GitHub URLs.
- Links between the two groups of moved docs become GitHub URLs.

Workflow triggers:
- `vsviewer-pytest.yml` runs on pull requests and pushes to `master` only.
- `wiki.yml` runs on pushes to `master`. It currently watches `main`, which doesn't exist.

Verification:
- `pytest tests/ --runslow` passes.
- `python -m pytest VsViewer/tests` passes.
- The clean-environment CI reproduction passes (recipe in the `project-ci-local-repro` memory).

Done when local `master` holds `andrew-latest` plus the cleanup, and all three checks pass.

## Phase 2 — Provisional Vs30 uncertainty in vs_calc (Option C)

For a Vs profile reaching integer depth Z (5–30 m):

    sigma_ln(Vs30) = sqrt(sigma_profile**2 + sigma_extrapolation(Z)**2)

- **`sigma_profile` = 0.2.** This is the lognormal standard deviation Foster et al. (2019) assigned to all Vs30 data except Kaiser Q1 (0.1), including McGann's CPT-derived Vs30 and the surface-wave data. It applies to every profile, whatever its source correlation, measured profiles included.
- **`sigma_extrapolation(Z)`** is ln(10) times the published Boore residual standard deviation (log10 units) at depth Z. The tables are the ones already in `vs30_correlations.py`: Boore et al. (2011) for `boore_2011`, Boore (2004) for `boore_2004`. It is 0 when Z = 30, because there is no extrapolation.
- **No Boore slope factor.** The slope d ln Vs30 / d ln VsZ (0.75–1.01) is deliberately not applied to `sigma_profile`. This is simpler and slightly conservative.
- **Units:** standard deviation of natural-log Vs30. This is the convention the vs30 package and VsViewer's `exp(±vs_sd)` plots already use.

The values don't depend on VsZ:

| Z (m) | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| `boore_2011` | 0.34 | 0.28 | 0.24 | 0.22 | 0.20 | 0.20 |
| `boore_2004` | — | 0.26 | 0.23 | 0.21 | 0.20 | 0.20 |

Code changes in `VsViewer/vs_calc`:
- **`constants.py`:** add the 0.2 constant, with a comment citing Foster et al. (2019).
- **`vs30_correlations.py`:** `boore_2011` and `boore_2004` return the extrapolation σ in natural-log units. This replaces the malformed Boore 2011 expression and the raw log10 value Boore 2004 returns now.
- **`VsProfile.calc_vs30`:** combine the two terms as above at every depth, so a 30 m profile returns 0.2 instead of 0.
- **`calc_weightings.calculate_weighted_vs30`:** compute the combined σ in log space with the law of total variance, `sqrt(sum_i w_i * (sigma_i**2 + (ln Vs30_i - sum_j w_j ln Vs30_j)**2))`. This replaces the current mix of natural-log and m/s variances followed by a log. The weighted central Vs30 stays the arithmetic weighted mean, unchanged.
- **Docstrings:** state the units and that the values are provisional.

Unchanged: central Vs30 values, per-depth `vs_sd` (still used by the profile plots), and each correlation's own σ.

Tests in `VsViewer/tests/vs_calc/`:
- A 30 m profile gives 0.2.
- `boore_2011` at Z = 5 gives sqrt(0.2² + (ln10 × 0.119)²).
- `boore_2004` at Z = 10 gives sqrt(0.2² + (ln10 × 0.07126)²).
- σ decreases monotonically from 5 m to 30 m for both correlations.
- The weighted σ of identical 30 m profiles equals 0.2, and it grows with spread.
- The central Vs30 of an existing test profile is unchanged.

Docs: no new `.md` file. The provisional σ is documented where it's used: the vs_calc docstrings and the comment on the 0.2 constant. That covers the formula, the precedent and its limits: no uncertainty from the choice of correlation, none from input assumptions (groundwater, unit weights, soil class), and no depth-covariance model. The follow-up note, by then in the archive's `dev/docs/vs_calc/`, gets a status line in Phase 2b.

Done when the VsViewer tests pass.

## Phase 2b — Fill `vs30_stddev` in the NZGD database (nzgd repo)

- `batch.py` keeps `profile.vs30_sd` with each result and inserts it instead of NULL. The run manifest's `uncertainty_policy` records the Option C formula and that it is provisional.
- A new publish mode fills `vs30_stddev` where it is NULL. It matches on the natural key and requires the stored `vs30` to equal the recomputed one, stopping on any mismatch. It keeps the existing publish's safety rules: full backup first, one transaction.
- Recalculate into a new run directory (about 5 minutes with 8 workers), using vs_calc from the Vs30 main checkout. Then publish, after Andrew confirms. The database is local, so this doesn't wait for the Phase 5 push.
- Document the fill mode and the column's unit (σ of ln Vs30) in `nzgd/docs/vs30_database_population.md`, the runbook moved there in Phase 1.
- Files touched in nzgd: `nzgd/scripts/estimate_vs30/batch.py`, `tests/estimate_vs30/test_batch.py` and that runbook. The database file itself is outside any repo.
- Add a status line to the uncertainty follow-up note in the archive's `dev/docs/vs_calc/`: the provisional stopgap is in place, and the full model remains deferred.

Done when every estimate row has a finite σ between 0.20 and 0.34, central values are unchanged, and the backup is kept.

## Phase 3 — Team CI

**3a. Remove GDAL**
- In `raster.compute_coast_distance_raster`:
  1. Rasterize the coast polygons onto the padded grid with `rasterio.features.rasterize`, using the pixel-centre rule as `gdal.Rasterize` does.
  2. Run `scipy.ndimage.distance_transform_edt` with sampling `(dy, dx)` to get the distance from each land pixel centre to the nearest sea pixel centre.
  3. Clip at the 20 km cap. If there is no sea in range, the value is the cap.
- Drop `from osgeo import gdal` and the temporary file. Remove GDAL from the dependencies and the deptry map, and delete the GDAL-warning test.
- A throwaway check on 2026-10-15 already compared the two methods: all of NZ at 1 km, 500 m and 200 m, plus Wellington and Christchurch at 100 m.
  - 99.8% of pixels are identical. The rest differ by at most 0.25 of a pixel, and GDAL's distance is always the longer one.
  - At those pixels GDAL's propagation misses the true nearest sea pixel (for example 300.0 m against the exact diagonal 282.8 m). scipy's transform is exact.
- Expected effect on the 5 km grid benchmarks:
  - One pixel moves by 2.4% in Vs30 in each of `modified_foster_2019` and `viktor_cpt_clustering`, the two models with the coastal adjustment.
  - `jaehwi_v1p0` and points mode are unaffected.
  - Regenerate those two benchmark tifs from the refactored pipeline once Andrew agrees.
- Everything else, including the `--runslow` tier, must pass unchanged. Any other failure means stop and report.

**3b. Package data.** Add `configs/*.yaml` to package data. Non-editable installs, including `pip install git+…`, currently lack the bundled model configs.

**3c. `pyproject.toml`**
- `[dependency-groups] dev = ["deptry", "npdlint", "ruff", "ty"]`, and add `pytest-cov` to the `test` extra.
- `[tool.deptry] pep621_dev_dependency_groups = ["test", "types"]`, replacing the `-ddg` CLI flag.
- `[tool.npdlint]` with `include = ["vs30"]` and the same rule codes as today's numpydoc checks. Drop `[tool.numpydoc_validation]`.
- `[tool.pytest.ini_options] testpaths = ["tests"]`.
- Commit `uv.lock`; the shared ruff job reads the ruff version from it.

**3d. Workflows**
- **New `ci.yml`:** calls `ucgmsim/meta-ci-action/.github/workflows/ci.yml@v2` on pull requests and pushes to `master`. Coverage is enabled for `vs30`, with the gate set to the default suite's measured coverage, rounded down.
- **Delete:** `deptry.yml`, `numpydoc.yml`, `pytest.yaml`, `ruff.yml` and `types.yml`.
- **New `extra-checks.yml`** (pushes to `master`, plus manual runs) with two jobs:
  - the `--runslow -m slow` tier
  - a wheel smoke test: build the wheel, install it into a clean venv, run `vs30 points` on one site and check the Vs30 is finite
- **Keep:** `vsviewer-pytest.yml` and `wiki.yml`, as fixed in Phase 1.
- **Add:** `lefthook.yml` and `.yamllint.yml`, as in `high-frequency`.

**3e. README.** Installation stays `pip install git+…`. Add a short development section: `uv sync --all-extras --dev` and `uvx lefthook install`.

Done when all of these pass locally:
- the lefthook pre-push gate with `--all-files`
- a fresh `uv sync --all-extras --dev` from scratch
- the slow tier
- the wheel smoke test

## Phase 4 — Close-out (local)

- Bump the version in `pyproject.toml` to 2026.10.1 and create the local tag `v2026.10.1`, so users can pin `pip install git+…@v2026.10.1`.
- Update the local `CLAUDE.md` for the wrap-up: `dev/` is gone, GDAL is gone, CI is new, and vs_calc has the provisional σ. It stays untracked in the main checkout.
- Finish the developer archive on local `dev_scripts_archive`. The issue-hunt record (`dev/docs/issue_hunt_2026_10.md`, trimmed from the two `temp_issue_hunt_*.md` notes) is already committed there as `3b842ff`. Still to do:
  - Add a note at the top of the branch's `README.md`: what the branch is, that its code is a snapshot from 13 May 2026, and that `master` is the current code.
  - Add to `dev/docs/`: this spec and the two parked design notes (HDF5 regeneration; qmap/coast v2). The vs_calc notes arrive in Phase 1.
  - Add the 2026-10-15 GDAL-vs-scipy coast-distance comparison under `dev/scripts/investigations/`.
  - Replace the stale May `dev/CLAUDE.md` with the up-to-date `CLAUDE.md`, made portable:
    - Machine paths become general instructions: `uv` setup, and the legacy code via the `legacy-final` tag plus Jaehwi's fork.
    - It is updated for the wrap-up's changes.
    - A header line says it describes `master` as of the wrap-up, and can be copied to the root of a `master` checkout.
    - It goes in `dev/` rather than the branch root, so Claude Code doesn't auto-load it for the May snapshot code.
- Add one line to `master`'s README pointing maintainers to `dev_scripts_archive`.
- Final notes audit: list every `.md`, README and note file in `master` and check each against the notes rule.
- Move `temp_issue_hunt_*.md` to Trash, since their trimmed content is now in the archive (Andrew's call).

## Phase 5 — Publish (one push)

1. Run the full local verification once more on the final `master`: `pytest tests/ --runslow`, the VsViewer tests and the lefthook gate.
2. Create the local tag `legacy-final` on `1f49fad` (the current master), so colleagues still on the pre-refactor code can pin `pip install git+…@legacy-final`.
3. With Andrew's go-ahead:
   1. Turn off "Include administrators" on `master`'s protection: `gh api -X DELETE repos/ucgmsim/Vs30/branches/master/protection/enforce_admins`.
   2. Push `master` and the tags `legacy-final` and `v2026.10.1`.
   3. Turn the setting straight back on with `gh api -X POST …/enforce_admins`, and check it reads enabled.
   4. Set the `protect` ruleset's enforcement to disabled, push `dev_scripts_archive` (a fast-forward), set it back to active, and check.
4. Watch the CI runs on `master` with `gh run watch`. If anything fails, read the logs (readable now that Andrew is admin) and fix forward. A follow-up push repeats step 3.
5. Check that PR #45 shows as merged.
6. Confirm the deletion list with Andrew, then delete the remote branches `vs30_refactor`, `andrew-latest`, `add-jaehwi-effstress`, `cs_comments` and `known-issue-mvn-phase-leak`.
7. Push nzgd `dev`.
8. Remove the `.worktrees/` worktrees. Delete the local copies of the deleted branches, plus `geology_rasterize_perf` and `prefilter_perf`.

Done when:
- `origin/master` holds the wrap-up
- CI is green
- branch protection is fully restored
- origin's branches are `master`, `dev_scripts_archive`, and the untouched `deployment`, `feedback`, `new_vs30_measurements`, `robin_fixes` and `sung`

## Order and effort

Phases run in the order 1 → 2 → 2b → 3 → 4 → 5. Phase 2b can overlap Phase 3. About 3 working days in total.

## Risks

- The final push needs branch protection relaxed briefly, and the archive's ruleset disabled briefly. Phase 5 step 3 turns each back on as part of the same action.
- The shared CI's first real run is the final push. The local lefthook gate mirrors it, and anything it misses is fixed forward.
- Without GDAL, small proximity differences could move benchmark pixels. The stop-and-report rule in 3a covers this.
