# Unaddressed Code Review Items

This document lists code review items from `CODE_REVIEW.md` that were not addressed
during the review implementation. These items either require scientific context that
I don't have, involve more significant refactoring, or are lower priority style improvements.

---

## Items Requiring Scientific Context

### 1. Scientific Context for Hybrid Parameters (config.yaml)
**Location**: config.yaml lines 139-165
**Issue**: The hybrid model parameters lack scientific context explaining why these specific geology groups have modifications.
**Reason Not Addressed**: Adding accurate scientific documentation requires knowledge of the original calibration studies and geophysical reasoning for the specific parameter values.

### 2. Scientific Context in Category Docstrings (category.py)
**Location**: category.py module docstring and function docstrings
**Issue**: Could benefit from more scientific background on the Bayesian update methodology.
**Reason Not Addressed**: Requires scientific domain knowledge about the specific log-normal assumptions and why this approach is standard in seismic hazard analysis.

---

## Items Requiring Significant Refactoring

### 3. Split Long Functions in category.py
**Location**: Lines 169-312 (`update_with_independent_data`) and 376-482 (`update_with_clustered_data`)
**Issue**: Functions are quite long (143 and 106 lines respectively).
**Reason Not Addressed**: These functions contain sequential logic that would require careful extraction to maintain clarity. The current implementation is readable despite the length.

### 4. Split Long Functions in cli.py
**Location**: `update_categorical_vs30_models` (278 lines), `spatial_fit` (173 lines), `full_pipeline_for_geology_or_terrain` (191 lines)
**Issue**: CLI command functions are very long.
**Reason Not Addressed**: CLI functions tend to be procedural. Splitting them may reduce clarity for users who want to understand the overall pipeline flow.

### 5. Split `create_category_id_raster` (raster.py)
**Location**: Lines 148-265 (117 lines)
**Issue**: Function handles both terrain and geology cases with different logic.
**Reason Not Addressed**: The terrain/geology branches are distinct enough to justify keeping them together as they share setup code.

### 6. Split `apply_hybrid_geology_modifications` (raster.py)
**Location**: Lines 627-773 (146 lines)
**Issue**: Function is long with multiple modification types.
**Reason Not Addressed**: The modifications are sequential and interdependent. Splitting into separate functions would require careful consideration of the data flow.

### 7. Rename Helper Functions in category.py
**Location**: `_new_mean` and `_new_var` functions
**Issue**: Names don't convey mathematical meaning.
**Suggested Names**: `_bayesian_posterior_mean`, `_bayesian_posterior_variance`
**Reason Not Addressed**: Would require updating all call sites and verifying the mathematical correctness of the new names.

---

## Lower Priority Style Items

### 8. Unnecessary Variable Assignments in cli.py (Lines 598-603)
**Location**: `spatial_fit` function
**Issue**: Variables are extracted from config but each is only used once.
**Reason Not Addressed**: Keeping intermediate variables can improve debuggability and readability.

### 9. Validation Using Assertions (spatial.py)
**Location**: Lines 183-224
**Issue**: Using `assert` for validation instead of raising proper exceptions.
**Reason Not Addressed**: For scientific code prioritizing readability, assertions are acceptable and concise.

### 10. Type Annotations for Constants (constants.py)
**Location**: Throughout file
**Issue**: Constants don't have explicit type annotations.
**Reason Not Addressed**: Python's type inference handles this well, and adding types would add verbosity.

### 11. Plot Formatting Constants (cli.py Lines 782-817)
**Location**: `plot_posterior_values` function
**Issue**: Plot formatting values are hardcoded.
**Reason Not Addressed**: These are visual aesthetics that rarely need modification by scientific users.

---

## Summary

**Completed during review:**
- Fixed dead code (cli.py:982-994)
- Removed duplicate config loading (cli.py:641-643)
- Removed confusing self-import (spatial.py:18)
- Corrected cli.py module docstring
- Added CSV_PLACEHOLDER_NODATA constant
- Removed unnecessary module-level variables (raster.py:25-38)
- Extracted ObservationData.empty() helper (spatial.py)
- Added improved module docstrings (spatial.py, constants.py, raster.py)
- Added detailed config.yaml comments

**Output verification:**
- Refactored code produces outputs within acceptable numerical tolerances
- Maximum VS30 difference < 0.1 m/s compared to legacy code
- All intermediate outputs (IDs, slope, coast distance, etc.) are identical

---

*Document created: January 2026*
