from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from typing import Tuple, Dict, Optional
from scipy.stats import norm

# ------------------------------------------------------------------ #
# Existing helpers (unchanged)
# ------------------------------------------------------------------ #
def _z_test_diff(p1: float, p2: float, n1: int, n2: int):
    if any(map(pd.isna, [p1, p2, n1, n2])) or n1 == 0 or n2 == 0:
        return np.nan, np.nan
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p2 - p1) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return p2 - p1, p


def pct_or_blank(v):           # % formatter that hides NaNs
    return "" if pd.isna(v) else f"{v:.2%}"


def diff_formatter(v, is_diff_row=False, hide_for_diff=False):
    """Custom formatter that hides values for Diff rows when specified"""
    if is_diff_row and hide_for_diff:
        return ""
    return "" if pd.isna(v) else f"{v:.2%}"


def _colour(val, p):
    if pd.isna(val) or pd.isna(p):
        return "black"
    if p < 0.05:
        return "green" if val > 0 else "red"
    if p < 0.10:
        return "blue" if val > 0 else "orange"
    return "black"


# ------------------------------------------------------------------ #
# Fixed build_strategy_table() – includes accts, accts_pct, bal cols
# ------------------------------------------------------------------ #
def build_strategy_table(
    df: pd.DataFrame,
    *,
    test_label: str,
    control_label: str,
    month_col: str = "cycle_start_year_month",
    accts_col: str = "accts",
    bal_col: str = "bal_dollars",
    metric_prefix: str = "rr",
    column_names: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Styler]:

    # Default column name mappings
    default_column_names = {
        month_col: "Month",
        "strategy": "Strategy", 
        accts_col: "Accts",
        "accts_pct": "Acct %",
        bal_col: "Balance $",
        "bal_pct": "Bal %",
        "rr_2_to_3_num": "2 to 3 #",
        "rr_2_to_3_dollars": "2 to 3 $",
        "rr_2_to_4_num": "2 to 4 #", 
        "rr_2_to_4_dollars": "2 to 4 $",
        "rr_2_to_5_num": "2 to 5 #",
        "rr_2_to_5_dollars": "2 to 5 $",
        "rr_2_to_6_num": "2 to 6 #",
        "rr_2_to_6_dollars": "2 to 6 $",
        "rr_2_to_7_num": "2 to 7 #",
        "rr_2_to_7_dollars": "2 to 7 $",
        "rr_2_to_8_num": "2 to 8 #",
        "rr_2_to_8_dollars": "2 to 8 $",
    }
    
    # Update with user-provided column names
    if column_names:
        default_column_names.update(column_names)
    
    final_column_names = default_column_names

    metric_cols = [c for c in df.columns if c.startswith(metric_prefix)]
    required = {month_col, accts_col, bal_col, "strategy"} | set(metric_cols)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Validate that both labels exist in the data
    available_strategies = set(df["strategy"].unique())
    if test_label not in available_strategies:
        raise ValueError(f"Test label '{test_label}' not found in data. Available: {available_strategies}")
    if control_label not in available_strategies:
        raise ValueError(f"Control label '{control_label}' not found in data. Available: {available_strategies}")

    # ---- aggregate duplicates & compute weighted proportions --------
    agg_rows = []
    for (month, strat), g in df.groupby([month_col, "strategy"]):
        if strat not in {test_label, control_label}:
            continue
        n = g[accts_col].sum()
        bal = g[bal_col].sum()
        row = {month_col: month, "strategy": strat, accts_col: n, bal_col: bal}
        for m in metric_cols:
            successes = (g[m] * g[accts_col]).sum()
            row[m] = successes / n if n else np.nan
        agg_rows.append(row)

    tidy = pd.DataFrame(agg_rows)
    if tidy.empty:
        raise ValueError("No valid cohort data found.")

    # ---- add % columns (share of month total) -----------------------
    for month, g in tidy.groupby(month_col):
        tot_accts = g[accts_col].sum()
        tot_bal = g[bal_col].sum()
        idx = tidy[month_col] == month
        if tot_accts > 0:
            tidy.loc[idx, "accts_pct"] = tidy.loc[idx, accts_col] / tot_accts
        else:
            tidy.loc[idx, "accts_pct"] = np.nan
        if tot_bal > 0:
            tidy.loc[idx, "bal_pct"] = tidy.loc[idx, bal_col] / tot_bal
        else:
            tidy.loc[idx, "bal_pct"] = np.nan

    # ---- create Diff rows (metric diffs only) -----------------------
    diff_rows = []
    pval_data = {}  # Store p-values separately for styling
    
    for month in tidy[month_col].unique():
        test_data = tidy.query(f"{month_col} == @month and strategy == @test_label")
        ctrl_data = tidy.query(f"{month_col} == @month and strategy == @control_label")
        
        # Skip if either group is missing for this month
        if test_data.empty or ctrl_data.empty:
            continue
            
        test = test_data.iloc[0]
        ctrl = ctrl_data.iloc[0]

        diff = {
            month_col: month,
            "strategy": "Diff",
            accts_col: np.nan,
            "accts_pct": np.nan,
            bal_col: np.nan,
            "bal_pct": np.nan,
        }
        
        month_pvals = {}
        # rr metric differences + p-values
        for m in metric_cols:
            d, p = _z_test_diff(
                ctrl[m], test[m], ctrl[accts_col], test[accts_col]  # ctrl first, then test
            )
            diff[m] = d
            month_pvals[m] = p
        
        diff_rows.append(diff)
        pval_data[month] = month_pvals

    tidy_full = pd.concat([tidy, pd.DataFrame(diff_rows)], ignore_index=True)

    # ---- Order strategies properly: Control, Test, Diff ----
    strategy_order = [control_label, test_label, "Diff"]
    tidy_full["strategy"] = pd.Categorical(
        tidy_full["strategy"], 
        categories=strategy_order, 
        ordered=True
    )

    # ---- wide table with MultiIndex rows ---------------------------
    display_cols = (
        [accts_col, "accts_pct", bal_col, "bal_pct"] + metric_cols
    )
    wide = (
        tidy_full
        .set_index([month_col, "strategy"])
        [display_cols]
        .sort_index()
    )

    # Apply column renaming
    wide = wide.rename(columns=final_column_names)
    
    # Update display_cols with new names for formatting
    display_cols_renamed = [final_column_names.get(col, col) for col in display_cols]
    metric_cols_renamed = [final_column_names.get(col, col) for col in metric_cols]

    # ---- Styler -----------------------------------------------------
    # column-specific formatters
    fmt = {}
    
    # Handle the renamed columns for formatting
    accts_renamed = final_column_names.get(accts_col, accts_col)
    bal_renamed = final_column_names.get(bal_col, bal_col)
    accts_pct_renamed = final_column_names.get("accts_pct", "accts_pct")
    bal_pct_renamed = final_column_names.get("bal_pct", "bal_pct")
    
    fmt[accts_renamed] = "{:,}"
    fmt[bal_renamed] = "{:,}"
    fmt[accts_pct_renamed] = "{:.1%}"
    fmt[bal_pct_renamed] = "{:.1%}"
    
    # Add metric column formatting
    for orig_col, renamed_col in zip(metric_cols, metric_cols_renamed):
        fmt[renamed_col] = pct_or_blank

    def colour_row(s: pd.Series):
        month, strat = s.name
        if strat != "Diff" or month not in pval_data:
            return [""] * len(s)
        pvals = pval_data[month]
        styles = []
        for col in s.index:
            # Find original column name for this renamed column
            orig_col = None
            for orig, renamed in final_column_names.items():
                if renamed == col:
                    orig_col = orig
                    break
            if orig_col and orig_col in metric_cols:
                styles.append(f"color: {_colour(s[col], pvals.get(orig_col, np.nan))}")
            else:
                styles.append("")
        return styles

    def bold_diff_rows(s: pd.Series):
        _, strat = s.name
        return ["font-weight: bold" if strat == "Diff" else ""] * len(s)

    def format_diff_rows(s: pd.Series):
        """Custom formatter to hide values in Diff rows for specific columns"""
        month, strat = s.name
        if strat != "Diff":
            return s
        
        # Hide values for these columns in Diff rows
        hide_cols = [accts_renamed, accts_pct_renamed, bal_renamed, bal_pct_renamed]
        formatted = s.copy()
        for col in hide_cols:
            if col in formatted.index:
                formatted[col] = ""
        return formatted

    # Get unique months to calculate row positions for styling
    months = wide.index.get_level_values(0).unique()
    
    # Create CSS for borders after every 3 rows (each month group)
    border_styles = []
    for i, month in enumerate(months):
        # Each month has 3 rows (Control, Test, Diff), so border after row 3, 6, 9, etc.
        row_num = (i + 1) * 3
        border_styles.append({
            "selector": f"tbody tr:nth-child({row_num})",
            "props": [("border-bottom", "2px solid #999")]
        })

    # Create the styler with proper CSS selectors
    styler = (
        wide.style
        .format(fmt)
        .apply(colour_row, axis=1)
        .apply(bold_diff_rows, axis=1)
        .set_table_styles([
            # Thick border under header
            {"selector": "thead tr", "props": [("border-bottom", "3px solid black")]},
        ] + border_styles)
        .set_caption(f"{control_label} vs {test_label} – Monthly Summary")
    )
    
    # Apply custom formatting to hide values in Diff rows for specific columns
    def hide_diff_values(val, row_name):
        month, strat = row_name
        col_name = val.name if hasattr(val, 'name') else None
        if strat == "Diff" and col_name in [accts_renamed, accts_pct_renamed, bal_renamed, bal_pct_renamed]:
            return ""
        return val

    # Apply the hiding function using applymap with a custom function
    def apply_hide_function(df_subset):
        result = df_subset.copy()
        for idx in df_subset.index:
            if idx[1] == "Diff":  # If this is a Diff row
                for col in [accts_renamed, accts_pct_renamed, bal_renamed, bal_pct_renamed]:
                    if col in result.columns:
                        result.loc[idx, col] = ""
        return result

    # Override the format for specific cells
    for col in [accts_renamed, accts_pct_renamed, bal_renamed, bal_pct_renamed]:
        if col in wide.columns:
            styler = styler.format({col: lambda x, col=col: "" if pd.isna(x) else (fmt.get(col, "{}")
                                   .format(x) if col not in [accts_pct_renamed, bal_pct_renamed] 
                                   else f"{x:.1%}")}, subset=(slice(None), "Diff"))

    return wide, styler