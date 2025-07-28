import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

def welch_t_test(sample_means, sample_vars, sample_ns):
    """
    Mirrors R’s:
      t_val <- diff(sample_means) / sqrt(sum(sample_vars/sample_ns))
      df    <- (sum(sample_vars/sample_ns))**2 /
               sum(sample_vars**2 / (sample_ns**2 * (sample_ns-1)))
      p_val <- 2 * pt(abs(t_val), df, lower.tail = FALSE)
    """
    sample_ns   = np.asarray(sample_ns,   dtype=float)
    sample_vars = np.asarray(sample_vars, dtype=float)
    sample_means= np.asarray(sample_means,dtype=float)

    # 1‑lag differences of the means
    deltas = np.diff(sample_means)

    # pooled variance of the difference
    var_sum = (sample_vars / sample_ns).sum()
    se      = np.sqrt(var_sum)

    # Welch–Satterthwaite degrees of freedom
    df_num = var_sum**2
    df_den = (sample_vars**2 / (sample_ns**2 * (sample_ns - 1))).sum()
    df_val = df_num / df_den

    # two‑sided p‑values
    p_vals = 2 * t_dist.sf(np.abs(deltas / se), df_val)

    return deltas / se, df_val, p_vals


# ─── create empty results frame ─────────────────────────────────────
loop_append = pd.DataFrame({
    'Vintage':     pd.Series(dtype='str'),
    'Vantage':     pd.Series(dtype='float'),
    'Balance':     pd.Series(dtype='float'),
    'ICS3_Score':  pd.Series(dtype='float')
})

# ─── loop across each month ────────────────────────────────────────
for month, grp in df.groupby('cycle_start_year_month'):
    n       = grp['accts'].astype(float)
    
    # Balance test
    bal_mean = grp['bal_dollars_avg']
    bal_var  = grp['bal_dollars_sd']**2
    _, _, p_bal = welch_t_test(bal_mean, bal_var, n)

    # ICS3 test
    i3_mean  = grp['ics3_score_avg']
    i3_var   = grp['ics3_score_sd']**2
    _, _, p_i3  = welch_t_test(i3_mean, i3_var, n)

    # Beacon test
    b_mean   = grp['beacon_scr_avg']
    b_var    = grp['beacon_scr_sd']**2
    _, _, p_b   = welch_t_test(b_mean, b_var, n)

    # assemble and append
    out = pd.DataFrame({
        'Vintage':    month,
        'Vantage':    p_b,
        'Balance':    p_bal,
        'ICS3_Score': p_i3
    })
    loop_append = pd.concat([loop_append, out], ignore_index=True)

# loop_append now matches your R loop_append
print(loop_append)
