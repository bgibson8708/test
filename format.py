import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def perform_monthly_ttests(df):
    """
    Perform Welch's t-test for each month comparing test vs control strategies
    
    Parameters:
    df: pandas DataFrame with the required columns
    
    Returns:
    styled pandas DataFrame with p-values
    """
    
    # Filter out NA/None strategies
    df_filtered = df[df['strategy'].notna() & (df['strategy'] != 'None')].copy()
    
    # Get unique months
    months = sorted(df_filtered['cycle_start_year_month'].unique())
    
    # Initialize results dictionary
    results = {
        'Month': [],
        'Balance': [],
        'Beacon Score': [],
        'ICS3 Score': []
    }
    
    # Perform t-tests for each month
    for month in months:
        month_data = df_filtered[df_filtered['cycle_start_year_month'] == month]
        
        # Get test and control data
        test_data = month_data[month_data['strategy'] == 'test']
        control_data = month_data[month_data['strategy'] == 'control']
        
        # Skip if either group is missing
        if len(test_data) == 0 or len(control_data) == 0:
            continue
            
        results['Month'].append(month)
        
        # Balance t-test
        if len(test_data) == 1 and len(control_data) == 1:
            # For single observations per group, use the provided stats
            bal_t, bal_p = welch_ttest_from_stats(
                test_data['bal_dollars_avg'].iloc[0],
                test_data['bal_dollars_sd'].iloc[0],
                test_data['accts'].iloc[0],
                control_data['bal_dollars_avg'].iloc[0],
                control_data['bal_dollars_sd'].iloc[0],
                control_data['accts'].iloc[0]
            )
        else:
            # If multiple rows per strategy, aggregate first
            bal_t, bal_p = aggregate_and_test(test_data, control_data, 'bal_dollars')
            
        results['Balance'].append(bal_p)
        
        # Beacon Score t-test
        if len(test_data) == 1 and len(control_data) == 1:
            beacon_t, beacon_p = welch_ttest_from_stats(
                test_data['beacon_scr_avg'].iloc[0],
                test_data['beacon_scr_sd'].iloc[0],
                test_data['accts'].iloc[0],
                control_data['beacon_scr_avg'].iloc[0],
                control_data['beacon_scr_sd'].iloc[0],
                control_data['accts'].iloc[0]
            )
        else:
            beacon_t, beacon_p = aggregate_and_test(test_data, control_data, 'beacon_scr')
            
        results['Beacon Score'].append(beacon_p)
        
        # ICS3 Score t-test
        if len(test_data) == 1 and len(control_data) == 1:
            ics3_t, ics3_p = welch_ttest_from_stats(
                test_data['ics3_score_avg'].iloc[0],
                test_data['ics3_score_sd'].iloc[0],
                test_data['accts'].iloc[0],
                control_data['ics3_score_avg'].iloc[0],
                control_data['ics3_score_sd'].iloc[0],
                control_data['accts'].iloc[0]
            )
        else:
            ics3_t, ics3_p = aggregate_and_test(test_data, control_data, 'ics3_score')
            
        results['ICS3 Score'].append(ics3_p)
    
    # Add Total row - aggregate all months
    all_test_data = df_filtered[df_filtered['strategy'] == 'test']
    all_control_data = df_filtered[df_filtered['strategy'] == 'control']
    
    if len(all_test_data) > 0 and len(all_control_data) > 0:
        results['Month'].append('Total')
        
        # Total Balance t-test
        total_bal_t, total_bal_p = aggregate_and_test(all_test_data, all_control_data, 'bal_dollars')
        results['Balance'].append(total_bal_p)
        
        # Total Beacon Score t-test
        total_beacon_t, total_beacon_p = aggregate_and_test(all_test_data, all_control_data, 'beacon_scr')
        results['Beacon Score'].append(total_beacon_p)
        
        # Total ICS3 Score t-test
        total_ics3_t, total_ics3_p = aggregate_and_test(all_test_data, all_control_data, 'ics3_score')
        results['ICS3 Score'].append(total_ics3_p)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply styling
    styled_df = style_pvalues(results_df)
    
    return results_df, styled_df

def welch_ttest_from_stats(mean1, sd1, n1, mean2, sd2, n2):
    """
    Perform Welch's t-test given summary statistics
    
    Returns:
    t_statistic, p_value
    """
    # Calculate standard errors
    se1 = sd1 / np.sqrt(n1)
    se2 = sd2 / np.sqrt(n2)
    
    # Calculate t-statistic
    t_stat = (mean1 - mean2) / np.sqrt(se1**2 + se2**2)
    
    # Calculate degrees of freedom (Welch-Satterthwaite)
    df = (se1**2 + se2**2)**2 / ((se1**2)**2/(n1-1) + (se2**2)**2/(n2-1))
    
    # Calculate two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return t_stat, p_value

def aggregate_and_test(test_data, control_data, metric_prefix):
    """
    Aggregate multiple observations and perform t-test
    """
    # Aggregate test data
    test_n_total = test_data['accts'].sum()
    test_mean = np.average(test_data[f'{metric_prefix}_avg'], weights=test_data['accts'])
    
    # Calculate pooled variance across all test observations
    test_var_components = []
    test_weights = []
    for idx, row in test_data.iterrows():
        n = row['accts']
        var = row[f'{metric_prefix}_sd']**2
        mean_i = row[f'{metric_prefix}_avg']
        # Within-group variance plus between-group variance
        test_var_components.append(var + (mean_i - test_mean)**2)
        test_weights.append(n)
    
    test_var = np.average(test_var_components, weights=test_weights)
    test_sd = np.sqrt(test_var)
    
    # Aggregate control data
    control_n_total = control_data['accts'].sum()
    control_mean = np.average(control_data[f'{metric_prefix}_avg'], weights=control_data['accts'])
    
    # Calculate pooled variance across all control observations
    control_var_components = []
    control_weights = []
    for idx, row in control_data.iterrows():
        n = row['accts']
        var = row[f'{metric_prefix}_sd']**2
        mean_i = row[f'{metric_prefix}_avg']
        # Within-group variance plus between-group variance
        control_var_components.append(var + (mean_i - control_mean)**2)
        control_weights.append(n)
    
    control_var = np.average(control_var_components, weights=control_weights)
    control_sd = np.sqrt(control_var)
    
    return welch_ttest_from_stats(test_mean, test_sd, test_n_total, 
                                  control_mean, control_sd, control_n_total)

def style_pvalues(df):
    """
    Style the dataframe to show p-values < 0.05 in red
    """
    def color_pvalues(val):
        if isinstance(val, (int, float)):
            if val < 0.05:
                return 'color: red'
            else:
                return 'color: black'
        return ''
    
    # Apply styling to p-value columns
    styled = df.style.applymap(color_pvalues, subset=['Balance', 'Beacon Score', 'ICS3 Score'])
    
    # Format p-values to 4 decimal places
    styled = styled.format({
        'Balance': '{:.4f}',
        'Beacon Score': '{:.4f}',
        'ICS3 Score': '{:.4f}'
    })
    
    # Add a separator line before Total row
    def highlight_total_row(row):
        if row['Month'] == 'Total':
            return ['border-top: 2px solid black'] * len(row)
        else:
            return [''] * len(row)
    
    styled = styled.apply(highlight_total_row, axis=1)
    
    return styled

# Example usage:
# results_df, styled_results = perform_monthly_ttests(df)
# 
# To display in Jupyter notebook:
# styled_results
# 
# To save as HTML:
# styled_results.to_html('ttest_results.html')
# 
# To get the raw dataframe:
# print(results_df)

# If you want to run this directly:
if __name__ == "__main__":
    # Assuming your dataframe 'df' is already loaded
    results_df, styled_results = perform_monthly_ttests(df)
    
    # Display results
    print("\nT-Test Results (P-Values):")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # If in Jupyter, display styled version
    try:
        from IPython.display import display
        display(styled_results)
    except:
        pass
