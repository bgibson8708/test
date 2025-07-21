import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def format_rollrate_table(df, strategy_name="No LM Test"):
    """
    Format a roll rate dataframe into an HTML table with color-coded differences.
    
    Parameters:
    df: pandas DataFrame with columns as described
    strategy_name: string name of the test strategy (default: "No LM Test")
    """
    
    # Create a copy to avoid modifying original
    df_work = df.copy()
    
    # Filter out NULL strategy values for main display
    df_filtered = df_work[df_work['strategy'].notna()].copy()
    
    # Calculate accts_percent and bal_percent including NULL values
    monthly_totals = df_work.groupby('cycle_start_year_month').agg({
        'accts': 'sum',
        'bal_dollars': 'sum'
    }).reset_index()
    
    # Merge totals back to filtered dataframe
    df_filtered = df_filtered.merge(
        monthly_totals, 
        on='cycle_start_year_month', 
        suffixes=('', '_total')
    )
    
    # Calculate percentages
    df_filtered['accts_percent'] = df_filtered['accts'] / df_filtered['accts_total']
    df_filtered['bal_percent'] = df_filtered['bal_dollars'] / df_filtered['bal_dollars_total']
    
    # Drop the total columns
    df_filtered = df_filtered.drop(['accts_total', 'bal_dollars_total'], axis=1)
    
    # Identify rr columns
    rr_cols = [col for col in df_filtered.columns if col.startswith('rr')]
    
    # Create formatted dataframe for display
    results = []
    
    for month in sorted(df_filtered['cycle_start_year_month'].unique()):
        month_data = df_filtered[df_filtered['cycle_start_year_month'] == month]
        
        # Ensure Control comes before Test
        control_data = month_data[month_data['strategy'] == 'Control']
        test_data = month_data[month_data['strategy'] == strategy_name]
        
        if len(control_data) > 0 and len(test_data) > 0:
            # Add Control row
            control_row = control_data.iloc[0].to_dict()
            results.append(control_row)
            
            # Add Test row
            test_row = test_data.iloc[0].to_dict()
            results.append(test_row)
            
            # Calculate differences and p-values
            diff_row = {
                'cycle_start_year_month': month,
                'strategy': 'Difference (Control - Test)',
                'accts': '',
                'accts_percent': '',
                'bal_dollars': '',
                'bal_percent': ''
            }
            
            # For each rr column, calculate difference and p-value
            for col in rr_cols:
                control_val = control_row[col]
                test_val = test_row[col]
                
                # Check if either value is NULL/NaN
                if pd.isna(control_val) or pd.isna(test_val):
                    diff_row[col] = ('', 'black')  # Empty string with black color
                    continue
                
                diff = control_val - test_val
                
                # For proportional z-test, we need the number of accounts
                if col.endswith('_acct'):
                    # Use account-based proportions
                    n_control = control_row['accts']
                    n_test = test_row['accts']
                else:  # ends with '_dol'
                    # Use balance-based proportions (weighted by dollars)
                    n_control = control_row['bal_dollars']
                    n_test = test_row['bal_dollars']
                
                # Perform proportional z-test
                if n_control > 0 and n_test > 0 and control_val >= 0 and test_val >= 0:
                    try:
                        # Convert percentages to counts for the test
                        x_control = int(control_val * n_control)
                        x_test = int(test_val * n_test)
                        
                        # Two-proportion z-test
                        count = np.array([x_control, x_test])
                        nobs = np.array([n_control, n_test])
                        
                        stat, pval = stats.proportions_ztest(count, nobs)
                        
                        # Determine color based on p-value and direction
                        if diff < 0:  # Negative is good
                            if pval < 0.05:
                                color = 'green'
                            elif pval < 0.1:
                                color = 'blue'
                            else:
                                color = 'black'
                        else:  # Positive is bad
                            if pval < 0.05:
                                color = 'red'
                            elif pval < 0.1:
                                color = 'orange'
                            else:
                                color = 'black'
                    except:
                        color = 'black'
                else:
                    color = 'black'
                
                # Store difference with color information
                diff_row[col] = (diff, color)
            
            results.append(diff_row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create HTML table
    html = '<table style="border-collapse: collapse; font-family: Arial, sans-serif;">\n'
    
    # Create header row with renamed columns
    html += '<thead>\n<tr style="background-color: #f0f0f0;">\n'
    html += '<th style="border: 1px solid #ddd; padding: 8px;">Month</th>\n'
    html += '<th style="border: 1px solid #ddd; padding: 8px;">Strategy</th>\n'
    html += '<th style="border: 1px solid #ddd; padding: 8px;">Accts</th>\n'
    html += '<th style="border: 1px solid #ddd; padding: 8px;">Acct %</th>\n'
    html += '<th style="border: 1px solid #ddd; padding: 8px;">Balance $</th>\n'
    html += '<th style="border: 1px solid #ddd; padding: 8px;">Bal %</th>\n'
    
    # Add rr column headers
    for col in rr_cols:
        if col.endswith('_dol'):
            # Extract the number (e.g., '3' from 'rr2_3_dol')
            num = col.split('_')[1]
            header = f'2 to {num} $'
        else:  # ends with '_acct'
            num = col.split('_')[1]
            header = f'2 to {num} #'
        html += f'<th style="border: 1px solid #ddd; padding: 8px;">{header}</th>\n'
    
    html += '</tr>\n</thead>\n<tbody>\n'
    
    # Add data rows
    for idx, row in results_df.iterrows():
        if 'Difference' in str(row['strategy']):
            html += '<tr style="background-color: #f9f9f9; font-weight: bold;">\n'
        else:
            html += '<tr>\n'
        
        # Month
        html += f'<td style="border: 1px solid #ddd; padding: 8px;">{row["cycle_start_year_month"]}</td>\n'
        
        # Strategy
        html += f'<td style="border: 1px solid #ddd; padding: 8px;">{row["strategy"]}</td>\n'
        
        # Accts
        if row['accts'] == '':
            html += '<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
        else:
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{int(row["accts"]):,}</td>\n'
        
        # Acct %
        if row['accts_percent'] == '':
            html += '<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
        else:
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{row["accts_percent"]*100:.2f}%</td>\n'
        
        # Balance $
        if row['bal_dollars'] == '':
            html += '<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
        else:
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{int(row["bal_dollars"]):,}</td>\n'
        
        # Bal %
        if row['bal_percent'] == '':
            html += '<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
        else:
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{row["bal_percent"]*100:.2f}%</td>\n'
        
        # RR columns
        for col in rr_cols:
            if 'Difference' in str(row['strategy']):
                # This is a difference row - handle tuple format
                if isinstance(row[col], tuple):
                    val, color = row[col]
                    if val == '':  # Handle empty values
                        html += f'<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
                    else:
                        html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right; color: {color}; font-weight: bold;">{val*100:.2f}%</td>\n'
                else:
                    # Shouldn't happen, but handle gracefully
                    html += f'<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
            else:
                # Regular row - check for NULL/NaN values
                if pd.isna(row[col]):
                    html += f'<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
                else:
                    html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{row[col]*100:.2f}%</td>\n'
        
        html += '</tr>\n'
    
    html += '</tbody>\n</table>'
    
    return html

# Example usage with NULL values:
def create_sample_data_with_nulls():
    """Create sample data with NULL values for testing"""
    data = {
        'cycle_start_year_month': ['202410', '202410', '202410', '202411', '202411', '202411'],
        'strategy': ['Control', 'No LM Test', None, 'Control', 'No LM Test', None],
        'accts': [1000, 1200, 300, 1100, 1300, 250],
        'bal_dollars': [500000, 600000, 150000, 550000, 650000, 125000],
        'rr2_3_dol': [0.15, 0.17, 0.20, 0.14, 0.16, 0.19],
        'rr2_3_acct': [0.12, 0.14, 0.18, 0.11, 0.13, 0.17],
        'rr2_4_dol': [0.25, 0.28, 0.30, 0.24, 0.27, 0.29],
        'rr2_4_acct': [0.22, 0.25, 0.28, 0.21, 0.24, 0.27],
        'rr2_5_dol': [0.35, 0.39, 0.40, 0.34, 0.38, 0.39],
        'rr2_5_acct': [0.32, 0.36, 0.38, 0.31, 0.35, 0.37],
        'rr2_6_dol': [0.45, 0.50, 0.50, 0.44, 0.49, 0.49],
        'rr2_6_acct': [0.42, 0.47, 0.48, 0.41, 0.46, 0.47],
        'rr2_7_dol': [0.55, 0.61, 0.60, None, None, None],  # NULL for newer month
        'rr2_7_acct': [0.52, 0.58, 0.58, None, None, None],  # NULL for newer month
        'rr2_8_dol': [0.65, 0.72, 0.70, None, None, None],  # NULL for newer month
        'rr2_8_acct': [0.62, 0.69, 0.68, None, None, None]  # NULL for newer month
    }
    return pd.DataFrame(data)

# To use with your actual data:
# df = pd.read_csv('your_file.csv')  # or however you load your data
# html_table = format_rollrate_table(df, strategy_name="Your Test Name")
# 
# # Save to file
# with open('rollrate_table.html', 'w') as f:
#     f.write(html_table)
# 
# # Or display in Jupyter notebook
# from IPython.display import HTML
# HTML(html_table)
