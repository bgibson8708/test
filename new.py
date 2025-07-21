import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def format_rollrate_table(df, strategy_name="No LM Test", debug=False):
    """
    Format a roll rate dataframe into an HTML table with color-coded differences.
    
    Parameters:
    df: pandas DataFrame with columns as described
    strategy_name: string name of the test strategy (default: "No LM Test")
    debug: boolean to print debug information (default: False)
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
    
    # Create HTML table directly without intermediate DataFrame
    html = '<table style="border-collapse: collapse; border-spacing: 0; font-family: Arial, sans-serif;">\n'
    
    # Create header row with renamed columns
    html += '<thead>\n<tr style="background-color: #f0f0f0; border-bottom: 3px solid black;">\n'
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
            header = f'2 to {num} 
    
    # Process each month
    for month in sorted(df_filtered['cycle_start_year_month'].unique()):
        month_data = df_filtered[df_filtered['cycle_start_year_month'] == month]
        
        # Ensure Control comes before Test
        control_data = month_data[month_data['strategy'] == 'Control']
        test_data = month_data[month_data['strategy'] == strategy_name]
        
        if len(control_data) > 0 and len(test_data) > 0:
            control_row = control_data.iloc[0]
            test_row = test_data.iloc[0]
            
            # Add Control row
            html += '<tr>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{control_row["cycle_start_year_month"]}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{control_row["strategy"]}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{int(control_row["accts"]):,}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{control_row["accts_percent"]*100:.2f}%</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{int(control_row["bal_dollars"]):,}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{control_row["bal_percent"]*100:.2f}%</td>\n'
            
            for col in rr_cols:
                if pd.isna(control_row[col]):
                    html += f'<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
                else:
                    html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{control_row[col]*100:.2f}%</td>\n'
            html += '</tr>\n'
            
            # Add Test row
            html += '<tr>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{test_row["cycle_start_year_month"]}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{test_row["strategy"]}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{int(test_row["accts"]):,}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{test_row["accts_percent"]*100:.2f}%</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{int(test_row["bal_dollars"]):,}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{test_row["bal_percent"]*100:.2f}%</td>\n'
            
            for col in rr_cols:
                if pd.isna(test_row[col]):
                    html += f'<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
                else:
                    html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{test_row[col]*100:.2f}%</td>\n'
            html += '</tr>\n'
            
            # Add Difference row
            html += '<tr style="background-color: #f9f9f9; font-weight: bold; border-bottom: 2px solid black;">\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{month}</td>\n'
            html += f'<td colspan="5" style="border: 1px solid #ddd; padding: 8px; text-align: center;">Difference (Control - Test)</td>\n'
            
            # Calculate differences for each rr column
            for col in rr_cols:
                control_val = control_row[col]
                test_val = test_row[col]
                
                # Check if either value is NULL/NaN
                if pd.isna(control_val) or pd.isna(test_val):
                    html += f'<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
                    continue
                
                diff = control_val - test_val
                
                # For proportional z-test, we need the number of accounts
                if col.endswith('_acct'):
                    # Use account-based proportions
                    n_control = int(control_row['accts'])
                    n_test = int(test_row['accts'])
                else:  # ends with '_dol'
                    # Use balance-based proportions (weighted by dollars)
                    n_control = int(control_row['bal_dollars'])
                    n_test = int(test_row['bal_dollars'])
                
                # Perform proportional z-test
                color = 'black'  # default
                pval = 1.0  # default
                
                if n_control > 0 and n_test > 0 and control_val > 0 and test_val > 0:
                    try:
                        # Convert proportions to successes
                        successes_control = int(control_val * n_control)
                        successes_test = int(test_val * n_test)
                        
                        # Calculate pooled proportion
                        p_pool = (successes_control + successes_test) / (n_control + n_test)
                        
                        # Calculate standard error
                        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_test))
                        
                        # Calculate z-statistic
                        if se > 0:
                            z_stat = (control_val - test_val) / se
                            # Two-tailed p-value
                            pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                            
                            if debug:
                                print(f"\n{col}: Control={control_val:.4f}, Test={test_val:.4f}, Diff={diff:.4f}")
                                print(f"  n_control={n_control}, n_test={n_test}")
                                print(f"  z_stat={z_stat:.4f}, p_value={pval:.4f}")
                        
                    except Exception as e:
                        if debug:
                            print(f"Error in {col}: {e}")
                        pval = 1.0
                
                # Determine color based on p-value and direction
                if diff < 0:  # Negative is good (Control < Test means Test is worse at rolling)
                    if pval < 0.05:
                        color = 'green'
                    elif pval < 0.10:
                        color = 'blue'
                    else:
                        color = 'black'
                else:  # Positive is bad (Control > Test means Test is better at rolling)
                    if pval < 0.05:
                        color = 'red'
                    elif pval < 0.10:
                        color = 'orange'
                    else:
                        color = 'black'
                
                if debug:
                    print(f"  Color assigned: {color}")
                
                # Add the cell with color
                html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right; color: {color};">{diff*100:.2f}%</td>\n'
            
            html += '</tr>\n'
    
    html += '</tbody>\n</table>'
    
    return html

# Create sample data with more significant differences
def create_sample_data_with_differences():
    """Create sample data with significant differences for testing"""
    data = {
        'cycle_start_year_month': ['202410', '202410', '202410', '202411', '202411', '202411'],
        'strategy': ['Control', 'No LM Test', None, 'Control', 'No LM Test', None],
        'accts': [1000, 1200, 300, 1100, 1300, 250],
        'bal_dollars': [500000, 600000, 150000, 550000, 650000, 125000],
        # Making differences more significant
        'rr2_3_dol': [0.15, 0.12, 0.20, 0.14, 0.10, 0.19],  # Control worse (good)
        'rr2_3_acct': [0.12, 0.09, 0.18, 0.11, 0.08, 0.17],  # Control worse (good)
        'rr2_4_dol': [0.25, 0.30, 0.30, 0.24, 0.29, 0.29],  # Control better (bad)
        'rr2_4_acct': [0.22, 0.27, 0.28, 0.21, 0.26, 0.27],  # Control better (bad)
        'rr2_5_dol': [0.35, 0.35, 0.40, 0.34, 0.34, 0.39],  # No difference
        'rr2_5_acct': [0.32, 0.32, 0.38, 0.31, 0.31, 0.37],  # No difference
        'rr2_6_dol': [0.45, 0.40, 0.50, 0.44, 0.39, 0.49],  # Control worse (good)
        'rr2_6_acct': [0.42, 0.37, 0.48, 0.41, 0.36, 0.47],  # Control worse (good)
        'rr2_7_dol': [0.55, 0.65, 0.60, None, None, None],  # Control better (bad)
        'rr2_7_acct': [0.52, 0.62, 0.58, None, None, None],  # Control better (bad)
        'rr2_8_dol': [0.65, 0.65, 0.70, None, None, None],  # No difference
        'rr2_8_acct': [0.62, 0.62, 0.68, None, None, None]  # No difference
    }
    return pd.DataFrame(data)

# Test the function with debug output
def test_with_debug_output():
    """Test function with debug output to see p-values"""
    df = create_sample_data_with_differences()
    print("Testing with debug output to see p-values and color assignments:")
    print("=" * 80)
    
    html = format_rollrate_table(df, debug=True)
    
    # Add a complete HTML wrapper for standalone testing
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Roll Rate Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-spacing: 0; }}
        </style>
    </head>
    <body>
        <h2>Roll Rate Analysis Report</h2>
        <p>Color Legend:</p>
        <ul>
            <li style="color: green; font-weight: bold;">Green: Control has lower roll rate (good), p &lt; 0.05</li>
            <li style="color: blue; font-weight: bold;">Blue: Control has lower roll rate (good), p &lt; 0.10</li>
            <li style="color: red; font-weight: bold;">Red: Test has lower roll rate (bad), p &lt; 0.05</li>
            <li style="color: orange; font-weight: bold;">Orange: Test has lower roll rate (bad), p &lt; 0.10</li>
            <li style="color: black; font-weight: bold;">Black: Not statistically significant</li>
        </ul>
        {html}
    </body>
    </html>
    """
    
    with open('rollrate_test_debug.html', 'w') as f:
        f.write(full_html)
    
    print("\n" + "=" * 80)
    print("Test HTML file created as 'rollrate_test_debug.html'")
    print("Check the console output above to see the p-values and color assignments.")
    
    return html

# To use with your actual data:
# df = pd.read_csv('your_file.csv')
# html_table = format_rollrate_table(df, strategy_name="Your Test Name", debug=True)
# 
# # Save to file
# with open('rollrate_table.html', 'w') as f:
#     f.write(html_table)
        else:  # ends with '_acct'
            num = col.split('_')[1]
            header = f'2 to {num} #'
        html += f'<th style="border: 1px solid #ddd; padding: 8px;">{header}</th>\n'
    
    html += '</tr>\n</thead>\n<tbody>\n'
    
    # Process each month
    for month in sorted(df_filtered['cycle_start_year_month'].unique()):
        month_data = df_filtered[df_filtered['cycle_start_year_month'] == month]
        
        # Ensure Control comes before Test
        control_data = month_data[month_data['strategy'] == 'Control']
        test_data = month_data[month_data['strategy'] == strategy_name]
        
        if len(control_data) > 0 and len(test_data) > 0:
            control_row = control_data.iloc[0]
            test_row = test_data.iloc[0]
            
            # Add Control row
            html += '<tr>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{control_row["cycle_start_year_month"]}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{control_row["strategy"]}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{int(control_row["accts"]):,}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{control_row["accts_percent"]*100:.2f}%</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{int(control_row["bal_dollars"]):,}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{control_row["bal_percent"]*100:.2f}%</td>\n'
            
            for col in rr_cols:
                if pd.isna(control_row[col]):
                    html += f'<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
                else:
                    html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{control_row[col]*100:.2f}%</td>\n'
            html += '</tr>\n'
            
            # Add Test row
            html += '<tr>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{test_row["cycle_start_year_month"]}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{test_row["strategy"]}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{int(test_row["accts"]):,}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{test_row["accts_percent"]*100:.2f}%</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{int(test_row["bal_dollars"]):,}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{test_row["bal_percent"]*100:.2f}%</td>\n'
            
            for col in rr_cols:
                if pd.isna(test_row[col]):
                    html += f'<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
                else:
                    html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{test_row[col]*100:.2f}%</td>\n'
            html += '</tr>\n'
            
            # Add Difference row
            html += '<tr style="background-color: #f9f9f9; font-weight: bold;">\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{month}</td>\n'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">Difference (Control - Test)</td>\n'
            html += '<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'  # Empty accts
            html += '<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'  # Empty accts_percent
            html += '<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'  # Empty bal_dollars
            html += '<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'  # Empty bal_percent
            
            # Calculate differences for each rr column
            for col in rr_cols:
                control_val = control_row[col]
                test_val = test_row[col]
                
                # Check if either value is NULL/NaN
                if pd.isna(control_val) or pd.isna(test_val):
                    html += f'<td style="border: 1px solid #ddd; padding: 8px;"></td>\n'
                    continue
                
                diff = control_val - test_val
                
                # For proportional z-test, we need the number of accounts
                if col.endswith('_acct'):
                    # Use account-based proportions
                    n_control = int(control_row['accts'])
                    n_test = int(test_row['accts'])
                else:  # ends with '_dol'
                    # Use balance-based proportions (weighted by dollars)
                    n_control = int(control_row['bal_dollars'])
                    n_test = int(test_row['bal_dollars'])
                
                # Perform proportional z-test
                color = 'black'  # default
                pval = 1.0  # default
                
                if n_control > 0 and n_test > 0 and control_val > 0 and test_val > 0:
                    try:
                        # Convert proportions to successes
                        successes_control = int(control_val * n_control)
                        successes_test = int(test_val * n_test)
                        
                        # Calculate pooled proportion
                        p_pool = (successes_control + successes_test) / (n_control + n_test)
                        
                        # Calculate standard error
                        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_test))
                        
                        # Calculate z-statistic
                        if se > 0:
                            z_stat = (control_val - test_val) / se
                            # Two-tailed p-value
                            pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                            
                            if debug:
                                print(f"\n{col}: Control={control_val:.4f}, Test={test_val:.4f}, Diff={diff:.4f}")
                                print(f"  n_control={n_control}, n_test={n_test}")
                                print(f"  z_stat={z_stat:.4f}, p_value={pval:.4f}")
                        
                    except Exception as e:
                        if debug:
                            print(f"Error in {col}: {e}")
                        pval = 1.0
                
                # Determine color based on p-value and direction
                if diff < 0:  # Negative is good (Control < Test means Test is worse at rolling)
                    if pval < 0.05:
                        color = 'green'
                    elif pval < 0.10:
                        color = 'blue'
                    else:
                        color = 'black'
                else:  # Positive is bad (Control > Test means Test is better at rolling)
                    if pval < 0.05:
                        color = 'red'
                    elif pval < 0.10:
                        color = 'orange'
                    else:
                        color = 'black'
                
                if debug:
                    print(f"  Color assigned: {color}")
                
                # Add the cell with color
                html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right; color: {color};">{diff*100:.2f}%</td>\n'
            
            html += '</tr>\n'
    
    html += '</tbody>\n</table>'
    
    return html

# Create sample data with more significant differences
def create_sample_data_with_differences():
    """Create sample data with significant differences for testing"""
    data = {
        'cycle_start_year_month': ['202410', '202410', '202410', '202411', '202411', '202411'],
        'strategy': ['Control', 'No LM Test', None, 'Control', 'No LM Test', None],
        'accts': [1000, 1200, 300, 1100, 1300, 250],
        'bal_dollars': [500000, 600000, 150000, 550000, 650000, 125000],
        # Making differences more significant
        'rr2_3_dol': [0.15, 0.12, 0.20, 0.14, 0.10, 0.19],  # Control worse (good)
        'rr2_3_acct': [0.12, 0.09, 0.18, 0.11, 0.08, 0.17],  # Control worse (good)
        'rr2_4_dol': [0.25, 0.30, 0.30, 0.24, 0.29, 0.29],  # Control better (bad)
        'rr2_4_acct': [0.22, 0.27, 0.28, 0.21, 0.26, 0.27],  # Control better (bad)
        'rr2_5_dol': [0.35, 0.35, 0.40, 0.34, 0.34, 0.39],  # No difference
        'rr2_5_acct': [0.32, 0.32, 0.38, 0.31, 0.31, 0.37],  # No difference
        'rr2_6_dol': [0.45, 0.40, 0.50, 0.44, 0.39, 0.49],  # Control worse (good)
        'rr2_6_acct': [0.42, 0.37, 0.48, 0.41, 0.36, 0.47],  # Control worse (good)
        'rr2_7_dol': [0.55, 0.65, 0.60, None, None, None],  # Control better (bad)
        'rr2_7_acct': [0.52, 0.62, 0.58, None, None, None],  # Control better (bad)
        'rr2_8_dol': [0.65, 0.65, 0.70, None, None, None],  # No difference
        'rr2_8_acct': [0.62, 0.62, 0.68, None, None, None]  # No difference
    }
    return pd.DataFrame(data)

# Test the function with debug output
def test_with_debug_output():
    """Test function with debug output to see p-values"""
    df = create_sample_data_with_differences()
    print("Testing with debug output to see p-values and color assignments:")
    print("=" * 80)
    
    html = format_rollrate_table(df, debug=True)
    
    # Add a complete HTML wrapper for standalone testing
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Roll Rate Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
        </style>
    </head>
    <body>
        <h2>Roll Rate Analysis Report</h2>
        <p>Color Legend:</p>
        <ul>
            <li style="color: green; font-weight: bold;">Green: Control has lower roll rate (good), p &lt; 0.05</li>
            <li style="color: blue; font-weight: bold;">Blue: Control has lower roll rate (good), p &lt; 0.10</li>
            <li style="color: red; font-weight: bold;">Red: Test has lower roll rate (bad), p &lt; 0.05</li>
            <li style="color: orange; font-weight: bold;">Orange: Test has lower roll rate (bad), p &lt; 0.10</li>
            <li style="color: black; font-weight: bold;">Black: Not statistically significant</li>
        </ul>
        {html}
    </body>
    </html>
    """
    
    with open('rollrate_test_debug.html', 'w') as f:
        f.write(full_html)
    
    print("\n" + "=" * 80)
    print("Test HTML file created as 'rollrate_test_debug.html'")
    print("Check the console output above to see the p-values and color assignments.")
    
    return html

# To use with your actual data:
# df = pd.read_csv('your_file.csv')
# html_table = format_rollrate_table(df, strategy_name="Your Test Name", debug=True)
# 
# # Save to file
# with open('rollrate_table.html', 'w') as f:
#     f.write(html_table)
