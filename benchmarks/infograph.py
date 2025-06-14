# %%
"""
I vibe coded this script because I couldn't be bothered. Shows nice infographics on why LMP is faster in operator benchmarks for CUDA
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# %%
# Configuration and Setup
plt.style.use('default')
sns.set_palette("husl")
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def categorize_operation(op):
    """Categorize operations into logical groups"""
    categories = {
        'Binary': ['add', 'sub', 'mul', 'div', 'pow'],
        'Unary': ['abs', 'clamp', 'cos', 'exp', 'log', 'neg', 'sin', 'sqrt', 'tan'],
        'Reduction': ['sum', 'min', 'max', 'prod']
    }
    
    for category, ops in categories.items():
        if op in ops:
            return category
    return 'Other'

# %%
# Data Parsing Functions
def parse_lmp_benchmarks(filepath):
    """Parse LMP benchmark CSV into structured format"""
    df = pd.read_csv(filepath)
    
    def extract_info(name):
        """Extract operation details from benchmark name"""
        name_clean = name.split('/')[0]
        parts = name_clean.split('_')
        operation = parts[0].lower()
        
        # Device detection
        device = 'CUDA' if 'CUDA' in name else 'CPU'
        
        # Forward/Backward detection
        forward_backward = 'Backward' if 'Backward' in name else 'Forward'
        
        # Dimension extraction
        if operation in ['add', 'sub', 'mul', 'div', 'pow']:
            if '1x64x1_64x1x64' in name:
                dimensions = 'broadcast'
            else:
                match = re.search(r'(\d+)x(\d+)_\d+x\d+', name)
                dimensions = f"{match.group(1)}x{match.group(2)}" if match else 'unknown'
        elif operation in ['sum', 'min', 'max', 'prod']:
            axis_dims = {
                'axis0_64x32': '64x32', 'axis1_64x32': '64x32',
                'axis0_64x512': '64x512', 'axis1_64x512': '64x512',
                'axis0_256x32': '256x32', 'axis1_256x32': '256x32',
                'axis0_256x512': '256x512', 'axis1_256x512': '256x512'
            }
            dimensions = next((dim for key, dim in axis_dims.items() if key in name), 'unknown')
        else:
            match = re.search(r'(\d+)x(\d+)', name)
            dimensions = f"{match.group(1)}x{match.group(2)}" if match else 'unknown'
        
        return operation, device, forward_backward, dimensions
    
    parsed_data = []
    for _, row in df.iterrows():
        operation, device, forward_backward, dimensions = extract_info(row['name'])
        parsed_data.append({
            'framework': 'LMP',
            'operation': operation,
            'device': device,
            'dimensions': dimensions,
            'forward_backward': forward_backward,
            'execution_time_us': row['real_time'],
            'execution_time_ms': row['real_time'] / 1000
        })
    
    return pd.DataFrame(parsed_data)

def parse_torch_benchmarks(filepath):
    """Parse PyTorch benchmark CSV into structured format"""
    df = pd.read_csv(filepath)
    
    def extract_torch_info(case_name, operation):
        """Extract operation details from PyTorch case name"""
        device = 'CUDA' if 'cuda' in case_name else 'CPU'
        
        if operation in ['add', 'sub', 'mul', 'div', 'pow']:
            if '[64,1,64]' in case_name and '[1,64,1]' in case_name:
                dimensions = 'broadcast'
            else:
                matches = re.findall(r'\[(\d+),(\d+)\]', case_name)
                dimensions = f"{matches[0][0]}x{matches[0][1]}" if matches else 'unknown'
        elif operation in ['sum', 'min', 'max', 'prod']:
            reduction_dims = {
                'R64_V32': '64x32', 'R64_V512': '64x512',
                'R256_V32': '256x32', 'R256_V512': '256x512'
            }
            dimensions = next((dim for key, dim in reduction_dims.items() if key in case_name), 'unknown')
        else:
            match = re.search(r'\[(\d+),(\d+)\]', case_name)
            dimensions = f"{match.group(1)}x{match.group(2)}" if match else 'unknown'
        
        return device, dimensions
    
    parsed_data = []
    for _, row in df.iterrows():
        device, dimensions = extract_torch_info(row['Case Name'], row['Benchamrking Module Name'])
        parsed_data.append({
            'framework': 'PyTorch',
            'operation': row['Benchamrking Module Name'],
            'device': device,
            'dimensions': dimensions,
            'forward_backward': 'Backward' if row['run_backward'] else 'Forward',
            'execution_time_us': row['Execution Time'],
            'execution_time_ms': row['Execution Time'] / 1000
        })
    
    return pd.DataFrame(parsed_data)

# %%
# Data Loading and Processing
def load_and_process_data():
    """Load and combine benchmark data from both frameworks"""
    torch_df = parse_torch_benchmarks('output/bench_torch.csv')
    lmp_df = parse_lmp_benchmarks('output/bench_lmp.csv')
    
    combined_df = pd.concat([torch_df, lmp_df], ignore_index=True)
    combined_df['operation_category'] = combined_df['operation'].apply(categorize_operation)
    
    # Save processed data
    combined_df.to_csv('output/processed_bench.csv', index=False)
    return combined_df

# Load the data
df = load_and_process_data()

# %%
# Visualization Helper Functions
def create_subplot_grid(title, figsize=(16, 12)):
    """Create a 2x2 subplot grid with common styling"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    return fig, axes

def style_axis(ax, title, xlabel, ylabel, log_scale=True):
    """Apply common styling to an axis"""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Framework')

def calculate_speedup(group):
    """Calculate speedup (PyTorch time / LMP time) for a group"""
    pytorch_data = group[group['framework'] == 'PyTorch']['execution_time_ms']
    lmp_data = group[group['framework'] == 'LMP']['execution_time_ms']
    
    if len(pytorch_data) > 0 and len(lmp_data) > 0:
        pytorch_time = pytorch_data.iloc[0]
        lmp_time = lmp_data.iloc[0]
        if lmp_time > 0:
            return pytorch_time / lmp_time
    return np.nan

# %%
# Performance by Tensor Size Visualization
def plot_performance_by_size(data):
    """Plot performance comparison for different tensor sizes"""
    square_dims = ['128x128', '256x256', '512x512', '1024x1024']
    square_data = data[data['dimensions'].isin(square_dims)]
    
    if square_data.empty:
        print("No data available for tensor size comparison")
        return
    
    fig, axes = create_subplot_grid('Performance Comparison by Tensor Size')
    
    devices = ['CPU', 'CUDA']
    passes = ['Forward', 'Backward']
    
    for i, device in enumerate(devices):
        for j, pass_type in enumerate(passes):
            subset = square_data[
                (square_data['device'] == device) & 
                (square_data['forward_backward'] == pass_type)
            ]
            
            if not subset.empty:
                perf_by_size = subset.groupby(['framework', 'dimensions'])['execution_time_ms'].mean().reset_index()
                pivot_data = perf_by_size.pivot(index='dimensions', columns='framework', values='execution_time_ms')
                pivot_data = pivot_data.fillna(0).reindex(square_dims)
                
                pivot_data.plot(kind='bar', ax=axes[i, j], color=COLORS[:2])
                style_axis(axes[i, j], f'{device} - {pass_type} Pass', 
                          'Tensor Dimensions', 'Average Execution Time (ms)')
            else:
                axes[i, j].set_title(f'{device} - {pass_type} Pass (No Data)')
    
    plt.tight_layout()
    plt.show()

plot_performance_by_size(df)

# %%
# Speedup Analysis Visualization
def plot_speedup_analysis(data):
    """Plot speedup analysis comparing PyTorch to LMP"""
    speedup_data = data.groupby(['operation', 'device', 'forward_backward', 'dimensions']).apply(calculate_speedup).reset_index()
    speedup_data.columns = list(speedup_data.columns[:-1]) + ['speedup']
    speedup_data = speedup_data.dropna()
    
    if speedup_data.empty:
        print("No speedup data available")
        return speedup_data
    
    plt.figure(figsize=(12, 8))
    avg_speedup = speedup_data.groupby(['device', 'forward_backward'])['speedup'].mean().reset_index()
    pivot_speedup = avg_speedup.pivot(index='device', columns='forward_backward', values='speedup')
    
    ax = pivot_speedup.plot(kind='bar', figsize=(12, 8), color=COLORS[2:4])
    plt.title('Average Speedup (PyTorch/LMP) by Device and Pass Type', fontsize=16, fontweight='bold')
    plt.xlabel('Device', fontsize=12)
    plt.ylabel('Speedup Factor', fontsize=12)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='No speedup')
    plt.legend(title='Pass Type')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return speedup_data

speedup_data = plot_speedup_analysis(df)

# %%
# Operation-specific Performance Visualization
def plot_operation_performance(data):
    """Plot performance comparison by operation type"""
    # Get operations that have data for both frameworks
    operations_with_both = data.groupby('operation').filter(
        lambda x: len(x['framework'].unique()) > 1
    )['operation'].unique()
    
    # Group and order operations by category
    op_categories = {'Binary': [], 'Unary': [], 'Reduction': [], 'Other': []}
    for op in operations_with_both:
        category = categorize_operation(op)
        op_categories[category].append(op)
    
    # Sort operations within each category
    for category in op_categories:
        op_categories[category].sort()
    
    # Create ordered list
    ordered_operations = []
    category_boundaries = {}
    current_pos = 0
    
    for category in ['Binary', 'Unary', 'Reduction', 'Other']:
        if op_categories[category]:
            category_boundaries[category] = (current_pos, current_pos + len(op_categories[category]) - 1)
            ordered_operations.extend(op_categories[category])
            current_pos += len(op_categories[category])
    
    # Filter and plot data
    ops_data = data[data['operation'].isin(ordered_operations)]
    
    if ops_data.empty:
        print("No operation-specific data available")
        return op_categories
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    fig.suptitle('Performance Comparison by Operation (Grouped by Type)', fontsize=16, fontweight='bold')
    
    # Plot for each device
    for idx, device in enumerate(['CPU', 'CUDA']):
        device_data = ops_data[ops_data['device'] == device]
        
        if not device_data.empty:
            perf_data = device_data.groupby(['framework', 'operation'])['execution_time_ms'].mean().reset_index()
            pivot_data = perf_data.pivot(index='operation', columns='framework', values='execution_time_ms')
            pivot_data = pivot_data.fillna(0)
            pivot_data = pivot_data.reindex([op for op in ordered_operations if op in pivot_data.index])
            
            pivot_data.plot(kind='bar', ax=axes[idx], color=COLORS[:2])
            style_axis(axes[idx], f'{device} Performance by Operation', 
                      'Operation', 'Average Execution Time (ms)')
            
            # Add category separators and labels
            x_pos = 0
            for category in ['Binary', 'Unary', 'Reduction', 'Other']:
                if category in category_boundaries:
                    if x_pos > 0:
                        axes[idx].axvline(x=x_pos - 0.5, color='red', linestyle='--', alpha=0.7)
                    
                    start, end = category_boundaries[category]
                    mid_pos = x_pos + (end - start) / 2
                    axes[idx].text(mid_pos, axes[idx].get_ylim()[1] * 0.8, category, 
                                  ha='center', va='bottom', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
                    x_pos += end - start + 1
    
    plt.tight_layout()
    plt.show()
    
    return op_categories

op_categories = plot_operation_performance(df)

# %%
# Summary Statistics and Reports
def print_summary_statistics(data, speedup_data):
    """Print comprehensive summary statistics"""
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    summary_stats = data.groupby(['framework', 'device']).agg({
        'execution_time_ms': ['mean', 'median', 'std', 'min', 'max', 'count']
    }).round(4)
    print(summary_stats)
    
    # Operations by category
    print("\n" + "=" * 60)
    print("OPERATIONS BY CATEGORY")
    print("=" * 60)
    for category in ['Binary', 'Unary', 'Reduction', 'Other']:
        if op_categories.get(category):
            ops = ', '.join(op_categories[category])
            print(f"\n{category} Operations ({len(op_categories[category])}): {ops}")
    
    # Speedup analysis
    if speedup_data is not None and not speedup_data.empty:
        print("\n" + "=" * 50)
        print("SPEEDUP ANALYSIS (PyTorch time / LMP time)")
        print("=" * 50)
        
        speedup_stats = speedup_data.groupby(['device'])['speedup'].agg(['mean', 'median', 'std', 'count']).round(2)
        print(speedup_stats)
        
        print("\n" + "=" * 50)
        print("EXTREME CASES")
        print("=" * 50)
        
        print("Top 5 scenarios where LMP is fastest compared to PyTorch:")
        top_speedups = speedup_data.nlargest(5, 'speedup')[['operation', 'device', 'forward_backward', 'speedup']]
        print(top_speedups.to_string(index=False))
        
        print("\nTop 5 scenarios where PyTorch is fastest compared to LMP:")
        worst_speedups = speedup_data.nsmallest(5, 'speedup')[['operation', 'device', 'forward_backward', 'speedup']]
        print(worst_speedups.to_string(index=False))

print_summary_statistics(df, speedup_data)

# %%

