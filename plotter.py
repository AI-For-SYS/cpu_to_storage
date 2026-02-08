import json
import matplotlib.pyplot as plt


def plot_results_threads_comparison(results_file='compare_file_operations/results/threads_comp_bs32_readinto.json'):
    """Plot write and read results separately to compare thread count impact."""
    
    # Load results from JSON file
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    write_data = results['write']
    read_data = results['read']
    config = results['config']
    
    # Calculate data size in GB
    num_blocks = config['num_blocks_to_copy']
    block_size = config['block_size']
    data_size_gb = (num_blocks * block_size) / (1024**3)
    
    # Extract metadata for subtitle
    file_system = config.get('file_system', 'Unknown')
    implementation = config.get('implementation', 'Unknown')
    block_size_mb = block_size / (1024 * 1024)
    
    # Create subtitle text
    subtitle = f"File System: {file_system} | Implementation: {implementation} (readinto)\n"
    subtitle += f"Block Size: {block_size_mb:.0f} MB | Blocks Copied: {num_blocks}"
    
    # Extract thread counts and metrics
    thread_counts = sorted([int(k) for k in write_data.keys()])
    
    # Prepare time data for plotting
    write_avg_time = [write_data[str(tc)]['avg'] for tc in thread_counts]
    write_median_time = [write_data[str(tc)]['median'] for tc in thread_counts]
    write_min_time = [write_data[str(tc)]['min'] for tc in thread_counts]
    write_max_time = [write_data[str(tc)]['max'] for tc in thread_counts]
    
    read_avg_time = [read_data[str(tc)]['avg'] for tc in thread_counts]
    read_median_time = [read_data[str(tc)]['median'] for tc in thread_counts]
    read_min_time = [read_data[str(tc)]['min'] for tc in thread_counts]
    read_max_time = [read_data[str(tc)]['max'] for tc in thread_counts]
    
    # Convert time (ms) to throughput (GB/s)
    # Throughput = data_size_gb / (time_ms / 1000)
    write_avg = [data_size_gb / t for t in write_avg_time]
    write_median = [data_size_gb / t for t in write_median_time]
    write_min = [data_size_gb / t for t in write_max_time]  # min throughput from max time
    write_max = [data_size_gb / t for t in write_min_time]  # max throughput from min time
    
    read_avg = [data_size_gb / t for t in read_avg_time]
    read_median = [data_size_gb / t for t in read_median_time]
    read_min = [data_size_gb / t for t in read_max_time]  # min throughput from max time
    read_max = [data_size_gb / t for t in read_min_time]  # max throughput from min time
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Write Results
    ax1.plot(thread_counts, write_avg, 'o-', label='Average', linewidth=2, markersize=8)
    ax1.plot(thread_counts, write_median, 's-', label='Median', linewidth=2, markersize=8)
    ax1.fill_between(thread_counts, write_min, write_max, alpha=0.2, label='Min-Max Range')
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Throughput (GB/s)', fontsize=12)
    ax1.set_title('Write Throughput vs Thread Count', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.text(0.5, -0.18, subtitle, transform=ax1.transAxes,
             fontsize=9, ha='center', va='top', style='italic', color='#555555')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(thread_counts)
    ax1.set_xticklabels(thread_counts)
    
    # Plot Read Results
    ax2.plot(thread_counts, read_avg, 'o-', label='Average', linewidth=2, markersize=8, color='orange')
    ax2.plot(thread_counts, read_median, 's-', label='Median', linewidth=2, markersize=8, color='red')
    ax2.fill_between(thread_counts, read_min, read_max, alpha=0.2, label='Min-Max Range', color='orange')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Throughput (GB/s)', fontsize=12)
    ax2.set_title('Read Throughput vs Thread Count', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.text(0.5, -0.18, subtitle, transform=ax2.transAxes,
             fontsize=9, ha='center', va='top', style='italic', color='#555555')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(thread_counts)
    ax2.set_xticklabels(thread_counts)
    
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    
    # Save plot
    output_plot = f'compare_file_operations/results/thread_performance_plot_bs{block_size_mb}_readinto.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_plot}")
    plt.close()
    
    print("Plotting complete!")


def plot_throughput_tables(results_file='compare_file_operations_results/block_size_comparison_10gb.json'):
    """Create formatted tables showing throughput for different block sizes and thread counts."""
    
    # Load results from JSON file
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    write_data = results['write']
    read_data = results['read']
    config = results['config']
    
    # Extract configuration
    thread_counts = config['thread_counts']
    block_sizes_mb = config['block_sizes_mb']
    total_data_gb = config.get('total_data_size_gb', 'Unknown')
    num_iterations = config['num_iterations']
    buffer_size_gb = config['buffer_size'] / (1024**3)
    file_system = config.get('file_system', 'Unknown')
    
    # Fixed settings comment
    fixed_settings = (f"Fixed settings: Buffer={buffer_size_gb:.0f}GB, "
                     f"Total Data={total_data_gb}GB, Iterations={num_iterations}, FS={file_system}")
    
    # Prepare output
    output_lines = []
    output_lines.append("="*100)
    output_lines.append("THROUGHPUT COMPARISON TABLES")
    output_lines.append("="*100)
    output_lines.append("")
    
    # Function to create a table
    def create_table(operation_data, operation_name):
        lines = []
        lines.append(f"\n{operation_name} Throughput (GB/s)")
        lines.append(fixed_settings)
        lines.append("")
        
        # Header
        header = "Threads |"
        for bs_mb in block_sizes_mb:
            header += f" {bs_mb:>6}MB |"
        lines.append(header)
        lines.append("-" * len(header))
        
        # Data rows
        best_throughput = 0
        best_position = None
        
        # NEW STRUCTURE: operation_data[block_size][thread_count]
        for thread_count in thread_counts:
            row = f"{thread_count:>7} |"
            thread_key = str(thread_count)
            
            for bs_mb in block_sizes_mb:
                block_size = bs_mb * 1024 * 1024
                block_key = str(block_size)
                
                # Access data with block_size as outer key
                if block_key in operation_data and thread_key in operation_data[block_key]:
                    avg_time = operation_data[block_key][thread_key]
                    throughput = total_data_gb / avg_time
                    row += f" {throughput:>7.2f} |"
                    
                    # Track best throughput
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_position = (thread_count, bs_mb)
                else:
                    row += "    N/A |"
            
            lines.append(row)
        
        lines.append("")
        if best_position:
            lines.append(f"Best throughput: {best_throughput:.2f} GB/s "
                        f"(Threads={best_position[0]}, Block Size={best_position[1]}MB)")
        
        return lines
    
    # Create write table
    output_lines.extend(create_table(write_data, "Write"))
    
    # Create read table
    output_lines.extend(create_table(read_data, "Read"))
    
    output_lines.append("")
    output_lines.append("="*100)
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Save to text file
    output_txt = results_file.replace('.json', '_tables.txt')
    with open(output_txt, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nTables saved to {output_txt}")
    
    # Create matplotlib figure with tables
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Throughput Comparison Tables', fontsize=16, fontweight='bold')
    
    # Function to create matplotlib table
    def create_mpl_table(ax, operation_data, operation_name):
        # Prepare data for table
        col_labels = [f'{bs}MB' for bs in block_sizes_mb]
        row_labels = [f'{tc} threads' for tc in thread_counts]
        
        cell_data = []
        cell_colors = []
        
        max_throughput = 0
        # NEW STRUCTURE: operation_data[block_size][thread_count]
        for thread_count in thread_counts:
            row = []
            thread_key = str(thread_count)
            
            for bs_mb in block_sizes_mb:
                block_size = bs_mb * 1024 * 1024
                block_key = str(block_size)
                
                # Access data with block_size as outer key
                if block_key in operation_data and thread_key in operation_data[block_key]:
                    avg_time = operation_data[block_key][thread_key]
                    throughput = total_data_gb / avg_time
                    row.append(f'{throughput:.2f}')
                    max_throughput = max(max_throughput, throughput)
                else:
                    row.append('N/A')
            
            cell_data.append(row)
        
        # Create color map for cells
        import matplotlib.cm as cm
        for row in cell_data:
            color_row = []
            for cell in row:
                if cell == 'N/A':
                    color_row.append('#f0f0f0')
                else:
                    # Color intensity based on throughput
                    intensity = float(cell) / max_throughput if max_throughput > 0 else 0
                    color_row.append(cm.Greens(0.3 + 0.6 * intensity))
            cell_colors.append(color_row)
        
        # Create table
        table = ax.table(cellText=cell_data, rowLabels=row_labels, colLabels=col_labels,
                        cellLoc='center', loc='center', cellColours=cell_colors)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style row labels
        for i in range(len(row_labels)):
            table[(i+1, -1)].set_facecolor('#e0e0e0')
            table[(i+1, -1)].set_text_props(weight='bold')
        
        ax.set_title(f'{operation_name} Throughput (GB/s)\n{fixed_settings}',
                    fontsize=12, pad=20)
        ax.axis('off')
    
    create_mpl_table(ax1, write_data, 'Write')
    create_mpl_table(ax2, read_data, 'Read')
    
    plt.tight_layout()
    
    # Save figure
    output_png = results_file.replace('.json', '_tables.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Table visualization saved to {output_png}")
    plt.close()


def plot_block_size_heatmaps(results_file='compare_file_operations_results/block_size_comparison_10gb.json'):
    """Create heatmap visualizations for block size comparison."""
    
    # Load results from JSON file
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    write_data = results['write']
    read_data = results['read']
    config = results['config']
    
    # Extract configuration
    thread_counts = config['thread_counts']
    block_sizes_mb = config['block_sizes_mb']
    total_data_gb = config.get('total_data_size_gb', 'Unknown')
    buffer_size_gb = config['buffer_size'] / (1024**3)
    file_system = config.get('file_system', 'Unknown')
    implementation = config.get('implementation', 'Unknown')
    
    # Subtitle
    subtitle = (f"File System: {file_system} | Implementation: {implementation}\n"
               f"Buffer: {buffer_size_gb:.0f}GB | Total Data: {total_data_gb}gB")
    
    # Function to create throughput matrix
    def create_throughput_matrix(operation_data):
        matrix = []
        # NEW STRUCTURE: operation_data[block_size][thread_count]
        for thread_count in thread_counts:
            row = []
            thread_key = str(thread_count)
            
            for bs_mb in block_sizes_mb:
                block_size = bs_mb * 1024 * 1024
                block_key = str(block_size)
                
                # Access data with block_size as outer key
                if block_key in operation_data and thread_key in operation_data[block_key]:
                    avg_time = operation_data[block_key][thread_key]
                    throughput = total_data_gb / avg_time
                    row.append(throughput)
                else:
                    row.append(0)  # Use 0 for missing data
            
            matrix.append(row)
        
        return matrix
    
    # Create matrices
    write_matrix = create_throughput_matrix(write_data)
    read_matrix = create_throughput_matrix(read_data)
    
    # Create figure with two heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Throughput Heatmaps: Block Size vs Thread Count',
                fontsize=16, fontweight='bold')
    
    # Write heatmap
    im1 = ax1.imshow(write_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(block_sizes_mb)))
    ax1.set_xticklabels([f'{bs}MB' for bs in block_sizes_mb], rotation=45, ha='right')
    ax1.set_yticks(range(len(thread_counts)))
    ax1.set_yticklabels([f'{tc}' for tc in thread_counts])
    ax1.set_xlabel('Block Size', fontsize=12)
    ax1.set_ylabel('Number of Threads', fontsize=12)
    ax1.set_title('Write Throughput (GB/s)', fontsize=14, fontweight='bold', pad=10)
    
    # Add values to cells with better visibility
    max_write = max(max(row) for row in write_matrix)
    for i in range(len(thread_counts)):
        for j in range(len(block_sizes_mb)):
            if write_matrix[i][j] > 0:
                # Use white text for darker backgrounds, black for lighter
                intensity = write_matrix[i][j] / max_write if max_write > 0 else 0
                text_color = 'white' if intensity > 0.5 else 'black'
                text = ax1.text(j, i, f'{write_matrix[i][j]:.2f}',
                              ha="center", va="center", color=text_color,
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor='none', alpha=0.3))
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Throughput (GB/s)', rotation=270, labelpad=20)
    
    # Read heatmap
    im2 = ax2.imshow(read_matrix, cmap='YlGnBu', aspect='auto')
    ax2.set_xticks(range(len(block_sizes_mb)))
    ax2.set_xticklabels([f'{bs}MB' for bs in block_sizes_mb], rotation=45, ha='right')
    ax2.set_yticks(range(len(thread_counts)))
    ax2.set_yticklabels([f'{tc}' for tc in thread_counts])
    ax2.set_xlabel('Block Size', fontsize=12)
    ax2.set_ylabel('Number of Threads', fontsize=12)
    ax2.set_title('Read Throughput (GB/s)', fontsize=14, fontweight='bold', pad=10)
    
    # Add values to cells with better visibility
    max_read = max(max(row) for row in read_matrix)
    for i in range(len(thread_counts)):
        for j in range(len(block_sizes_mb)):
            if read_matrix[i][j] > 0:
                # Use white text for darker backgrounds, black for lighter
                intensity = read_matrix[i][j] / max_read if max_read > 0 else 0
                text_color = 'white' if intensity > 0.5 else 'black'
                text = ax2.text(j, i, f'{read_matrix[i][j]:.2f}',
                              ha="center", va="center", color=text_color,
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor='none', alpha=0.3))
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Throughput (GB/s)', rotation=270, labelpad=20)
    
    # Add subtitle
    fig.text(0.5, 0.02, subtitle, ha='center', fontsize=10, style='italic', color='#555555')
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.98))
    
    # Save figure
    output_png = results_file.replace('.json', '_heatmaps.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap visualization saved to {output_png}")
    plt.close()
    
    print("Heatmap plotting complete!")


def plot_block_size_throughput_by_threads(results_file1='compare_file_operations/results/block_size_comparison_1000blocks.json',
                                          results_file2=None):
    """Create 6 plots (2 operations × 3 thread counts) showing throughput vs block size.
    
    Each plot shows throughput as a function of block size for a specific operation and thread count.
    If results_file2 is provided, compares two implementations on the same plots.
    Includes all metadata from the config.
    
    Args:
        results_file1: Path to first results JSON file
        results_file2: Optional path to second results JSON file for comparison
    """
    
    # Load results from first JSON file
    with open(results_file1, 'r') as f:
        results1 = json.load(f)
    
    write_data1 = results1['write']
    read_data1 = results1['read']
    config1 = results1['config']
    
    # Extract configuration from first file
    thread_counts = config1['thread_counts']
    block_sizes_mb = config1['block_sizes_mb']
    num_blocks = config1['num_blocks']
    num_iterations = config1['num_iterations']
    buffer_size_gb = config1['buffer_size'] / (1024**3)
    file_system = config1.get('file_system', 'Unknown')
    implementation1 = config1.get('implementation', 'Unknown')
    
    # Initialize variables for second file
    write_data2 = None
    read_data2 = None
    implementation2 = None
    
    # Load second file if provided
    if results_file2:
        with open(results_file2, 'r') as f:
            results2 = json.load(f)
        
        write_data2 = results2['write']
        read_data2 = results2['read']
        config2 = results2['config']
        implementation2 = config2.get('implementation', 'Unknown')
        
        # Create metadata text for comparison
        metadata = (f"File System: {file_system} | Comparing: {implementation1} vs {implementation2}\n"
                    f"Buffer: {buffer_size_gb:.1f}GB | Blocks: {num_blocks} | Iterations: {num_iterations}")
    else:
        # Single file metadata
        metadata = (f"File System: {file_system} | Implementation: {implementation1}\n"
                    f"Buffer: {buffer_size_gb:.1f}GB | Blocks: {num_blocks} | Iterations: {num_iterations}")
    
    # Create figure with 2 rows (operations) × 3 columns (thread counts)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Throughput vs Block Size' + (' - Implementation Comparison' if results_file2 else ''),
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add overall metadata below the title with more spacing
    fig.text(0.5, 0.92, metadata, ha='center', fontsize=10, style='italic', color='#555555')
    
    operations = [('write', write_data1, write_data2 if results_file2 else None, 'Write', 'blue', 'cyan'),
                  ('read', read_data1, read_data2 if results_file2 else None, 'Read', 'orange', 'red')]
    
    for row_idx, (op_name, op_data1, op_data2, op_label, color1, color2) in enumerate(operations):
        for col_idx, thread_count in enumerate(thread_counts):
            ax = axes[row_idx, col_idx]
            thread_key = str(thread_count)
            
            # Collect data for first implementation
            throughputs1 = []
            block_sizes_for_plot = []
            
            for bs_mb in block_sizes_mb:
                bs_key = str(bs_mb)
                # NEW STRUCTURE: op_data[thread_count][block_size]
                if thread_key in op_data1 and bs_key in op_data1[thread_key]:
                    avg_time = op_data1[thread_key][bs_key]
                    # Calculate throughput: (block_size_mb * num_blocks) / (time_seconds * 1024) = GB/s
                    total_data_gb = (bs_mb * num_blocks) / 1024
                    throughput = total_data_gb / avg_time
                    throughputs1.append(throughput)
                    block_sizes_for_plot.append(bs_mb)
            
            # Plot first implementation
            ax.plot(block_sizes_for_plot, throughputs1, 'o-', color=color1,
                   linewidth=2.5, markersize=10, markeredgewidth=1.5,
                   markeredgecolor='white', label=implementation1)
            
            # Add value labels on points for first implementation
            for bs, tp in zip(block_sizes_for_plot, throughputs1):
                ax.annotate(f'{tp:.2f}', xy=(bs, tp),
                           xytext=(0, 8), textcoords='offset points',
                           ha='center', fontsize=7, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor=color1, alpha=0.7))
            
            # Plot second implementation if provided
            if results_file2 and op_data2:
                throughputs2 = []
                block_sizes_for_plot2 = []
                
                for bs_mb in block_sizes_mb:
                    bs_key = str(bs_mb)
                    # NEW STRUCTURE: op_data[thread_count][block_size]
                    if thread_key in op_data2 and bs_key in op_data2[thread_key]:
                        avg_time = op_data2[thread_key][bs_key]
                        total_data_gb = (bs_mb * num_blocks) / 1024
                        throughput = total_data_gb / avg_time
                        throughputs2.append(throughput)
                        block_sizes_for_plot2.append(bs_mb)
                
                # Plot second implementation
                ax.plot(block_sizes_for_plot2, throughputs2, 's--', color=color2,
                       linewidth=2.5, markersize=8, markeredgewidth=1.5,
                       markeredgecolor='white', label=implementation2)
                
                # Add value labels on points for second implementation
                for bs, tp in zip(block_sizes_for_plot2, throughputs2):
                    ax.annotate(f'{tp:.2f}', xy=(bs, tp),
                               xytext=(0, -12), textcoords='offset points',
                               ha='center', fontsize=7, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor=color2, alpha=0.7))
            
            # Formatting
            ax.set_xlabel('Block Size (MB)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Throughput (GB/s)', fontsize=11, fontweight='bold')
            ax.set_title(f'{op_label} - {thread_count} Threads',
                        fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            # Use linear scale for x-axis
            ax.set_xticks(block_sizes_for_plot)
            ax.set_xticklabels([str(bs) for bs in block_sizes_for_plot])
            
            # Set y-axis to start from 0 for better comparison
            ax.set_ylim(bottom=0)
            
            # Add legend
            ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout(rect=(0, 0.02, 1, 0.94))
    
    # Save figure
    if results_file2:
        # Create comparison output filename
        output_png = results_file1.replace('.json', '_vs_comparison_throughput_plots.png')
    else:
        output_png = results_file1.replace('.json', '_throughput_plots.png')
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nThroughput plots saved to {output_png}")
    plt.close()
    
    print("Throughput plotting complete!")


if __name__ == "__main__":
    # Generate plots
    # plot_results_threads_comparison()
    
    # Generate block size comparison plots
    # results_file = 'compare_file_operations/results/block_size_comparison_10gb.json'
    # plot_throughput_tables(results_file)
    # plot_block_size_heatmaps(results_file)
    
    # Generate throughput vs block size plots for 1000 blocks test
    results_file_cpp = 'compare_file_operations/results/block_size_comparison_1000blocks_cpp_True.json'
    results_file_python = 'compare_file_operations/results/block_size_comparison_1000blocks_2.json'

    plot_block_size_throughput_by_threads(results_file_cpp,results_file_python)