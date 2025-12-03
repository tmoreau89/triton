import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Data is stored in csv files with the following columns:
# K,triton_tflops,cublas_tflops
# e.g. 128,698.575074996039,645.5313701032715


def visualize_results(results_csv_1: Path, results_csv_2: Path, output_dir: Path):
    """
    Visualize the results of two benchmark runs as bar charts
    """
    # s result on the left
    results_1 = pd.read_csv(results_csv_1)
    # s result on the right
    results_2 = pd.read_csv(results_csv_2)
    # Create a grid
    sns.set_theme(style="whitegrid")
    
    # Reshape data for grouped bar plots
    results_1_melted = results_1.melt(id_vars=['K'], 
                                       value_vars=['triton_tflops', 'cublas_tflops'],
                                       var_name='Library', 
                                       value_name='TFLOPS')
    results_1_melted['Run'] = 'Dec 1st'
    
    results_2_melted = results_2.melt(id_vars=['K'], 
                                       value_vars=['triton_tflops', 'cublas_tflops'],
                                       var_name='Library', 
                                       value_name='TFLOPS')
    results_2_melted['Run'] = 'Oct 10th'
    
    # Combine both datasets
    combined = pd.concat([results_1_melted, results_2_melted], ignore_index=True)
    
    # Clean up library names
    combined['Library'] = combined['Library'].str.replace('_tflops', '').str.replace('triton', 'Triton').str.replace('cublas', 'cuBLAS')
    
    # Create combined label for legend
    combined['Label'] = combined['Library'] + ' - ' + combined['Run']
    
    # Define custom colors: blue/lightblue for Triton, orange/lightsalmon for cuBLAS
    color_palette = {
        'Triton - Dec 1st': '#0000FF',      # blue
        'Triton - Oct 10th': '#87CEEB',     # lighter blue
        'cuBLAS - Dec 1st': '#FF8C00',      # orange
        'cuBLAS - Oct 10th': '#FFA07A'      # lighter orange
    }
    
    # Define bar order: Triton bars first, then cuBLAS bars
    bar_order = [
        'Triton - Dec 1st',
        'Triton - Oct 10th',
        'cuBLAS - Dec 1st',
        'cuBLAS - Oct 10th'
    ]
    
    # Plot with grouped bars and custom colors
    sns.barplot(x='K', y='TFLOPS', hue='Label', data=combined, palette=color_palette, hue_order=bar_order)
    plt.ylabel('TFLOPS')
    plt.xlabel('K')
    plt.legend()
    plt.title('Triton vs cuBLAS - ' + results_csv_1.name.replace('.csv', ''))
    plt.savefig(output_dir / results_csv_1.name.replace('.csv', '.png'))
    plt.close()

if __name__ == "__main__":
    visualize_results(Path('benchmark_results_12_1/fp8_results.csv'), Path('prod_benchmark_results2/fp8_results.csv'), Path('visualizations'))
    visualize_results(Path('benchmark_results_12_1/fp16_results.csv'), Path('prod_benchmark_results2/fp16_results.csv'), Path('visualizations'))
    visualize_results(Path('benchmark_results_12_1/mxfp8_results.csv'), Path('prod_benchmark_results2/mxfp8_results.csv'), Path('visualizations'))