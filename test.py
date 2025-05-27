"""
Image Compression Scheduler Testing Module

This module tests the performance of different scheduling algorithms (FCFS, Greedy, DP)
under various conditions and produces visual charts for comparison.

Usage:
    python scheduler_tester.py --input_dir <path_to_images> --output_dir <path_for_results>
"""

# Set default directory paths here for easier configuration
DEFAULT_INPUT_DIR = r"D:\thesis-dataset\archive30k\flickr30k_images"  # Directory containing source images
DEFAULT_OUTPUT_DIR = r"D:\thesis-dataset\30kafter"  # Directory for results and charts
DEFAULT_QUALITY = 50  # Compression quality (1-100, lower = more compression)

import os
import time
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict

# Import the scheduling algorithms and compressor
import fcfs, greedy, dp, compressor

class SchedulerTester:
    def __init__(self, input_dir, output_dir, quality=50, charts_dir=None):
        """
        Initialize the scheduler tester.
        
        Args:
            input_dir: Directory with source images
            output_dir: Directory for compressed images and results
            quality: Compression quality (1-100)
            charts_dir: Directory for saving charts (defaults to output_dir/charts)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.quality = quality
        self.charts_dir = charts_dir or os.path.join(output_dir, "charts")
        self.datasets_dir = os.path.join(output_dir, "datasets")
        self.results = []
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        
    def create_datasets(self):
        """Create varied test datasets from input directory"""
        print("Creating test datasets...")
        
        # Get all image files from the input directory
        all_files = []
        for filename in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, filename)
            if os.path.isfile(file_path) and self._is_image(file_path):
                all_files.append((filename, file_path, os.path.getsize(file_path)))
        
        if not all_files:
            raise ValueError(f"No image files found in {self.input_dir}")
        
        # Create datasets
        datasets = {}
        
        # 1. Small vs Large Files Dataset
        # Sort by file size
        all_files.sort(key=lambda x: x[2])
        
        # Small images dataset (bottom 30%)
        small_dataset_dir = os.path.join(self.datasets_dir, "small_images")
        os.makedirs(small_dataset_dir, exist_ok=True)
        small_images = all_files[:max(3, len(all_files) // 3)]
        for filename, file_path, _ in small_images:
            shutil.copy2(file_path, os.path.join(small_dataset_dir, filename))
        datasets["small_images"] = small_dataset_dir
        
        # Large images dataset (top 30%)
        large_dataset_dir = os.path.join(self.datasets_dir, "large_images")
        os.makedirs(large_dataset_dir, exist_ok=True)
        large_images = all_files[-max(3, len(all_files) // 3):]
        for filename, file_path, _ in large_images:
            shutil.copy2(file_path, os.path.join(large_dataset_dir, filename))
        datasets["large_images"] = large_dataset_dir
        
        # 2. Uniform vs Diverse Content Datasets
        # Get resolution and aspect ratio variance to determine uniformity
        image_metadata = []
        for filename, file_path, size in all_files:
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    resolution = width * height
                    image_metadata.append((filename, file_path, size, width, height, aspect_ratio, resolution))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        # Calculate variances
        if image_metadata:
            aspect_ratios = [m[5] for m in image_metadata]
            resolutions = [m[6] for m in image_metadata]
            
            # Find images with similar aspect ratios
            mean_aspect = np.mean(aspect_ratios)
            aspect_diffs = [abs(ar - mean_aspect) for ar in aspect_ratios]
            
            # Sort by distance from the mean aspect ratio
            uniform_indices = np.argsort(aspect_diffs)[:max(3, len(image_metadata) // 3)]
            diverse_indices = np.argsort(aspect_diffs)[-max(3, len(image_metadata) // 3):]
            
            # Create uniform dataset (similar aspect ratios)
            uniform_dataset_dir = os.path.join(self.datasets_dir, "uniform_images")
            os.makedirs(uniform_dataset_dir, exist_ok=True)
            for idx in uniform_indices:
                filename, file_path, _, _, _, _, _ = image_metadata[idx]
                shutil.copy2(file_path, os.path.join(uniform_dataset_dir, filename))
            datasets["uniform_images"] = uniform_dataset_dir
            
            # Create diverse dataset (varied aspect ratios)
            diverse_dataset_dir = os.path.join(self.datasets_dir, "diverse_images")
            os.makedirs(diverse_dataset_dir, exist_ok=True)
            for idx in diverse_indices:
                filename, file_path, _, _, _, _, _ = image_metadata[idx]
                shutil.copy2(file_path, os.path.join(diverse_dataset_dir, filename))
            datasets["diverse_images"] = diverse_dataset_dir
        
        # 3. Mixed dataset - a blend of all image types
        mixed_dataset_dir = os.path.join(self.datasets_dir, "mixed_images")
        os.makedirs(mixed_dataset_dir, exist_ok=True)
        
        # Take a mix of images (some small, some large, some uniform, some diverse)
        mixed_indices = set()
        if len(all_files) >= 10:
            # Add some small, large, uniform, and diverse images to the mix
            mixed_indices.update(range(min(3, len(all_files) // 10)))  # Some small
            mixed_indices.update(range(len(all_files) - min(3, len(all_files) // 10), len(all_files)))  # Some large
            if image_metadata:
                for idx in uniform_indices[:min(3, len(uniform_indices) // 2)]:
                    mixed_indices.add(idx)
                for idx in diverse_indices[:min(3, len(diverse_indices) // 2)]:
                    mixed_indices.add(idx)
        else:
            # Just use all files if we don't have many
            mixed_indices = set(range(len(all_files)))
        
        for idx in mixed_indices:
            if idx < len(all_files):
                filename, file_path, _ = all_files[idx]
                shutil.copy2(file_path, os.path.join(mixed_dataset_dir, filename))
        datasets["mixed_images"] = mixed_dataset_dir
        
        print(f"Created {len(datasets)} datasets:")
        for name, path in datasets.items():
            file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"  - {name}: {file_count} images")
        
        return datasets
    
    def test_all_algorithms(self, datasets=None):
        """
        Run tests on all algorithms with all datasets
        
        Args:
            datasets: Dictionary of dataset name -> directory path
                     If None, will create datasets from the input directory
        """
        if datasets is None:
            datasets = self.create_datasets()
        
        # For each dataset, test all algorithms
        for dataset_name, dataset_dir in datasets.items():
            print(f"\nTesting with dataset: {dataset_name}")
            
            # Create output directory for this dataset
            dataset_output_dir = os.path.join(self.output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Get image info with timings
            image_info, processing_times, total_time = self._get_image_info(dataset_dir, dataset_output_dir)
            
            if not image_info:
                print(f"  No valid images found in {dataset_dir}")
                continue
                
            print(f"  Found {len(image_info)} images, total processing time: {total_time:.2f}s")
            
            # Define time budgets for testing
            time_budgets = [
                total_time * 0.3,  # Tight budget (30%)
                total_time * 0.6,  # Medium budget (60%)
                total_time * 0.9,  # Loose budget (90%)
                None  # No budget
            ]
            
            # Test each algorithm with each time budget
            for algorithm in ["fcfs", "greedy", "dp"]:
                for time_budget in time_budgets:
                    self._test_algorithm(algorithm, image_info, dataset_dir, dataset_output_dir, 
                                         processing_times, time_budget, dataset_name)
            
            # Generate per-dataset charts
            self._generate_dataset_charts(dataset_name)
        
        # Generate comparative charts across all datasets
        self._generate_comparative_charts()
        
        return self.results
    
    def _test_algorithm(self, algorithm, image_info, input_dir, output_dir, processing_times, time_budget, dataset_name):
        """Run a test for a specific algorithm with a specific time budget"""
        # Clear the output directory
        self._clear_directory(output_dir)
        
        budget_label = f"{time_budget:.2f}s" if time_budget is not None else "No budget"
        print(f"  Testing {algorithm.upper()} with {budget_label}")
        
        # Measure scheduling time (algorithm overhead)
        start_time = time.time()
        if algorithm == "fcfs":
            # FCFS doesn't support time budget in its original form
            # For FCFS with time budget, we'll need to implement it here
            if time_budget is not None:
                # Create a list of items with their processing times
                items = [(i, info[0], info[4]) for i, info in enumerate(image_info)]
                selected = []
                total_time = 0
                
                # Process in order until we exceed the time budget
                for i, filename, proc_time in items:
                    if total_time + proc_time <= time_budget:
                        selected.append((i, filename))
                        total_time += proc_time
                
                ordered_filenames = [item[1] for item in selected]
            else:
                # Regular FCFS without time budget
                ordered_filenames = fcfs.schedule(image_info)
        elif algorithm == "greedy":
            ordered_filenames = greedy.schedule(image_info, time_budget)
        elif algorithm == "dp":
            ordered_filenames = dp.schedule(image_info, time_budget)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        scheduling_time = time.time() - start_time
        
        # Run the actual compression
        start_time = time.time()
        processed, skipped, _, _ = compressor.compress_images(
            input_dir, output_dir, self.quality, ordered_filenames, processing_times
        )
        processing_time = time.time() - start_time
        
        # Calculate space saved
        space_saved = 0
        for filename in ordered_filenames:
            for fname, orig_size, _, saved, _ in image_info:
                if fname == filename:
                    space_saved += saved
                    break
        
        # Calculate efficiency (space saved per second)
        efficiency = space_saved / max(0.001, processing_time)
        
        # Calculate completion rate
        completion_rate = len(ordered_filenames) / len(image_info)
        
        # Store results
        self.results.append({
            'dataset': dataset_name,
            'algorithm': algorithm,
            'time_budget': time_budget,
            'scheduling_time': scheduling_time,
            'processing_time': processing_time,
            'space_saved': space_saved,
            'files_processed': len(ordered_filenames),
            'total_files': len(image_info),
            'completion_rate': completion_rate,
            'efficiency': efficiency
        })
        
        print(f"    - Files processed: {len(ordered_filenames)}/{len(image_info)}")
        print(f"    - Space saved: {space_saved:.2f} KB")
        print(f"    - Processing time: {processing_time:.2f}s")
        print(f"    - Scheduling overhead: {scheduling_time:.2f}s")
        print(f"    - Efficiency: {efficiency:.2f} KB/s")
        
    def _get_image_info(self, input_dir, output_dir):
        """
        Process all images to get accurate timing and compression information
        
        Returns:
            image_info: List of tuples (filename, orig_size, comp_size, space_saved, proc_time)
            processing_times: Dictionary mapping filenames to processing times
            total_time: Total processing time for all images
        """
        # Clear the output directory
        self._clear_directory(output_dir)
        
        # Get all images from the directory
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and self._is_image(os.path.join(input_dir, f))]
        
        # Process all images to get accurate timing
        processed, skipped, total_time, measured_times = compressor.compress_images(
            input_dir, output_dir, self.quality
        )
        
        # Gather image info
        image_info = []
        for filename in all_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Skip if not compressed successfully
            if not os.path.exists(output_path):
                continue
                
            # Get file sizes
            original_size = os.path.getsize(input_path) / 1024  # KB
            compressed_size = os.path.getsize(output_path) / 1024  # KB
            space_saved = original_size - compressed_size
            
            # Get processing time
            proc_time = measured_times.get(filename, 0)
            
            # Add to image info
            image_info.append((filename, original_size, compressed_size, space_saved, proc_time))
        
        return image_info, measured_times, total_time
    
    def _generate_dataset_charts(self, dataset_name):
        """Generate charts for a specific dataset"""
        # Filter results for this dataset
        dataset_results = [r for r in self.results if r['dataset'] == dataset_name]
        
        if not dataset_results:
            return
            
        # Create a DataFrame
        df = pd.DataFrame(dataset_results)
        
        # Create charts directory for this dataset
        dataset_charts_dir = os.path.join(self.charts_dir, dataset_name)
        os.makedirs(dataset_charts_dir, exist_ok=True)
        
        # 1. Space saved comparison
        self._create_bar_chart(df, 'algorithm', 'space_saved', 'time_budget',
                              f'Space Saved by Algorithm ({dataset_name})', 
                              'Algorithm', 'Space Saved (KB)',
                              os.path.join(dataset_charts_dir, 'space_saved.png'))
        
        # 2. Efficiency comparison
        self._create_bar_chart(df, 'algorithm', 'efficiency', 'time_budget',
                              f'Space-Time Efficiency by Algorithm ({dataset_name})', 
                              'Algorithm', 'Efficiency (KB/s)',
                              os.path.join(dataset_charts_dir, 'efficiency.png'))
        
        # 3. Completion rate comparison
        self._create_bar_chart(df, 'algorithm', 'completion_rate', 'time_budget',
                              f'Completion Rate by Algorithm ({dataset_name})', 
                              'Algorithm', 'Completion Rate',
                              os.path.join(dataset_charts_dir, 'completion_rate.png'),
                              percentage=True)
        
        # 4. Processing time comparison
        self._create_bar_chart(df, 'algorithm', 'processing_time', 'time_budget',
                              f'Processing Time by Algorithm ({dataset_name})', 
                              'Algorithm', 'Processing Time (s)',
                              os.path.join(dataset_charts_dir, 'processing_time.png'))
        
        # 5. Scheduling overhead comparison
        self._create_bar_chart(df, 'algorithm', 'scheduling_time', 'time_budget',
                              f'Scheduling Overhead by Algorithm ({dataset_name})', 
                              'Algorithm', 'Scheduling Time (s)',
                              os.path.join(dataset_charts_dir, 'scheduling_overhead.png'))
                              
        # 6. Time budget impact chart
        self._create_time_budget_impact_chart(df, dataset_name, dataset_charts_dir)
        
        # 7. Space-Time scatter plot
        self._create_space_time_scatter(df, dataset_name, dataset_charts_dir)
    
    def _generate_comparative_charts(self):
        """Generate charts comparing performance across all datasets"""
        if not self.results:
            return
            
        # Create a DataFrame with all results
        df = pd.DataFrame(self.results)
        
        # 1. Space saved by dataset and algorithm (no budget)
        self._create_comparative_bar_chart(df[df['time_budget'].isna()], 'algorithm', 'space_saved', 'dataset',
                                         'Space Saved by Algorithm and Dataset (No Budget)', 
                                         'Algorithm', 'Space Saved (KB)',
                                         os.path.join(self.charts_dir, 'space_saved_by_dataset.png'))
        
        # 2. Efficiency by dataset and algorithm (no budget)
        self._create_comparative_bar_chart(df[df['time_budget'].isna()], 'algorithm', 'efficiency', 'dataset',
                                         'Efficiency by Algorithm and Dataset (No Budget)', 
                                         'Algorithm', 'Efficiency (KB/s)',
                                         os.path.join(self.charts_dir, 'efficiency_by_dataset.png'))
        
        # 3. Completion rate by dataset, algorithm, and time budget
        self._create_completion_rate_heatmap(df, os.path.join(self.charts_dir, 'completion_rate_heatmap.png'))
        
        # 4. Scheduling overhead by algorithm and dataset
        self._create_comparative_bar_chart(df, 'algorithm', 'scheduling_time', 'dataset',
                                         'Scheduling Overhead by Algorithm and Dataset', 
                                         'Algorithm', 'Scheduling Time (s)',
                                         os.path.join(self.charts_dir, 'scheduling_overhead_by_dataset.png'))
        
        # 5. Algorithm performance radar chart (normalized metrics)
        self._create_radar_chart(df, os.path.join(self.charts_dir, 'algorithm_radar_chart.png'))
        
        # 6. Time budget impact across datasets
        self._create_time_budget_facet_chart(df, os.path.join(self.charts_dir, 'time_budget_impact_by_dataset.png'))
    
    def _create_bar_chart(self, df, x, y, hue, title, xlabel, ylabel, filename, percentage=False):
        """Create a bar chart with grouped bars"""
        plt.figure(figsize=(10, 6))
        
        # Convert time budget to more readable format for legend
        df_plot = df.copy()
        df_plot['budget_label'] = df_plot['time_budget'].apply(
            lambda x: 'No Budget' if pd.isna(x) else f'{x:.2f}s')
        
        # Create the grouped bar chart
        ax = sns.barplot(data=df_plot, x=x, y=y, hue='budget_label')
        
        # Customize the chart
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # Add percentage to y-axis if requested
        if percentage:
            ax.set_yticklabels([f'{x:.0%}' for x in ax.get_yticks()])
            
        # Add value labels on bars
        for container in ax.containers:
            labels = []
            for v in container:
                value = v.get_height()
                if percentage:
                    labels.append(f'{value:.0%}')
                else:
                    labels.append(f'{value:.1f}')
            ax.bar_label(container, labels=labels, padding=3)
            
        # Adjust legend
        plt.legend(title='Time Budget')
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def _create_comparative_bar_chart(self, df, x, y, hue, title, xlabel, ylabel, filename):
        """Create a bar chart comparing metrics across datasets"""
        plt.figure(figsize=(12, 6))
        
        # Create the bar chart
        ax = sns.barplot(data=df, x=x, y=y, hue=hue)
        
        # Customize the chart
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)
            
        # Adjust legend
        plt.legend(title='Dataset')
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def _create_completion_rate_heatmap(self, df, filename):
        """Create a heatmap showing completion rate by algorithm, dataset, and time budget"""
        # Prepare data: pivot to get algorithm x budget with dataset as values
        pivot_data = pd.pivot_table(
            df, 
            values='completion_rate', 
            index=['dataset', 'algorithm'],
            columns='time_budget', 
            aggfunc='mean'
        )
        
        # Rename columns for better readability
        new_columns = {}
        for col in pivot_data.columns:
            if pd.isna(col):
                new_columns[col] = 'No Budget'
            else:
                new_columns[col] = f'{col:.2f}s'
        
        pivot_data = pivot_data.rename(columns=new_columns)
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.0%', cmap='YlGnBu', vmin=0, vmax=1)
        
        plt.title('Completion Rate by Algorithm, Dataset, and Time Budget')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def _create_radar_chart(self, df, filename):
        """Create a radar chart comparing normalized algorithm performance"""
        # Filter to no budget case for cleaner comparison
        no_budget_df = df[df['time_budget'].isna()]
        
        if no_budget_df.empty:
            return
            
        # Select metrics to compare
        metrics = ['space_saved', 'efficiency', 'completion_rate', 'files_processed']
        
        # Normalize metrics by algorithm (across all datasets)
        normalized_df = pd.DataFrame()
        
        for algorithm in no_budget_df['algorithm'].unique():
            alg_df = no_budget_df[no_budget_df['algorithm'] == algorithm]
            
            # Calculate means for this algorithm
            alg_means = alg_df[metrics].mean()
            
            # Create a row for the normalized data
            normalized_df = pd.concat([normalized_df, pd.DataFrame([alg_means], index=[algorithm])])
        
        # Normalize each metric to 0-1 scale
        for metric in metrics:
            normalized_df[metric] = normalized_df[metric] / normalized_df[metric].max()
        
        # Create the radar chart
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Set number of metrics
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Set axis labels with better names
        metric_labels = ['Space Saved', 'Efficiency', 'Completion Rate', 'Files Processed']
        plt.xticks(angles[:-1], metric_labels, size=12)
        
        # Draw axis lines for each angle and label
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1], ["25%", "50%", "75%", "100%"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each algorithm
        colors = {'fcfs': 'blue', 'greedy': 'green', 'dp': 'red'}
        for algorithm in normalized_df.index:
            values = normalized_df.loc[algorithm].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=algorithm.upper(), 
                   color=colors.get(algorithm, 'black'))
            ax.fill(angles, values, alpha=0.1, color=colors.get(algorithm, 'black'))
        
        plt.title('Algorithm Performance Comparison\n(Normalized Metrics, No Budget)', size=15)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def _create_time_budget_impact_chart(self, df, dataset_name, output_dir):
        """Create a chart showing the impact of time budget on performance"""
        plt.figure(figsize=(12, 6))
        
        # Convert time budget to readable format
        df_plot = df.copy()
        df_plot['budget_label'] = df_plot['time_budget'].apply(
            lambda x: 'No Budget' if pd.isna(x) else f'{x:.2f}s')
        
        # Sort values for better readability
        budget_order = sorted(
            [b for b in df_plot['budget_label'].unique() if b != 'No Budget'], 
            key=lambda x: float(x[:-1]) if x != 'No Budget' else float('inf')
        ) + ['No Budget']
        
        # Create the line chart
        sns.lineplot(data=df_plot, x='budget_label', y='completion_rate', hue='algorithm', 
                     marker='o', sort=False)
        
        # Set custom order for x-axis
        plt.xticks(range(len(budget_order)), budget_order)
        
        # Customize the chart
        plt.title(f'Impact of Time Budget on Completion Rate ({dataset_name})')
        plt.xlabel('Time Budget')
        plt.ylabel('Completion Rate')
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.legend(title='Algorithm')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_budget_impact.png'), dpi=300)
        plt.close()
    
    def _create_time_budget_facet_chart(self, df, filename):
        """Create a facet chart showing time budget impact across all datasets"""
        # Convert time budget to readable format
        df_plot = df.copy()
        df_plot['budget_label'] = df_plot['time_budget'].apply(
            lambda x: 'No Budget' if pd.isna(x) else f'{x:.2f}s')
        
        # Create the facet chart
        g = sns.FacetGrid(df_plot, col='dataset', height=4, aspect=1.2, sharey=True)
        g.map_dataframe(sns.lineplot, x='budget_label', y='completion_rate', hue='algorithm', 
                       marker='o')
        
        # Customize each subplot
        for ax in g.axes.flat:
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        g.set_titles('{col_name} Dataset')
        g.set_axis_labels('Time Budget', 'Completion Rate')
        g.add_legend(title='Algorithm')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def _create_space_time_scatter(self, df, dataset_name, output_dir):
        """Create a scatter plot showing space saved vs processing time"""
        plt.figure(figsize=(10, 6))
        
        # Convert time budget to readable format for legend
        df_plot = df.copy()
        df_plot['budget_label'] = df_plot['time_budget'].apply(
            lambda x: 'No Budget' if pd.isna(x) else f'{x:.2f}s')
        
        # Create markers/colors for algorithms and time budgets
        algorithms = df_plot['algorithm'].unique()
        budgets = df_plot['budget_label'].unique()
        
        markers = {'fcfs': 'o', 'greedy': 's', 'dp': '^'}
        colors = {'No Budget': 'blue', 
                 f'{df_plot["time_budget"].min():.2f}s': 'green',
                 f'{df_plot["time_budget"].median():.2f}s': 'orange',
                 f'{df_plot["time_budget"].max():.2f}s': 'red'}
        
        # Plot each algorithm/budget combination
        for algorithm in algorithms:
            for budget in budgets:
                data = df_plot[(df_plot['algorithm'] == algorithm) & (df_plot['budget_label'] == budget)]
                if not data.empty:
                    plt.scatter(
                        data['processing_time'], 
                        data['space_saved'],
                        label=f"{algorithm.upper()} ({budget})",
                        marker=markers.get(algorithm, 'o'),
                        s=100,  # Size
                        alpha=0.7
                    )
        
        plt.title(f'Space-Time Performance ({dataset_name})')
        plt.xlabel('Processing Time (s)')
        plt.ylabel('Space Saved (KB)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'space_time_scatter.png'), dpi=300)
        plt.close()
    
    def _clear_directory(self, directory):
        """Clear all files in a directory but keep the directory itself"""
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    
    def _is_image(self, file_path):
        """Check if a file is a valid image"""
        try:
            with Image.open(file_path) as img:
                # If it loads without error, it's an image
                return True
        except:
            return False
    
    def get_results_dataframe(self):
        """Return the results as a pandas DataFrame"""
        return pd.DataFrame(self.results)
    
    def save_results(self, filename='results.csv'):
        """Save the test results to a CSV file"""
        if self.results:
            df = pd.DataFrame(self.results)
            output_path = os.path.join(self.output_dir, filename)
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
            return output_path
        return None


def main():
    """Main function to run the scheduler tester"""
    parser = argparse.ArgumentParser(description='Test image compression scheduling algorithms')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR, 
                       help='Directory with source images')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, 
                       help='Directory for results and charts')
    parser.add_argument('--quality', type=int, default=DEFAULT_QUALITY, 
                       help='Compression quality (1-100, lower = more compression)')
    
    args = parser.parse_args()
    
    # Create tester
    tester = SchedulerTester(args.input_dir, args.output_dir, args.quality)
    
    # Run tests
    tester.test_all_algorithms()
    
    # Save results
    tester.save_results()
    
    print("Testing completed. Check the output directory for results and charts.")
    
    # Print summary statistics
    results_df = tester.get_results_dataframe()
    if not results_df.empty:
        # Group by algorithm and calculate means
        summary = results_df.groupby('algorithm').agg({
            'space_saved': 'mean',
            'processing_time': 'mean',
            'efficiency': 'mean',
            'scheduling_time': 'mean',
            'completion_rate': 'mean'
        }).reset_index()
        
        print("\nSummary Statistics (averages across all datasets):")
        print("=" * 80)
        print(f"{'Algorithm':<10} {'Space Saved (KB)':<20} {'Processing Time (s)':<20} "
              f"{'Efficiency (KB/s)':<20} {'Scheduling Time (s)':<20} {'Completion Rate':<15}")
        print("-" * 80)
        
        for _, row in summary.iterrows():
            print(f"{row['algorithm']:<10} {row['space_saved']:<20.2f} {row['processing_time']:<20.2f} "
                  f"{row['efficiency']:<20.2f} {row['scheduling_time']:<20.2f} {row['completion_rate']:<15.2%}")
        print("=" * 80)


if __name__ == "__main__":
    # You can easily run the script without command line arguments
    # by just modifying the default directories at the top of the file
    main()
    
    # Alternative manual execution without using command-line args:
    # tester = SchedulerTester(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_QUALITY)
    # tester.test_all_algorithms()
    # tester.save_results()