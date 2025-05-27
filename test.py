"""
Image Compression Scheduler Testing Module

This module tests the performance of different scheduling algorithms (FCFS, Greedy, DP)
under various conditions and produces visual charts for comparison.
It uses app.py as the main driver for image compression processing.

Memory-optimized version that cleans up after each dataset.

Usage:
    python test.py --input_dir <path_to_images> --output_dir <path_for_results>
                  [--quality QUALITY] [--sample_percentage SAMPLE_PERCENTAGE]
                  [--create_datasets {True,False}] [--quick_test {True,False}]
"""

import os
import time
import shutil
import argparse
import logging
import sys
import json
import gc  # Added for garbage collection
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

# Data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Local modules
import app
import fcfs
import greedy
import dp


# Set default directory paths here for easier configuration
DEFAULT_INPUT_DIR = r"./pics/flickr30k_images"  # Directory containing source images
# DEFAULT_INPUT_DIR = r"./test"  # Directory containing source images
DEFAULT_OUTPUT_DIR = r"./results"  # Directory for results and charts
DEFAULT_QUALITY = 50  # Compression quality (1-100, lower = more compression)
DEFAULT_SAMPLE_PERCENTAGE = 100  # Percentage of image to process for estimation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("scheduler_tester")


class SchedulerTester:
    """
    Test performance of different image compression scheduling algorithms.
    
    This class creates test datasets, runs various scheduling algorithms under
    different conditions, and produces comparative visualizations and metrics.
    
    Memory-optimized version that processes one dataset at a time and cleans up after each.
    """
    
    def __init__(self, input_dir: str, output_dir: str, quality: int = 50,
                 sample_percentage: int = DEFAULT_SAMPLE_PERCENTAGE, 
                 charts_dir: Optional[str] = None,
                 create_datasets: bool = True,
                 quick_test: bool = False):
        """
        Initialize the scheduler tester.
        
        Args:
            input_dir: Directory with source images
            output_dir: Directory for compressed images and results
            quality: Compression quality (1-100)
            sample_percentage: Percentage of image to process for estimation
            charts_dir: Directory for saving charts (defaults to output_dir/charts)
            create_datasets: Whether to create test datasets
            quick_test: If True, run a limited test with fewer time budgets
        """
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.quality = quality
        self.sample_percentage = sample_percentage
        self.charts_dir = charts_dir or os.path.join(output_dir, "charts")
        self.datasets_dir = os.path.join(output_dir, "datasets")
        self.results = []  # We'll keep appending to this but clean up other data
        self.create_datasets_flag = create_datasets
        self.quick_test = quick_test
        
        # Create necessary directories
        self._create_directories()
        
        # Store test timestamp
        self.test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized SchedulerTester with input_dir={input_dir}, output_dir={output_dir}")

    def _create_directories(self) -> None:
        """Create all necessary directories for test outputs."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        # Create subdirectories for organization
        os.makedirs(os.path.join(self.charts_dir, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(self.charts_dir, "comparative"), exist_ok=True)

    def create_datasets(self) -> Dict[str, str]:
        """
        Create varied test datasets from input directory.
        
        Returns:
            Dictionary mapping dataset names to their directory paths
        """
        if not self.create_datasets_flag:
            logger.info("Skipping dataset creation (using existing datasets)")
            # Return paths to existing datasets if they exist
            existing_datasets = {}
            for dataset_name in ["small_images", "large_images", "uniform_images", "diverse_images", "mixed_images"]:
                dataset_path = os.path.join(self.datasets_dir, dataset_name)
                if os.path.exists(dataset_path) and os.listdir(dataset_path):
                    existing_datasets[dataset_name] = dataset_path
            
            if existing_datasets:
                logger.info(f"Found {len(existing_datasets)} existing datasets")
                return existing_datasets
            else:
                logger.warning("No existing datasets found. Creating new datasets.")
                self.create_datasets_flag = True
        
        logger.info("Creating test datasets...")
        
        # Get all image files from the input directory
        all_files = []
        for filename in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, filename)
            if os.path.isfile(file_path) and self._is_image(file_path):
                all_files.append((filename, file_path, os.path.getsize(file_path)))
        
        if not all_files:
            error_msg = f"No image files found in {self.input_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
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
                    aspect_ratio = width / height if height > 0 else 1.0
                    resolution = width * height
                    image_metadata.append((filename, file_path, size, width, height, aspect_ratio, resolution))
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
        
        # Calculate variances
        if image_metadata:
            aspect_ratios = [m[5] for m in image_metadata]
            
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
        
        # Log dataset creation results
        logger.info(f"Created {len(datasets)} datasets:")
        for name, path in datasets.items():
            file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            logger.info(f"  - {name}: {file_count} images")
        
        # Free memory
        del all_files
        del image_metadata
        gc.collect()
        
        return datasets
    
    def test_all_algorithms(self, datasets: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Run tests on all algorithms with all datasets
        
        Args:
            datasets: Dictionary of dataset name -> directory path
                     If None, will create datasets from the input directory
        
        Returns:
            List of test result dictionaries
        """
        if datasets is None:
            datasets = self.create_datasets()
        
        # For each dataset, test all algorithms
        for dataset_name, dataset_dir in datasets.items():
            logger.info(f"\nTesting with dataset: {dataset_name}")
            
            # Process one dataset at a time to save memory
            self._process_dataset(dataset_name, dataset_dir)
            
            # Free up memory after processing each dataset
            gc.collect()
        
        # Generate comparative charts across all datasets
        # (This uses the aggregated results from all datasets)
        self._generate_comparative_charts()
        
        return self.results
    
    def _process_dataset(self, dataset_name: str, dataset_dir: str) -> None:
        """
        Process a single dataset with all algorithms and time budgets.
        
        Args:
            dataset_name: Name of the dataset
            dataset_dir: Path to the dataset directory
        """
        # Create output directory for this dataset
        dataset_output_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Get image info with timings by running the app's estimation process
        logger.info(f"Running estimation for {dataset_name}...")
        
        try:
            # Use app's run_estimation function
            image_info, processing_times, total_original_size, total_compressed_size, total_space_saved, total_time, _ = app.run_estimation(
                dataset_dir, dataset_output_dir, self.quality, self.sample_percentage
            )
            
            if not image_info:
                logger.warning(f"  No valid images found in {dataset_dir}")
                return
                
            logger.info(f"  Found {len(image_info)} images, total processing time: {total_time:.4f}s")
            
            # Define time budgets for testing
            if self.quick_test:
                # Use fewer time budgets for quick testing
                time_budgets = [total_time * 0.25, total_time * 0.5, total_time * 0.75, total_time, None]
            else:
                # More varied time budgets (10% to 100% in 10% increments)
                time_budgets = [total_time * (i / 10) for i in range(1, 11)]
                time_budgets.append(None)  # Also test with no budget
            
            # Test each algorithm with each time budget
            for algorithm in ["fcfs", "greedy", "dp"]:
                for time_budget in time_budgets:
                    self._test_algorithm(algorithm, image_info, dataset_dir, dataset_output_dir, 
                                        processing_times, time_budget, dataset_name)
            
            # Generate per-dataset charts before moving to next dataset
            self._generate_dataset_charts(dataset_name)
            
            # Clean up this dataset's processed data to save memory
            # Just keep the results which are already saved to self.results
            del image_info
            del processing_times
            
        except Exception as e:
            logger.error(f"Error testing dataset {dataset_name}: {e}", exc_info=True)
            
    def _test_algorithm(self, algorithm: str, image_info: List[Tuple], 
                        input_dir: str, output_dir: str, 
                        processing_times: Dict[str, float], 
                        time_budget: Optional[float], 
                        dataset_name: str) -> None:
        """
        Run a test for a specific algorithm with a specific time budget.
        
        Args:
            algorithm: Name of the scheduling algorithm to test
            image_info: List of image info tuples from estimation
            input_dir: Directory with source images
            output_dir: Directory for output images
            processing_times: Dictionary of processing times by filename
            time_budget: Time budget in seconds (None for no budget)
            dataset_name: Name of the dataset being tested
        """
        # Clear the output directory
        self._clear_directory(output_dir)
        
        budget_label = f"{time_budget:.4f}s" if time_budget is not None else "No budget"
        logger.info(f"  Testing {algorithm.upper()} with {budget_label}")
        
        # Measure scheduling time (algorithm overhead)
        start_time = time.time()
        
        try:
            # Use app's process_with_scheduler function
            total_processing_time, _ = app.process_with_scheduler(
                algorithm, 
                image_info, 
                input_dir, 
                output_dir, 
                self.quality, 
                processing_times,
                time_budget,
                {},  # compressed_files
                self.sample_percentage
            )
            
            scheduling_time = time.time() - start_time - total_processing_time
            
            # Calculate space saved and files processed from the results
            # We need to figure out which files were actually processed
            ordered_filenames = []
            
            # Determine which scheduling function was used
            if algorithm == "fcfs":
                ordered_filenames = fcfs.schedule(image_info, time_budget)
            elif algorithm == "greedy":
                ordered_filenames = greedy.schedule(image_info, time_budget)
            elif algorithm == "dp":
                ordered_filenames = dp.schedule(image_info, time_budget)
                
            # Calculate space saved
            space_saved = 0
            for filename in ordered_filenames:
                for fname, orig_size, _, saved, _ in image_info:
                    if fname == filename:
                        space_saved += saved
                        break
            
            # Calculate efficiency (space saved per second)
            efficiency = space_saved / max(0.001, total_processing_time)
            
            # Calculate completion rate
            completion_rate = len(ordered_filenames) / len(image_info)
            
            # Store results
            result = {
                'dataset': dataset_name,
                'algorithm': algorithm,
                'time_budget': time_budget,
                'scheduling_time': scheduling_time,
                'processing_time': total_processing_time,
                'space_saved': space_saved,
                'files_processed': len(ordered_filenames),
                'total_files': len(image_info),
                'completion_rate': completion_rate,
                'efficiency': efficiency,
                'timestamp': self.test_timestamp
            }
            
            self.results.append(result)
            
            logger.info(f"    - Files processed: {len(ordered_filenames)}/{len(image_info)}")
            logger.info(f"    - Space saved: {space_saved:.4f} KB")
            logger.info(f"    - Processing time: {total_processing_time:.4f}s")
            logger.info(f"    - Scheduling overhead: {scheduling_time:.4f}s")
            logger.info(f"    - Efficiency: {efficiency:.4f} KB/s")
            
        except Exception as e:
            logger.error(f"Error testing {algorithm} with {budget_label}: {e}", exc_info=True)

    def _create_bar_chart(self, df: pd.DataFrame, x: str, y: str, hue: str, 
                          title: str, xlabel: str, ylabel: str, 
                          filename: str, percentage: bool = False) -> None:
        """
        Create a bar chart with grouped bars.
        
        Args:
            df: DataFrame with the data
            x: Column to use for x-axis
            y: Column to use for y-axis
            hue: Column to use for grouping bars
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Path to save the chart
            percentage: Whether to format y-axis as percentage
        """
        # Increase figure size for better visualization
        plt.figure(figsize=(20, 12))
        
        # Convert time budget to more readable format for legend
        df_plot = df.copy()
        df_plot['budget_label'] = df_plot['time_budget'].apply(
            lambda x: 'No Budget' if pd.isna(x) else f'{x:.4f}s')
        
        # Get the list of unique algorithms and budget labels
        algorithms = df_plot[x].unique()
        budget_labels = df_plot['budget_label'].unique()
        
        # Configure the width and positions for bars
        total_width = 0.8  # Total width for all groups of bars
        group_width = total_width / len(budget_labels)  # Width for each budget group
        bar_width = group_width * 0.9  # Actual bar width (90% of group width to add spacing between bars)
        
        # Create custom positions for each bar group
        positions = np.arange(len(algorithms))
        
        # Create custom bar plot instead of using seaborn
        ax = plt.gca()
        
        # Setup color palette for different budget labels
        colors = plt.cm.viridis(np.linspace(0, 1, len(budget_labels)))
        
        # Sort budget labels for consistent ordering (with 'No Budget' at the end)
        sorted_budgets = sorted(
            [b for b in budget_labels if b != 'No Budget'], 
            key=lambda x: float(x[:-1]) if x != 'No Budget' else float('inf')
        ) + ['No Budget']
        
        # Create legend handles
        legend_handles = []
        
        # Plot each group of bars
        for i, budget in enumerate(sorted_budgets):
            # Calculate the position for this group
            group_pos = positions + (i - len(budget_labels)/2 + 0.5) * group_width
            
            # Filter data for this budget
            budget_data = df_plot[df_plot['budget_label'] == budget]
            
            # Create bars for each algorithm
            heights = []
            for j, alg in enumerate(algorithms):
                alg_data = budget_data[budget_data[x] == alg]
                if not alg_data.empty:
                    height = alg_data[y].values[0]
                    heights.append(height)
                    bar = ax.bar(group_pos[j], height, width=bar_width, color=colors[i], 
                            edgecolor='white', linewidth=1.5, label=budget if j == 0 else "")
                    
                    # Add value labels on bars
                    if percentage:
                        label = f'{height:.0%}'
                    else:
                        label = f'{height:.1f}'
                    ax.text(group_pos[j], height + (ax.get_ylim()[1] * 0.01), label, 
                            ha='center', va='bottom', fontsize=10)
            
            # Add to legend only once
            if heights:
                legend_handles.append(plt.Rectangle((0,0), 1, 1, color=colors[i], label=budget))
        
        # Set x-ticks at algorithm positions
        ax.set_xticks(positions)
        ax.set_xticklabels([alg.upper() for alg in algorithms])
        
        # Customize the chart
        plt.title(title, fontsize=18, pad=20)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        
        # Add percentage to y-axis if requested
        if percentage:
            current_values = ax.get_yticks()
            ax.set_yticks(current_values)
            ax.set_yticklabels([f'{x:.0%}' for x in current_values])
        
        # Add grid lines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust legend placement and style
        plt.legend(handles=legend_handles, title='Time Budget', title_fontsize=14, 
                 fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Ensure enough space for the chart
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    def _create_comparative_bar_chart(self, df: pd.DataFrame, x: str, y: str, hue: str, 
                                     title: str, xlabel: str, ylabel: str, 
                                     filename: str) -> None:
        """
        Create a bar chart comparing metrics across datasets.
        
        Args:
            df: DataFrame with the data
            x: Column to use for x-axis
            y: Column to use for y-axis
            hue: Column to use for grouping bars
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Path to save the chart
        """
        # Increase figure size for better readability
        plt.figure(figsize=(18, 12))
        
        # Get the list of unique algorithms and datasets
        algorithms = df[x].unique()
        datasets = df[hue].unique()
        
        # Configure the width and positions for bars
        total_width = 0.8  # Total width for all groups of bars
        group_width = total_width / len(datasets)  # Width for each dataset group
        bar_width = group_width * 0.9  # Actual bar width (90% of group width to add spacing between bars)
        
        # Create custom positions for each bar group
        positions = np.arange(len(algorithms))
        
        # Create custom bar plot
        ax = plt.gca()
        
        # Setup color palette for different datasets
        colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))
        
        # Create legend handles
        legend_handles = []
        
        # Plot each group of bars
        for i, dataset in enumerate(datasets):
            # Calculate the position for this group
            group_pos = positions + (i - len(datasets)/2 + 0.5) * group_width
            
            # Filter data for this dataset
            dataset_data = df[df[hue] == dataset]
            
            # Create bars for each algorithm
            heights = []
            for j, alg in enumerate(algorithms):
                alg_data = dataset_data[dataset_data[x] == alg]
                if not alg_data.empty:
                    height = alg_data[y].values[0]
                    heights.append(height)
                    bar = ax.bar(group_pos[j], height, width=bar_width, color=colors[i], 
                            edgecolor='white', linewidth=1.5, label=dataset if j == 0 else "")
                    
                    # Add value labels on bars
                    ax.text(group_pos[j], height + (ax.get_ylim()[1] * 0.01), f'{height:.1f}', 
                            ha='center', va='bottom', fontsize=10)
            
            # Add to legend only once
            if heights:
                legend_handles.append(plt.Rectangle((0,0), 1, 1, color=colors[i], label=dataset))
        
        # Set x-ticks at algorithm positions
        ax.set_xticks(positions)
        ax.set_xticklabels([alg.upper() for alg in algorithms], rotation=30, ha='right')
        
        # Customize the chart
        plt.title(title, fontsize=18, pad=20)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        
        # Add grid lines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust legend placement and style
        plt.legend(handles=legend_handles, title='Dataset', title_fontsize=14, 
                 fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Ensure enough space for the chart
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    def _generate_dataset_charts(self, dataset_name: str) -> None:
        """
        Generate charts for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to generate charts for
        """
        # Filter results for this dataset
        dataset_results = [r for r in self.results if r['dataset'] == dataset_name]
        
        if not dataset_results:
            logger.warning(f"No results for dataset {dataset_name}, skipping chart generation")
            return
            
        # Create a DataFrame
        df = pd.DataFrame(dataset_results)
        
        # Create charts directory for this dataset
        dataset_charts_dir = os.path.join(self.charts_dir, "datasets", dataset_name)
        os.makedirs(dataset_charts_dir, exist_ok=True)
        
        try:
            # Set seaborn style for better looking charts
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 12
            
            # 1. Space saved comparison
            self._create_bar_chart(df, 'algorithm', 'space_saved', 'time_budget',
                                f'Space Saved by Algorithm ({dataset_name})', 
                                'Algorithm', 'Space Saved (KB)',
                                os.path.join(dataset_charts_dir, 'space_saved.png'))
            plt.close() # Free memory after each chart
            
            # 2. Efficiency comparison
            self._create_bar_chart(df, 'algorithm', 'efficiency', 'time_budget',
                                f'Space-Time Efficiency by Algorithm ({dataset_name})', 
                                'Algorithm', 'Efficiency (KB/s)',
                                os.path.join(dataset_charts_dir, 'efficiency.png'))
            plt.close()
            
            # 3. Completion rate comparison
            self._create_bar_chart(df, 'algorithm', 'completion_rate', 'time_budget',
                                f'Completion Rate by Algorithm ({dataset_name})', 
                                'Algorithm', 'Completion Rate',
                                os.path.join(dataset_charts_dir, 'completion_rate.png'),
                                percentage=True)
            plt.close()
            
            # 4. Processing time comparison
            self._create_bar_chart(df, 'algorithm', 'processing_time', 'time_budget',
                                f'Processing Time by Algorithm ({dataset_name})', 
                                'Algorithm', 'Processing Time (s)',
                                os.path.join(dataset_charts_dir, 'processing_time.png'))
            plt.close()
            
            # 5. Scheduling overhead comparison
            self._create_bar_chart(df, 'algorithm', 'scheduling_time', 'time_budget',
                                f'Scheduling Overhead by Algorithm ({dataset_name})', 
                                'Algorithm', 'Scheduling Time (s)',
                                os.path.join(dataset_charts_dir, 'scheduling_overhead.png'))
            plt.close()
                                
            # 6. Time budget impact chart
            self._create_time_budget_impact_chart(df, dataset_name, dataset_charts_dir)
            plt.close()
            
            # 7. Space-Time scatter plot
            self._create_space_time_scatter(df, dataset_name, dataset_charts_dir)
            plt.close()
            
            logger.info(f"Generated charts for dataset: {dataset_name}")
        
        except Exception as e:
            logger.error(f"Error generating charts for dataset {dataset_name}: {e}", exc_info=True)
        
        # Clear the plot resources to save memory
        plt.close('all')
        gc.collect()
    
    def _create_time_budget_impact_chart(self, df: pd.DataFrame, dataset_name: str, output_dir: str) -> None:
        """
        Create a chart showing the impact of time budget on performance metrics.
        
        Args:
            df: DataFrame with the test results
            dataset_name: Name of the dataset
            output_dir: Directory to save the chart
        """
        plt.figure(figsize=(14, 10))
        
        # Convert time budget to readable format
        df_plot = df.copy()
        df_plot['budget_label'] = df_plot['time_budget'].apply(
            lambda x: 'No Budget' if pd.isna(x) else f'{x:.4f}s')
        
        # Sort values for better readability
        budget_order = sorted(
            [b for b in df_plot['budget_label'].unique() if b != 'No Budget'], 
            key=lambda x: float(x[:-1]) if x != 'No Budget' else float('inf')
        ) + ['No Budget']
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=True)
        
        # Plot completion rate
        for algorithm in df_plot['algorithm'].unique():
            alg_data = df_plot[df_plot['algorithm'] == algorithm]
            
            # Sort by time budget for proper display
            alg_data = alg_data.sort_values('time_budget')
            
            # Plot completion rate
            axes[0].plot(alg_data['budget_label'], alg_data['completion_rate'], 
                     marker='o', linewidth=2, markersize=8, label=algorithm.upper())
        
        axes[0].set_title(f'Impact of Time Budget on Completion Rate ({dataset_name})', fontsize=16)
        axes[0].set_ylabel('Completion Rate', fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].legend(title='Algorithm', fontsize=12)
        
        # Format y-axis as percentage for completion rate
        axes[0].set_ylim(0, 1.1)
        vals = axes[0].get_yticks()
        axes[0].set_yticklabels([f'{x:.0%}' for x in vals])
        
        # Plot efficiency
        for algorithm in df_plot['algorithm'].unique():
            alg_data = df_plot[df_plot['algorithm'] == algorithm]
            
            # Sort by time budget for proper display
            alg_data = alg_data.sort_values('time_budget')
            
            # Plot efficiency
            axes[1].plot(alg_data['budget_label'], alg_data['efficiency'], 
                     marker='s', linewidth=2, markersize=8, label=algorithm.upper())
        
        axes[1].set_title(f'Impact of Time Budget on Efficiency ({dataset_name})', fontsize=16)
        axes[1].set_xlabel('Time Budget', fontsize=14)
        axes[1].set_ylabel('Efficiency (KB/s)', fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend(title='Algorithm', fontsize=12)
        
        # Set custom order for x-axis
        plt.xticks(range(len(budget_order)), budget_order, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_budget_impact.png'), dpi=300)
        plt.close()
    
    def _create_space_time_scatter(self, df: pd.DataFrame, dataset_name: str, output_dir: str) -> None:
        """
        Create a scatter plot showing the tradeoff between space saved and processing time.
        
        Args:
            df: DataFrame with the test results
            dataset_name: Name of the dataset
            output_dir: Directory to save the chart
        """
        plt.figure(figsize=(12, 8))
        
        # Setup color palette for different algorithms
        colors = {'fcfs': '#1f77b4', 'greedy': '#ff7f0e', 'dp': '#2ca02c'}
        markers = {'fcfs': 'o', 'greedy': 's', 'dp': '^'}
        
        # Group by algorithm for cleaner visualization
        for algorithm in ['fcfs', 'greedy', 'dp']:
            alg_data = df[df['algorithm'] == algorithm]
            
            if not alg_data.empty:
                plt.scatter(alg_data['processing_time'], alg_data['space_saved'], 
                           s=100, alpha=0.7, label=algorithm.upper(), 
                           color=colors.get(algorithm, 'blue'), marker=markers.get(algorithm, 'o'))
        
        # Add a title and axis labels
        plt.title(f'Space Saved vs. Processing Time Tradeoff ({dataset_name})', fontsize=16)
        plt.xlabel('Processing Time (seconds)', fontsize=14)
        plt.ylabel('Space Saved (KB)', fontsize=14)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend(title='Algorithm', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, 'space_time_tradeoff.png'), dpi=300)
        plt.close()
    
    def _generate_comparative_charts(self) -> None:
        """
        Generate comparative charts across all datasets.
        This function creates charts that compare results across different datasets.
        """
        logger.info("Generating comparative charts across all datasets...")
        
        if not self.results:
            logger.warning("No results available for generating comparative charts")
            return
            
        try:
            # Create a DataFrame from all results
            df = pd.DataFrame(self.results)
            
            # Directory for comparative charts
            comparative_dir = os.path.join(self.charts_dir, "comparative")
            os.makedirs(comparative_dir, exist_ok=True)
            
            # 1. Space saved by dataset and algorithm (no budget)
            no_budget_df = df[df['time_budget'].isna()]
            if not no_budget_df.empty:
                # Group by dataset and algorithm to get mean values
                summary_df = no_budget_df.groupby(['dataset', 'algorithm']).agg({
                    'space_saved': 'mean',
                    'efficiency': 'mean',
                    'completion_rate': 'mean',
                    'scheduling_time': 'mean'
                }).reset_index()
                
                # Create comparative charts
                self._create_comparative_bar_chart(
                    summary_df, 'algorithm', 'space_saved', 'dataset',
                    'Space Saved by Algorithm and Dataset (No Budget)', 
                    'Algorithm', 'Space Saved (KB)',
                    os.path.join(comparative_dir, 'space_saved_by_dataset.png')
                )
                plt.close()
                
                self._create_comparative_bar_chart(
                    summary_df, 'algorithm', 'efficiency', 'dataset',
                    'Efficiency by Algorithm and Dataset (No Budget)', 
                    'Algorithm', 'Efficiency (KB/s)',
                    os.path.join(comparative_dir, 'efficiency_by_dataset.png')
                )
                plt.close()
                
                self._create_comparative_bar_chart(
                    summary_df, 'algorithm', 'scheduling_time', 'dataset',
                    'Scheduling Overhead by Algorithm and Dataset (No Budget)', 
                    'Algorithm', 'Scheduling Time (s)',
                    os.path.join(comparative_dir, 'scheduling_overhead_by_dataset.png')
                )
                plt.close()
            
            # 2. Create a completion rate heatmap for all time budgets
            self._create_completion_rate_heatmap(df, os.path.join(comparative_dir, 'completion_rate_heatmap.png'))
            plt.close()
            
            # 3. Create algorithm comparison radar chart
            self._create_radar_chart(df, os.path.join(comparative_dir, 'algorithm_radar_chart.png'))
            plt.close()
            
            # 4. Create time budget impact across datasets
            self._create_time_budget_facet_chart(df, os.path.join(comparative_dir, 'time_budget_impact_by_dataset.png'))
            plt.close()
            
            logger.info("Comparative charts generation completed")
            
        except Exception as e:
            logger.error(f"Error generating comparative charts: {e}", exc_info=True)
            
        # Clear the plot resources to save memory
        plt.close('all')
        gc.collect()
    
    def _create_completion_rate_heatmap(self, df: pd.DataFrame, filename: str) -> None:
        """
        Create a heatmap showing completion rate by algorithm, dataset, and time budget.
        
        Args:
            df: DataFrame with the test results
            filename: Path to save the heatmap
        """
        # Convert time budget to readable format
        df_plot = df.copy()
        df_plot['budget_label'] = df_plot['time_budget'].apply(
            lambda x: 'No Budget' if pd.isna(x) else f'{x:.4f}s')
        
        # Prepare data: pivot to get algorithm x budget with dataset as values
        pivot_data = pd.pivot_table(
            df_plot, 
            values='completion_rate', 
            index=['dataset', 'algorithm'],
            columns='budget_label', 
            aggfunc='mean'
        )
        
        # Sort columns for better readability
        budget_columns = sorted(
            [b for b in pivot_data.columns if b != 'No Budget'], 
            key=lambda x: float(x[:-1]) if x != 'No Budget' else float('inf')
        ) + ['No Budget']
        pivot_data = pivot_data[budget_columns]
        
        # Create the heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot_data, annot=True, fmt='.0%', cmap='YlGnBu', vmin=0, vmax=1)
        
        plt.title('Completion Rate by Algorithm, Dataset, and Time Budget', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def _create_radar_chart(self, df: pd.DataFrame, filename: str) -> None:
        """
        Create a radar chart comparing normalized algorithm performance.
        
        Args:
            df: DataFrame with the test results
            filename: Path to save the chart
        """
        # Filter to no budget case for cleaner comparison
        no_budget_df = df[df['time_budget'].isna()]
        
        if no_budget_df.empty:
            logger.warning("No data available for radar chart (no 'No Budget' results)")
            return
            
        # Select metrics to compare
        metrics = ['space_saved', 'efficiency', 'completion_rate']
        
        # Group by algorithm and calculate mean across datasets
        summary_df = no_budget_df.groupby('algorithm')[metrics].mean().reset_index()
        
        # Normalize each metric to 0-1 scale for fair comparison
        normalized_df = summary_df.copy()
        for metric in metrics:
            max_val = normalized_df[metric].max()
            if max_val > 0:
                normalized_df[metric] = normalized_df[metric] / max_val
        
        # Create the radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Set number of metrics
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Set axis labels with better names
        metric_labels = ['Space Saved', 'Efficiency', 'Completion Rate']
        plt.xticks(angles[:-1], metric_labels, size=14)
        
        # Draw axis lines for each angle and label
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1], ["25%", "50%", "75%", "100%"], color="grey", size=12)
        plt.ylim(0, 1)
        
        # Plot each algorithm
        colors = {'fcfs': '#1f77b4', 'greedy': '#ff7f0e', 'dp': '#2ca02c'}
        for idx, row in normalized_df.iterrows():
            algorithm = row['algorithm']
            values = row[metrics].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=algorithm.upper(), 
                   color=colors.get(algorithm, 'black'))
            ax.fill(angles, values, alpha=0.1, color=colors.get(algorithm, 'black'))
        
        plt.title('Algorithm Performance Comparison\n(Normalized Metrics, No Budget)', size=16)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def _create_time_budget_facet_chart(self, df: pd.DataFrame, filename: str) -> None:
        """
        Create a facet chart showing time budget impact across all datasets.
        
        Args:
            df: DataFrame with the test results
            filename: Path to save the chart
        """
        # Convert time budget to readable format
        df_plot = df.copy()
        df_plot['budget_label'] = df_plot['time_budget'].apply(
            lambda x: 'No Budget' if pd.isna(x) else f'{x:.4f}s')
        
        # Get unique datasets
        datasets = df_plot['dataset'].unique()
        
        # Create a figure with subplots for each dataset
        fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 4 * len(datasets)), sharex=True)
        
        # Sort budget labels for consistent ordering
        budget_order = sorted(
            [b for b in df_plot['budget_label'].unique() if b != 'No Budget'], 
            key=lambda x: float(x[:-1]) if x != 'No Budget' else float('inf')
        ) + ['No Budget']
        
        # Plot for each dataset
        for i, dataset in enumerate(datasets):
            ax = axes[i] if len(datasets) > 1 else axes
            
            # Filter for this dataset
            dataset_data = df_plot[df_plot['dataset'] == dataset]
            
            # Group by algorithm and budget, computing mean values
            grouped = dataset_data.groupby(['algorithm', 'budget_label'])['completion_rate'].mean().reset_index()
            
            # Plot each algorithm
            for algorithm in grouped['algorithm'].unique():
                alg_data = grouped[grouped['algorithm'] == algorithm]
                
                # Sort by budget for proper display
                budget_order_idx = {b: i for i, b in enumerate(budget_order)}
                alg_data['order'] = alg_data['budget_label'].map(budget_order_idx)
                alg_data = alg_data.sort_values('order')
                
                # Plot the line
                ax.plot(alg_data['budget_label'], alg_data['completion_rate'], 
                       marker='o', linewidth=2, markersize=8, label=algorithm.upper())
            
            # Customize the subplot
            ax.set_title(f'{dataset} Dataset', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format y-axis as percentage
            ax.set_ylim(0, 1.1)
            vals = ax.get_yticks()
            ax.set_yticklabels([f'{x:.0%}' for x in vals])
            
            # Add legend to the first subplot only
            if i == 0:
                ax.legend(title='Algorithm', fontsize=12)
        
        # Set common labels and customizations
        fig.suptitle('Impact of Time Budget on Completion Rate Across Datasets', fontsize=16, y=1.0)
        plt.xlabel('Time Budget', fontsize=14)
        axes[0].set_ylabel('Completion Rate', fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save the chart
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _clear_directory(self, directory: str) -> None:
        """
        Clear all files in a directory but keep the directory itself.
        
        Args:
            directory: Directory to clear
        """
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}. Reason: {e}")
    
    def _is_image(self, file_path: str) -> bool:
        """
        Check if a file is a valid image.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is a valid image, False otherwise
        """
        try:
            with Image.open(file_path) as img:
                # If it loads without error, it's an image
                return True
        except Exception:
            return False
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Return the results as a pandas DataFrame.
        
        Returns:
            DataFrame containing all test results
        """
        return pd.DataFrame(self.results)
    
    def save_results(self, filename: str = 'results.csv') -> str:
        """
        Save the test results to a CSV file.
        
        Args:
            filename: Name of the CSV file to save
            
        Returns:
            Path to the saved file, or None if no results available
        """
        if self.results:
            df = pd.DataFrame(self.results)
            output_path = os.path.join(self.output_dir, filename)
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            return output_path
        return None
    
    def print_summary(self) -> None:
        """Print a summary of the test results to the console."""
        results_df = self.get_results_dataframe()
        if results_df.empty:
            logger.warning("No results available for summary")
            return
            
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
            print(f"{row['algorithm'].upper():<10} {row['space_saved']:<20.4f} {row['processing_time']:<20.4f} "
                  f"{row['efficiency']:<20.4f} {row['scheduling_time']:<20.4f} {row['completion_rate']:<15.2%}")
        print("=" * 80)


def main() -> None:
    """Main function to run the scheduler tester with memory optimization."""
    parser = argparse.ArgumentParser(description='Test image compression scheduling algorithms')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR, 
                       help='Directory with source images')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, 
                       help='Directory for results and charts')
    parser.add_argument('--quality', type=int, default=DEFAULT_QUALITY, 
                       help='Compression quality (1-100, lower = more compression)')
    parser.add_argument('--sample_percentage', type=int, default=DEFAULT_SAMPLE_PERCENTAGE,
                       help='Percentage of image to process for estimation (1-100)')
    parser.add_argument('--create_datasets', type=str, choices=['True', 'False'], default='True',
                       help='Whether to create test datasets (True, False)')
    parser.add_argument('--quick_test', type=str, choices=['True', 'False'], default='False',
                       help='Run a limited test with fewer time budgets (True, False)')
    
    args = parser.parse_args()
    
    # Convert string arguments to proper types
    create_datasets = args.create_datasets.lower() == 'true'
    quick_test = args.quick_test.lower() == 'true'
    
    logger.info("Starting image compression scheduler testing")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Quality: {args.quality}")
    logger.info(f"Sample percentage: {args.sample_percentage}")
    logger.info(f"Create datasets: {create_datasets}")
    logger.info(f"Quick test: {quick_test}")
    
    try:
        # Create tester
        tester = SchedulerTester(
            args.input_dir, 
            args.output_dir, 
            args.quality, 
            args.sample_percentage,
            create_datasets=create_datasets,
            quick_test=quick_test
        )
        
        # Run tests
        start_time = time.time()
        tester.test_all_algorithms()
        end_time = time.time()
        
        # Save results
        tester.save_results()
        
        # Print summary
        tester.print_summary()
        
        logger.info(f"Testing completed in {end_time - start_time:.4f} seconds")
        logger.info(f"Check {args.output_dir} for results and charts")
        
        print(f"\nTesting completed in {end_time - start_time:.4f} seconds")
        print(f"Check {args.output_dir} for results and charts")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        print(f"Error during testing: {e}")
        sys.exit(1)
    finally:
        # Final memory cleanup
        gc.collect()


if __name__ == "__main__":
    main()