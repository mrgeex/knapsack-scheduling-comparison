import math
import numpy as np
import os
import tempfile

def schedule(image_info, time_budget=None):
    """
    Implements a dynamic programming scheduling algorithm for image compression.
    
    If time_budget is None, calculates the total processing time and uses it as the
    budget for DP, scheduling with the knapsack algorithm.
    If time_budget is provided, uses a 2D knapsack approach to maximize disk space saved
    while staying within the time budget, with memory mapping for large problems.
    
    Args:
    image_info: List of tuples (filename, original_size, compressed_size, space_saved, processing_time)
    time_budget: Optional time constraint in seconds
        
    Returns:
    List of filenames in the order they should be processed
    """
    
    # Calculate total available processing time
    total_proc_time = sum(info[4] for info in image_info)
    
    if time_budget is None or time_budget <= 0:
        # No time budget provided - use total processing time as budget
        time_budget = math.ceil(total_proc_time)
        print(f"\nNo time budget specified. Using total processing time: {time_budget:.4f}s")
    
    print(f"Using DP with time budget: {time_budget:.4f}s")
    
    # Auto-tune scale factor based on the minimum non-zero processing time
    min_proc_time = float('inf')
    for _, _, _, _, proc_time in image_info:
        if 0 < proc_time < min_proc_time:
            min_proc_time = proc_time
    
    # If all proc_times are zero (unlikely), set a default
    if min_proc_time == float('inf'):
        min_proc_time = 0.01
    
    # Calculate the smallest time difference to preserve
    # This helps avoid unnecessarily large scale factors
    time_diffs = []
    proc_times = [info[4] for info in image_info if info[4] > 0]
    proc_times.sort()
    
    # Find the smallest non-zero difference between consecutive times
    for i in range(1, len(proc_times)):
        diff = proc_times[i] - proc_times[i-1]
        if diff > 0:
            time_diffs.append(diff)
    
    # If we found differences, use the smallest one to determine scale
    if time_diffs:
        min_diff = min(time_diffs)
        # Use the smaller of min_time and min_diff for precision
        min_precision_needed = min(min_proc_time, min_diff)
    else:
        min_precision_needed = min_proc_time
    
    # Set scale factor based on minimum precision needed
    # This ensures we have at least integer precision for the smallest time
    scale_factor = int(1 / min_precision_needed)
    
    # Ensure scale factor is at least 100 for reasonable precision
    # But cap it to avoid overly large tables
    scale_factor = max(min(scale_factor, 10000), 100)
    
    # Calculate the scaled budget with our dynamic scale factor
    scaled_budget = int(time_budget * scale_factor)
    
    # Parameters for memory considerations
    bytes_per_float = 8  # 8 bytes per float64
    memory_limit_gb = 1.0  # 1GB limit for in-memory approach
    
    # Calculate memory usage for 2D table
    n_items = len(image_info)
    memory_usage_gb = (n_items + 1) * (scaled_budget + 1) * bytes_per_float / (1024**3)
    
    # Decide if we need memory mapping based on calculated size
    use_memmap = memory_usage_gb > memory_limit_gb
    
    print(f"Auto-tuned scale factor: {scale_factor}")
    if time_diffs:
        print(f"Minimum processing time: {min_proc_time:.6f}s")
        print(f"Minimum time difference: {min(time_diffs):.6f}s")
        print(f"Using precision: {min_precision_needed:.6f}s")
    else:
        print(f"Minimum processing time: {min_proc_time:.6f}s")
        print(f"Using precision: {min_precision_needed:.6f}s")
    print(f"Estimated DP table size: {memory_usage_gb:.4f}GB")
    
    if use_memmap:
        print(f"Using memory mapping due to large DP table size ({memory_usage_gb:.4f}GB)")
        return _schedule_with_memmap(image_info, time_budget, scale_factor, scaled_budget)
    else:
        print(f"Using in-memory DP approach ({memory_usage_gb:.4f}GB)")
        return _schedule_in_memory(image_info, time_budget, scale_factor, scaled_budget)


def _prepare_ratio_items(image_info, scale_factor):
    """
    Prepare and sort items by value-to-weight ratio for DP knapsack.
    
    Args:
    image_info: List of tuples (filename, original_size, compressed_size, space_saved, processing_time)
    scale_factor: Factor to scale the processing times
    
    Returns:
    List of tuples (original_index, filename, space_saved, proc_time, scaled_time, ratio)
    """
    ratio_items = []
    for i, (filename, original_size, compressed_size, space_saved, proc_time) in enumerate(image_info):
        proc_time = max(proc_time, 0.0000001)  # Avoid division by zero
        ratio = space_saved / proc_time
        
        # Scale processing time using auto-tuned scale factor
        scaled_time = max(int(math.ceil(proc_time * scale_factor)), 1)
        
        ratio_items.append((i, filename, space_saved, proc_time, scaled_time, ratio))
    
    # Sort by ratio (descending) for better DP solution
    ratio_items.sort(key=lambda x: x[5], reverse=True)
    
    return ratio_items


def _schedule_in_memory(image_info, time_budget, scale_factor, scaled_budget):
    """
    Implements the in-memory version of the knapsack algorithm with a classic 2D table.
    
    Args:
    image_info: List of tuples (filename, original_size, compressed_size, space_saved, processing_time)
    time_budget: Time constraint in seconds
    scale_factor: Factor to scale the processing times
    scaled_budget: Scaled time budget (time_budget * scale_factor)
    
    Returns:
    List of filenames in the order they should be processed
    """
    # Process items in order of value-to-weight ratio (efficiency) for better results
    ratio_items = _prepare_ratio_items(image_info, scale_factor)
    
    # Number of items
    n = len(ratio_items)
    
    # Initialize 2D DP table
    dp = np.zeros((n + 1, scaled_budget + 1), dtype=np.float64)
    
    # Initialize 2D table to track items
    keep = np.zeros((n + 1, scaled_budget + 1), dtype=bool)
    
    # Build DP table - classic 2D approach
    for i in range(1, n + 1):
        item_idx, filename, space_saved, proc_time, scaled_time, ratio = ratio_items[i-1]
        
        for j in range(scaled_budget + 1):
            # If item doesn't fit, use previous value
            if scaled_time > j:
                dp[i, j] = dp[i-1, j]
                keep[i, j] = False
            else:
                # Max of including or excluding this item
                if dp[i-1, j] < dp[i-1, j-scaled_time] + space_saved:
                    dp[i, j] = dp[i-1, j-scaled_time] + space_saved
                    keep[i, j] = True
                else:
                    dp[i, j] = dp[i-1, j]
                    keep[i, j] = False
    
    # Find max value and select the maximum time budget solution
    max_value = dp[n].max()
    j_candidates = np.where(dp[n] == max_value)[0]
    j = j_candidates.max()  # Use maximum time when multiple options give same value
    
    # Reconstruct solution
    selected_indices = []
    
    # Trace back through the keep table
    for i in range(n, 0, -1):
        if keep[i, j]:
            selected_indices.append(i-1)  # Index in ratio_items
            j -= ratio_items[i-1][4]  # Subtract scaled_time
    
    # Convert selected indices to actual items
    selected_items = [ratio_items[i] for i in selected_indices]
    
    # Extract filenames and calculate totals
    ordered_filenames = [item[1] for item in selected_items]
    total_proc_time = sum(item[3] for item in selected_items)
    total_space_saved = sum(item[2] for item in selected_items)
    
    # Print results
    print(f"\nKnapsack Scheduling ({len(ordered_filenames)} of {len(image_info)} image(s)):")
    print(f"Time Budget: {time_budget:.4f}s, Used: {total_proc_time:.4f}s ({total_proc_time/time_budget:.1%})")
    print(f"Total space saved: {total_space_saved:.4f}KB")
    
    for idx, (_, filename, space_saved, proc_time, _, _) in enumerate(selected_items):
        print(f"{idx+1}. {filename} ({proc_time:.4f}s, Space saved: {space_saved:.4f}KB)")
    
    return ordered_filenames


def _schedule_with_memmap(image_info, time_budget, scale_factor, scaled_budget):
    """
    Implements the memory-mapped version of the knapsack algorithm with a classic 2D table.
    
    Args:
    image_info: List of tuples (filename, original_size, compressed_size, space_saved, processing_time)
    time_budget: Time constraint in seconds
    scale_factor: Factor to scale the processing times
    scaled_budget: Scaled time budget (time_budget * scale_factor)
    
    Returns:
    List of filenames in the order they should be processed
    """
    # Process items in order of value-to-weight ratio (efficiency) for better results
    ratio_items = _prepare_ratio_items(image_info, scale_factor)
    
    # Number of items
    n = len(ratio_items)
    
    # Create temporary file for memory-mapped array
    # This will be automatically deleted when closed
    keep_file = tempfile.NamedTemporaryFile(delete=False)
    
    try:
        # Only need to keep two rows in memory at a time for 2D DP
        # One for current row and one for previous row
        prev_row = np.zeros(scaled_budget + 1, dtype=np.float64)
        curr_row = np.zeros(scaled_budget + 1, dtype=np.float64)
        
        # Use memory-mapped array to keep track of which items to keep
        keep = np.memmap(keep_file.name, dtype=bool, mode='w+', 
                         shape=(n + 1, scaled_budget + 1))
        keep[:] = False
        
        # Build DP table row by row
        for i in range(1, n + 1):
            item_idx, filename, space_saved, proc_time, scaled_time, ratio = ratio_items[i-1]
            
            # Swap rows
            prev_row, curr_row = curr_row, prev_row
            
            # Process current row
            for j in range(scaled_budget + 1):
                # If item doesn't fit, use previous value
                if scaled_time > j:
                    curr_row[j] = prev_row[j]
                    keep[i, j] = False
                else:
                    # Max of including or excluding this item
                    if prev_row[j] < prev_row[j-scaled_time] + space_saved:
                        curr_row[j] = prev_row[j-scaled_time] + space_saved
                        keep[i, j] = True
                    else:
                        curr_row[j] = prev_row[j]
                        keep[i, j] = False
            
            # Flush changes to disk after each row
            keep.flush()
        
        # Find max value and select the maximum time budget solution
        max_value = curr_row.max()
        j_candidates = np.where(curr_row == max_value)[0]
        j = j_candidates.max()  # Use maximum time when multiple options give same value
        
        # Reconstruct solution
        selected_indices = []
        
        # Trace back through the keep array
        for i in range(n, 0, -1):
            if keep[i, j]:
                selected_indices.append(i-1)  # Index in ratio_items
                j -= ratio_items[i-1][4]  # Subtract scaled_time
        
        # Convert selected indices to actual items
        selected_items = [ratio_items[i] for i in selected_indices]
        
        # Extract filenames and calculate totals
        ordered_filenames = [item[1] for item in selected_items]
        total_proc_time = sum(item[3] for item in selected_items)
        total_space_saved = sum(item[2] for item in selected_items)
        
        # Print results
        print(f"\nKnapsack Scheduling with Memmap ({len(ordered_filenames)} of {len(image_info)} image(s)):")
        print(f"Time Budget: {time_budget:.4f}s, Used: {total_proc_time:.4f}s ({total_proc_time/time_budget:.1%})")
        print(f"Total space saved: {total_space_saved:.4f}KB")
        
        for idx, (_, filename, space_saved, proc_time, _, _) in enumerate(selected_items):
            print(f"{idx+1}. {filename} ({proc_time:.4f}s, Space saved: {space_saved:.4f}KB)")
        
        return ordered_filenames
        
    finally:
        # Clean up temp file
        keep_file.close()
        try:
            os.unlink(keep_file.name)
        except:
            pass