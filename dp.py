"""
Dynamic Programming scheduler for image compression.

This module implements a dynamic programming algorithm for scheduling image processing.
The algorithm aims to maximize total value (disk space saved) within a given time budget.
Uses NumPy's memmap to handle large datasets with minimal memory usage.
"""
import os
import tempfile
import numpy as np

def schedule(image_info, time_budget=None):
  """
  Implements a dynamic programming scheduling algorithm for image compression.

  If time_budget is provided, uses a knapsack approach to select images that maximize
  total value (disk space saved) while staying within the time budget.
  Otherwise, uses Smith's rule (weighted shortest processing time first).

  Args:
  image_info: List of tuples (filename, original_size, compressed_size, space_saved, processing_time)
  time_budget: Optional time constraint in seconds
      
  Returns:
  List of filenames in the order they should be processed
  """
  
  if time_budget is None or time_budget <= 0:
    # No time budget constraint - use original Smith's rule
    items = []
    # Create items with weights (space saved) and processing times
    for i, (filename, original_size, compressed_size, space_saved, proc_time) in enumerate(image_info):
      # Weight is the space saved (more space saved is more important)
      weight = space_saved
      # Processing time is the measured time
      proc_time = max(proc_time, 0.0000001)  # Avoid division by zero with much smaller value
      # Calculate ratio of weight to processing time (Smith's rule)
      ratio = weight / proc_time

      items.append((i, filename, space_saved, proc_time, ratio))

    # Sort items by weight/processing time ratio (highest first)
    # This is Smith's rule, which is optimal for minimizing weighted completion time
    sorted_items = sorted(items, key=lambda x: x[4], reverse=True)

    # Extract filenames in the sorted order
    ordered_filenames = [item[1] for item in sorted_items]

    print(f"\nScheduling order ({len(image_info)} image(s)):")
    for i, filename, space_saved, _proc_time, ratio in sorted_items:
      print(f"{i+1}. {filename} (Space saved: {space_saved:.2f}KB, Ratio: {ratio:.2f})")

    return ordered_filenames
  
  else:
    # Apply knapsack algorithm with time budget constraint using memmap to save memory
    n = len(image_info)
    
    # Convert processing times to integers (multiply by 1000 for 0.001s precision)
    scale_factor = 1000  # Increased from 100 to 1000 for better precision
    scaled_budget = int(time_budget * scale_factor)
    
    # Create items with scaled processing times
    scaled_items = []
    for i, (filename, original_size, compressed_size, space_saved, proc_time) in enumerate(image_info):
      scaled_time = max(int(proc_time * scale_factor), 1)  # Ensure minimum of 1
      scaled_items.append((i, filename, space_saved, proc_time, scaled_time))
    
    # Calculate memory requirements
    estimated_size_bytes = (n + 1) * (scaled_budget + 1) * 4  # 4 bytes per float32
    estimated_size_mb = estimated_size_bytes / (1024 * 1024)
    
    # Adjust scale factor if the memory requirements are too high
    if estimated_size_mb > 500:  # Limit to 500MB to be safe
      original_scale = scale_factor
      scale_factor = int((500 * 1024 * 1024) / ((n + 1) * (scaled_budget + 1) * 4) * scale_factor)
      scale_factor = max(scale_factor, 1)  # Ensure at least 1
      scaled_budget = int(time_budget * scale_factor)
      print(f"Memory optimization: Reduced scale factor from {original_scale} to {scale_factor}")
      print(f"New estimated memory usage: {(n + 1) * (scaled_budget + 1) * 4 / (1024 * 1024):.2f} MB")
    
    # Create temporary file for memory mapping
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_filename = temp_file.name
    temp_file.close()
    
    try:
      # Create memory-mapped array for DP table
      dp = np.memmap(temp_filename, dtype=np.float32, mode='w+', shape=(n + 1, scaled_budget + 1))
      
      # Initialize the DP table with zeros
      dp[:] = 0
      dp.flush()
      
      # Build the DP table - process row by row to minimize memory usage
      for i in range(1, n + 1):
        _, filename, space_saved, proc_time, scaled_time = scaled_items[i-1]
        
        # Process the current row
        for j in range(scaled_budget + 1):
          if scaled_time > j:
            dp[i, j] = dp[i-1, j]
          else:
            dp[i, j] = max(dp[i-1, j], dp[i-1, j-scaled_time] + space_saved)
        
        # Flush changes to disk after each row
        dp.flush()
      
      # Trace back to find selected items
      selected = []
      j = scaled_budget
      
      for i in range(n, 0, -1):
        if dp[i, j] != dp[i-1, j]:
          # This item was selected
          idx, filename, space_saved, proc_time, scaled_time = scaled_items[i-1]
          selected.append((idx, filename, space_saved, proc_time))
          j -= scaled_time
      
      # Sort selected items by their original index to maintain relative order
      selected.sort()
      
      # Extract filenames of selected items
      ordered_filenames = [item[1] for item in selected]
      
      # Calculate total space saved and processing time
      total_space_saved = sum(item[2] for item in selected)
      total_proc_time = sum(item[3] for item in selected)
      
      print(f"\nKnapsack Scheduling ({len(ordered_filenames)} of {n} image(s)):")
      print(f"Time Budget: {time_budget:.2f}s, Used: {total_proc_time:.2f}s")
      print(f"Total space saved: {total_space_saved:.2f}KB")
      
      for idx, (_, filename, space_saved, proc_time) in enumerate(selected):
        print(f"{idx+1}. {filename} ({proc_time:.2f}s, Space saved: {space_saved:.2f}KB)")
      
      return ordered_filenames
    
    finally:
      # Clean up the memory map and temporary file
      if 'dp' in locals():
        del dp
      
      # Remove the temporary file
      if os.path.exists(temp_filename):
        try:
          os.unlink(temp_filename)
        except Exception as e:
          print(f"Warning: Failed to delete temporary file {temp_filename}: {e}")