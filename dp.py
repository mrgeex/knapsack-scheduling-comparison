"""
Dynamic Programming scheduler for image compression.

This module implements a dynamic programming algorithm for scheduling image processing.
The algorithm aims to maximize total value (file size) within a given time budget.
"""
def schedule(image_info, time_budget=None):
  """
  Implements a dynamic programming scheduling algorithm for image compression.

  If time_budget is provided, uses a knapsack approach to select images that maximize
  total value (file size) while staying within the time budget.
  Otherwise, uses Smith's rule (weighted shortest processing time first).

  Args:
  image_info: List of tuples (filename, file_size, processing_time)
  time_budget: Optional time constraint in seconds
      
  Returns:
  List of filenames in the order they should be processed
  """
  
  if time_budget is None or time_budget <= 0:
    # No time budget constraint - use original Smith's rule
    items = []
    # Create items with weights (file size) and processing times
    for i, (filename, filesize, proc_time) in enumerate(image_info):
      # Weight is the file size (larger files are more important)
      weight = filesize
      # Processing time is the measured time
      proc_time = max(proc_time, 0.001)  # Avoid division by zero
      # Calculate ratio of weight to processing time (Smith's rule)
      ratio = weight / proc_time

      items.append((i, filename, filesize, proc_time, ratio))

    # Sort items by weight/processing time ratio (highest first)
    # This is Smith's rule, which is optimal for minimizing weighted completion time
    sorted_items = sorted(items, key=lambda x: x[4], reverse=True)

    # Extract filenames in the sorted order
    ordered_filenames = [item[1] for item in sorted_items]

    print(f"\nScheduling order ({len(image_info)} image(s)):")
    for i, filename, _a, _b, ratio in sorted_items:
      print(f"{i+1}. {filename} (Ratio: {ratio:.2f})")

    return ordered_filenames
  
  else:
    # Apply knapsack algorithm with time budget constraint
    n = len(image_info)
    
    # Convert processing times to integers (multiply by 100 for 0.01s precision)
    scale_factor = 100
    scaled_budget = int(time_budget * scale_factor)
    
    # Create items with scaled processing times
    scaled_items = []
    for i, (filename, filesize, proc_time) in enumerate(image_info):
      scaled_time = max(int(proc_time * scale_factor), 1)  # Ensure minimum of 1
      scaled_items.append((i, filename, filesize, proc_time, scaled_time))
    
    # Create 2D DP table: dp[i][j] = maximum value (filesize) that can be achieved
    # with first i items and time budget j
    dp = [[0 for _ in range(scaled_budget + 1)] for _ in range(n + 1)]
    
    # Build the DP table
    for i in range(1, n + 1):
      for j in range(scaled_budget + 1):
        # Current item's properties (0-indexed in items list)
        _, filename, filesize, proc_time, scaled_time = scaled_items[i-1]
        
        # If current item's time exceeds remaining budget, skip it
        if scaled_time > j:
          dp[i][j] = dp[i-1][j]
        else:
          # Choose maximum of: 
          # 1. Not including current item
          # 2. Including current item + best solution with remaining budget
          dp[i][j] = max(dp[i-1][j], dp[i-1][j-scaled_time] + filesize)
    
    # Trace back to find selected items
    selected = []
    j = scaled_budget
    
    for i in range(n, 0, -1):
      if dp[i][j] != dp[i-1][j]:
        # This item was selected
        idx, filename, filesize, proc_time, scaled_time = scaled_items[i-1]
        selected.append((idx, filename, filesize, proc_time))
        j -= scaled_time
    
    # Sort selected items by their original index to maintain relative order
    selected.sort()
    
    # Extract filenames of selected items
    ordered_filenames = [item[1] for item in selected]
    
    # Calculate total filesize and processing time
    total_filesize = sum(item[2] for item in selected)
    total_proc_time = sum(item[3] for item in selected)
    
    print(f"\nKnapsack Scheduling ({len(ordered_filenames)} of {n} image(s)):")
    print(f"Time Budget: {time_budget:.2f}s, Used: {total_proc_time:.2f}s")
    print(f"Total compressed size: {total_filesize:.2f}KB")
    
    for idx, (_, filename, filesize, proc_time) in enumerate(selected):
      print(f"{idx+1}. {filename} ({proc_time:.2f}s, {filesize:.2f}KB)")
    
    return ordered_filenames