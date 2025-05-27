"""
Greedy scheduler for image compression.

This module implements a greedy algorithm for scheduling image processing.
The algorithm orders images by measured processing time (shortest first).
If a time budget is provided, it uses a greedy knapsack approach.
"""
def schedule(image_info, time_budget=None):
  """
  Implements a greedy scheduling algorithm for image compression.
  
  If time_budget is provided, uses a greedy knapsack approach to select images that 
  maximize total value (disk space saved) while staying within the time budget.
  Otherwise, simply orders by processing time (shortest first).
  
  Args:
  image_info: List of tuples (filename, original_size, compressed_size, space_saved, processing_time)
  time_budget: Optional time constraint in seconds
      
  Returns:
  List of filenames in the order they should be processed
  """

  if time_budget is None or time_budget <= 0:
    # No time budget constraint - sort by processing time
    # Sort images by processing time (Ascending)
    sorted_images = sorted(image_info, key=lambda x: x[4])

    # Extract filenames in the sorted order
    ordered_filenames = [info[0] for info in sorted_images]

    print(f"\nScheduling order ({len(image_info)} image(s)):")
    for i, (filename, _, _, space_saved, proc_time) in enumerate(sorted_images):
      print(f"{i+1}. {filename} (Processing Time: {proc_time:.2f}s, Space saved: {space_saved:.2f}KB)")

    return ordered_filenames
  
  else:
    # Apply greedy knapsack algorithm with time budget constraint
    # For greedy, we'll use value/weight ratio (space saved / processing time)
    items = []
    for i, (filename, _, _, space_saved, proc_time) in enumerate(image_info):
      # Ensure processing time is not zero
      proc_time = max(proc_time, 0.001)
      # Calculate ratio of space saved to processing time
      ratio = space_saved / proc_time
      items.append((i, filename, space_saved, proc_time, ratio))
    
    # Sort by value/weight ratio (greedy approach to knapsack)
    sorted_items = sorted(items, key=lambda x: x[4], reverse=True)
    
    # Select items until we exceed the time budget
    selected = []
    total_time = 0
    total_space_saved = 0
    
    for i, filename, space_saved, proc_time, ratio in sorted_items:
      if total_time + proc_time <= time_budget:
        selected.append((i, filename, space_saved, proc_time, ratio))
        total_time += proc_time
        total_space_saved += space_saved
    
    # Sort selected items by their original index to maintain relative order
    selected.sort()
    
    # Extract filenames of selected items
    ordered_filenames = [item[1] for item in selected]
    
    print(f"\nGreedy Knapsack Scheduling ({len(ordered_filenames)} of {len(image_info)} image(s)):")
    print(f"Time Budget: {time_budget:.2f}s, Used: {total_time:.2f}s")
    print(f"Total space saved: {total_space_saved:.2f}KB")
    
    for idx, (_, filename, space_saved, proc_time, ratio) in enumerate(selected):
      print(f"{idx+1}. {filename} (Value/Time Ratio: {ratio:.2f}, Space saved: {space_saved:.2f}KB)")
    
    return ordered_filenames