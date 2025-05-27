def schedule(image_info, time_budget=None):
    """
    Implements a dynamic programming scheduling algorithm for image compression.
    
    If time_budget is provided, uses a knapsack approach to select images that maximize
    total space saved (original size - compressed size) while staying within the time budget.
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
        for i, (filename, filesize, proc_time) in enumerate(image_info):
            proc_time = max(proc_time, 0.001)
            ratio = filesize / proc_time
            items.append((i, filename, filesize, proc_time, ratio))
        
        sorted_items = sorted(items, key=lambda x: x[4], reverse=True)
        ordered_filenames = [item[1] for item in sorted_items]
        
        print(f"\nScheduling order ({len(image_info)} image(s)):")
        for i, filename, _a, _b, ratio in sorted_items:
            print(f"{i+1}. {filename} (Ratio: {ratio:.2f})")
        
        return ordered_filenames
    
    else:
        # Apply knapsack algorithm with time budget constraint
        n = len(image_info)
        
        # Convert processing times to integers for better precision
        scale_factor = 100
        scaled_budget = int(time_budget * scale_factor)
        
        # Create items with scaled processing times
        scaled_items = []
        for i, (filename, filesize, proc_time) in enumerate(image_info):
            compressed_size = filesize * 0.5  # Assuming 50% compression ratio (adjust as needed)
            space_saved = filesize - compressed_size
            scaled_time = max(int(proc_time * scale_factor), 1)
            scaled_items.append((i, filename, space_saved, proc_time, scaled_time))
        
        # Create 2D DP table for knapsack problem
        dp = [[0 for _ in range(scaled_budget + 1)] for _ in range(n + 1)]
        
        # Build the DP table
        for i in range(1, n + 1):
            for j in range(scaled_budget + 1):
                _, filename, space_saved, proc_time, scaled_time = scaled_items[i-1]
                
                if scaled_time > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-scaled_time] + space_saved)
        
        # Trace back to find selected items
        selected = []
        j = scaled_budget
        for i in range(n, 0, -1):
            if dp[i][j] != dp[i-1][j]:
                idx, filename, space_saved, proc_time, scaled_time = scaled_items[i-1]
                selected.append((idx, filename, space_saved, proc_time))
                j -= scaled_time
        
        selected.sort()  # Sort selected items by their original index
        
        ordered_filenames = [item[1] for item in selected]
        
        total_space_saved = sum(item[2] for item in selected)
        total_proc_time = sum(item[3] for item in selected)
        
        print(f"\nKnapsack Scheduling ({len(ordered_filenames)} of {n} image(s)): ")
        print(f"Time Budget: {time_budget:.2f}s, Used: {total_proc_time:.2f}s")
        print(f"Total space saved: {total_space_saved:.2f}KB")
        
        for idx, (_, filename, space_saved, proc_time) in enumerate(selected):
            print(f"{idx+1}. {filename} (Space Saved: {space_saved:.2f}KB, Time: {proc_time:.2f}s)")
        
        return ordered_filenames
