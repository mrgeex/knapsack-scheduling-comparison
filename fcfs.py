import compressor

def schedule(image_info, time_budget=None):
    """
    First-Come, First-Served scheduling algorithm.
    
    If time_budget is provided, selects images in order until the budget is reached.
    Otherwise, processes all images in the order they were found.
    
    Args:
    image_info: List of tuples (filename, original_size, compressed_size, space_saved, processing_time)
    time_budget: Optional time constraint in seconds
        
    Returns:
    List of filenames in the order they should be processed
    """
    
    if time_budget is None or time_budget <= 0:
        # No time budget constraint - process all files in order (original FCFS)
        ordered_filenames = [info[0] for info in image_info]
        
        # Calculate total space saved
        total_space_saved = sum(info[3] for info in image_info)
        
        print(f"\nScheduling order ({len(image_info)} image(s)):")
        print(f"Total space saved: {total_space_saved:.2f}KB")
        for i, (filename, _, _, space_saved, _) in enumerate(image_info):
            print(f"{i+1}. {filename} (Space saved: {space_saved:.2f}KB)")
        
        return ordered_filenames
    
    else:
        # Apply time budget constraint - select files in order until budget is reached
        selected = []
        total_time = 0
        total_space_saved = 0
        
        # Process items in original order (FCFS with budget constraint)
        for i, (filename, _, _, space_saved, proc_time) in enumerate(image_info):
            if total_time + proc_time <= time_budget:
                selected.append((i, filename))
                total_time += proc_time
                total_space_saved += space_saved
        
        ordered_filenames = [item[1] for item in selected]
        
        print(f"\nScheduling order ({len(ordered_filenames)} of {len(image_info)} image(s)):")
        print(f"Time Budget: {time_budget:.2f}s, Used: {total_time:.2f}s")
        print(f"Total space saved: {total_space_saved:.2f}KB")
        for i, (_, filename) in enumerate(selected):
            # Find the corresponding space saved for this filename
            for fname, _, _, space_saved, proc_time in image_info:
                if fname == filename:
                    print(f"{i+1}. {filename} (Processing Time: {proc_time:.2f}s, Space saved: {space_saved:.2f}KB)")
                    break
        
        return ordered_filenames