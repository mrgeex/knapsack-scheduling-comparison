"""
Image Compression Scheduler

This application schedules image compression using different algorithms.
The actual compression is handled by the compress_images.py module.
"""

import os
import compressor, fcfs, greedy, dp
import time
import subprocess
import platform

# Configure these settings:
INPUT_FOLDER = r"./test"
OUTPUT_FOLDER = r"./after"
QUALITY = 50
SAMPLE_PERCENTAGE = 10  # Percentage of image to process for estimation

def clear_terminal():
    """Clear the terminal screen based on the operating system"""
    if platform.system() == "Windows":
        subprocess.call('cls', shell=True)
    else:
        subprocess.call('clear', shell=True)

def process_with_scheduler(scheduler_name, image_info, input_folder, output_folder, quality, 
                           processing_times=None, time_budget=None, compressed_files=None,
                           sample_percentage=SAMPLE_PERCENTAGE):
    """Process images using the selected scheduling algorithm"""
    
    print(f"\nUsing {scheduler_name.upper()} Scheduling Algorithm")
    
    # Determine the ordered filenames based on the scheduling algorithm
    if scheduler_name == "fcfs":
        if time_budget is not None:
            items = [(i, info[0], info[4]) for i, info in enumerate(image_info)]
            selected = []
            total_time = 0
            total_space_saved = 0
            
            for i, filename, proc_time in items:
                if total_time + proc_time <= time_budget:
                    selected.append((i, filename))
                    total_time += proc_time
                    for fname, _, _, space_saved, _ in image_info:
                        if fname == filename:
                            total_space_saved += space_saved
                            break
            
            ordered_filenames = [item[1] for item in selected]
            
            print(f"\nScheduling order ({len(ordered_filenames)} of {len(image_info)} image(s)):")
            print(f"Time Budget: {time_budget:.2f}s, Used: {total_time:.2f}s")
            print(f"Total space saved: {total_space_saved:.2f}KB")
            for i, filename in enumerate(ordered_filenames):
                print(f"{i+1}. {filename}")
        else:
            ordered_filenames = fcfs.schedule(image_info)
            total_space_saved = sum(info[3] for info in image_info)
  
    elif scheduler_name == "greedy":
        if time_budget is not None:
            print(f"Greedy Knapsack (Time Budget: {time_budget:.2f}s)")
            ordered_filenames = greedy.schedule(image_info, time_budget)
        else:
            print("Shortest Processing Time First")
            ordered_filenames = greedy.schedule(image_info)
            total_space_saved = sum(info[3] for info in image_info)
  
    elif scheduler_name == "dp":
        if time_budget is not None:
            print(f"DP Knapsack (Time Budget: {time_budget:.2f}s)")
            ordered_filenames = dp.schedule(image_info, time_budget)
        else:
            print("Highest Ratio First (Smith's Rule)")
            ordered_filenames = dp.schedule(image_info)
            total_space_saved = sum(info[3] for info in image_info)
    else:
        print(f"Error: Unknown scheduling algorithm {{{scheduler_name}}}")
        return 0, {}

    # If sample_percentage = 100 and files were already processed, just show results without recompressing
    if sample_percentage == 100 and compressed_files:
        # Filter ordered_filenames to only include files that were previously compressed
        processed_filenames = [f for f in ordered_filenames if f in compressed_files]
        
        # Calculate total time from our existing measurements
        total_time = sum(processing_times.get(filename, 0) for filename in processed_filenames)
        
        # Calculate total space saved
        total_space_saved = 0
        for filename in processed_filenames:
            for fname, _, _, space_saved, _ in image_info:
                if fname == filename:
                    total_space_saved += space_saved
                    break
                    
        print("\nScheduling Completed (using pre-compressed files).")
        print(f"Files to compress: {len(processed_filenames)}")
        print(f"Estimated Total Processing Time: {total_time:.2f} seconds")
        print(f"Total Space Saved: {total_space_saved:.2f} KB")
        
        return total_time, processing_times
    else:
        # Otherwise, actually compress the files
        processed, skipped, processing_time, measured_times = compressor.compress_images(
            input_folder,
            output_folder,
            quality,
            ordered_filenames,
            processing_times
        )

        if time_budget is not None and scheduler_name != "fcfs":
            total_space_saved = 0
            for filename in ordered_filenames:
                for fname, _, _, space_saved, _ in image_info:
                    if fname == filename:
                        total_space_saved += space_saved
                        break

        print("\nScheduling Completed.")
        print(f"Successful compression: {processed}")
        print(f"Total Processing Time: {processing_time:.2f} seconds")
        print(f"Total Space Saved: {total_space_saved:.2f} KB")

        return processing_time, measured_times

def get_time_budget():
    """Prompt the user to set a time budget for scheduling"""
    while True:
        try:
            time_budget = float(input("\nEnter time budget in seconds (0 for no constraint): "))
            if time_budget < 0:
                print("Error: Time budget must be non-negative!")
                continue
            if time_budget == 0:
                return None
            return time_budget
        except ValueError:
            print("Error: Please enter a valid number!")

def get_sample_percentage():
    """Prompt the user to set a sample percentage for estimation"""
    while True:
        try:
            percentage = int(input("\nEnter sample percentage for estimation (1-100, higher = more accurate): "))
            if percentage < 1 or percentage > 100:
                print("Error: Percentage must be between 1 and 100!")
                continue
            return percentage
        except ValueError:
            print("Error: Please enter a valid number!")

def run_estimation(input_folder, output_folder, quality, sample_percentage):
    """Run the estimation phase using partial processing"""
    print(f"Running estimation using {sample_percentage}% partial processing...")
    
    # If sample_percentage is 100%, do full processing and save the outputs
    if sample_percentage == 100:
        print("Using full processing (100%) - this will take the same time as initialization")
        print("But files will be compressed during this process and won't need recompression")
        
        # Get basic image info
        image_info = compressor.get_image_info(input_folder)
        
        if not image_info:
            print("Error: No valid images found in the folder!")
            return None, None, 0, 0, 0, 0, {}
            
        # Do full compression and measure times
        start_time = time.time()
        processed, skipped, _, measured_times = compressor.compress_images(
            input_folder, output_folder, quality
        )
        total_time = time.time() - start_time
        
        # Calculate sizes and update with actual times and sizes
        total_original_size = sum(info[1] for info in image_info)
        
        # Get compressed sizes and create a list of compressed files
        total_compressed_size = 0
        compressed_files = []
        for file_name in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file_name)
            if os.path.isfile(file_path):
                compressed_files.append(file_name)
                total_compressed_size += os.path.getsize(file_path) / 1024
        
        # Calculate space saved
        total_space_saved = total_original_size - total_compressed_size
        
        # Update image info with measured times and compressed sizes
        updated_image_info = []
        for filename, original_size, _ in image_info:
            # Get compressed file size
            compressed_size = 0
            output_path = os.path.join(output_folder, filename)
            if os.path.isfile(output_path):
                compressed_size = os.path.getsize(output_path) / 1024
            
            # Calculate space saved
            space_saved = original_size - compressed_size
            
            # Add processing time
            proc_time = measured_times.get(filename, 0)
            
            # Updated format: (filename, original_size, compressed_size, space_saved, proc_time)
            updated_image_info.append((filename, original_size, compressed_size, space_saved, proc_time))
        
        return updated_image_info, measured_times, total_original_size, total_compressed_size, total_space_saved, total_time, compressed_files
    
    else:
        # For partial processing, use the estimation function
        image_info = compressor.estimate_image_info(input_folder, quality, sample_percentage)
        
        if not image_info:
            print("Error: No valid images found in the folder!")
            return None, None, 0, 0, 0, 0, {}
        
        # Calculate summary statistics
        total_original_size = sum(info[1] for info in image_info)
        total_estimated_compressed_size = sum(info[2] for info in image_info)
        total_estimated_space_saved = sum(info[3] for info in image_info)
        total_estimated_time = sum(info[4] for info in image_info)
        
        # Create a dictionary of estimated processing times
        estimated_times = {info[0]: info[4] for info in image_info}
        
        return image_info, estimated_times, total_original_size, total_estimated_compressed_size, total_estimated_space_saved, total_estimated_time, {}

def display_menu(is_initialized, image_count=0, total_original_size=0, total_compressed_size=0, total_space_saved=0, total_time=0, time_budget=None, sample_percentage=SAMPLE_PERCENTAGE):
    """Display the main menu with basic info"""
    print("=" * 50)
    print("IMAGE COMPRESSION SCHEDULER")
    print("=" * 50)
    
    if is_initialized:
        print(f"Found images: {image_count}")
        print(f"Total original size: {total_original_size:.2f} KB")
        
        if sample_percentage == 100:
            print(f"Compressed size: {total_compressed_size:.2f} KB")
            print(f"Space saved: {total_space_saved:.2f} KB")
            print(f"Compression ratio: {(1 - total_compressed_size/total_original_size) * 100:.2f}%")
            print(f"Total processing time: {total_time:.2f} seconds")
        else:
            print(f"Estimated compressed size: {total_compressed_size:.2f} KB")
            print(f"Estimated space saved: {total_space_saved:.2f} KB")
            print(f"Estimated compression ratio: {(1 - total_compressed_size/total_original_size) * 100:.2f}%")
            print(f"Estimated total processing time: {total_time:.2f} seconds")
            
        print(f"Sample percentage: {sample_percentage}%")
        print(f"Time budget: {time_budget if time_budget else 'None'}")
    else:
        print("System not initialized. Please run estimation first.")
    
    print("=" * 50)
    print("1. Run estimation (required before scheduling)")
    print("2. Change sample percentage (currently: {}%)".format(sample_percentage))
    print("3. Set the time budget (deadline)")
    print("4. FCFS Scheduling")
    print("5. Greedy Scheduling")
    print("6. DP Scheduling")
    print("0. Exit")
    print("=" * 50)

def main():
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"\nError: Input folder '{INPUT_FOLDER}' does not exist!")
        return
    
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    clear_terminal()
    
    # Main menu loop
    is_initialized = False
    image_info = None
    estimated_times = None
    total_original_size = 0
    total_estimated_compressed_size = 0 
    total_estimated_space_saved = 0
    total_estimated_time = 0
    time_budget = None
    sample_percentage = SAMPLE_PERCENTAGE
    compressed_files = {}
    
    while True:
        display_menu(is_initialized, 
                    image_count=len(image_info) if image_info else 0,
                    total_original_size=total_original_size,
                    total_compressed_size=total_estimated_compressed_size,
                    total_space_saved=total_estimated_space_saved,
                    total_time=total_estimated_time,
                    time_budget=time_budget,
                    sample_percentage=sample_percentage)
        
        try:
            choice = int(input("\nEnter your choice (0-6): "))
            
            if choice == 0:
                print("Exiting program. Goodbye!")
                break
                
            elif choice == 1:
                # Run estimation with specified sample percentage
                image_info, estimated_times, total_original_size, total_estimated_compressed_size, total_estimated_space_saved, total_estimated_time, compressed_files = run_estimation(
                    INPUT_FOLDER, OUTPUT_FOLDER, QUALITY, sample_percentage
                )
                if image_info:
                    is_initialized = True
                    if sample_percentage == 100:
                        print("\nFull processing completed successfully!")
                        print("Images are already compressed and won't need recompression")
                    else:
                        print("\nEstimation completed successfully!")
                input("\nPress Enter to return to menu...")
                
            elif choice == 2:
                # Change sample percentage
                new_percentage = get_sample_percentage()
                if new_percentage != sample_percentage:
                    sample_percentage = new_percentage
                    if is_initialized:
                        print("\nYou may want to run estimation again with the new sample percentage.")
                        # If we change from 100% to something else, we should clear the compressed_files
                        if sample_percentage != 100:
                            compressed_files = {}
                
            elif choice == 3:
                # Set time budget
                time_budget = get_time_budget()
                
            elif choice in [4, 5, 6]:
                if not is_initialized:
                    print("\nError: Please run estimation first (option 1)!")
                    input("\nPress Enter to continue...")
                else:
                    # Map menu option to scheduler name
                    scheduler_map = {4: "fcfs", 5: "greedy", 6: "dp"}
                    process_with_scheduler(
                        scheduler_map[choice], 
                        image_info, 
                        INPUT_FOLDER, 
                        OUTPUT_FOLDER, 
                        QUALITY, 
                        estimated_times, 
                        time_budget,
                        compressed_files,
                        sample_percentage
                    )
                    input("\nPress Enter to return to menu...")
                
            else:
                print("Error: Please enter a number between 0-6!")
                input("\nPress Enter to continue...")
                
            clear_terminal()
            
        except ValueError:
            print("Error: Please enter a valid number!")
            input("\nPress Enter to continue...")
            clear_terminal()

if __name__ == "__main__":
    main()