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
INPUT_FOLDER = "./before"
OUTPUT_FOLDER = "./after"
QUALITY = 50

def clear_terminal():
    """Clear the terminal screen based on the operating system"""
    if platform.system() == "Windows":
        subprocess.call('cls', shell=True)
    else:
        subprocess.call('clear', shell=True)

def update_image_info_with_times(image_info, measured_times):
    """Updates the image_info list with actual measured processing times"""
    updated_info = []
    for filename, filesize, _ in image_info:
        if filename in measured_times:
            updated_info.append((filename, filesize, measured_times[filename]))
        else:
            updated_info.append((filename, filesize, 0))
    return updated_info

def process_with_scheduler(scheduler_name, image_info, input_folder, output_folder, quality, processing_times=None, time_budget=None, compressed_sizes=None):
    """Process images using the selected scheduling algorithm without recompressing"""
    
    print(f"\nUsing {scheduler_name.upper()} Scheduling Algorithm")
    
    # If we have compressed sizes from initialization, we will use them
    if compressed_sizes is None:
        compressed_sizes = {}
        # Assuming we calculate the compressed sizes during initialization
        for filename, filesize, _ in image_info:
            output_path = os.path.join(output_folder, filename)
            if os.path.isfile(output_path):
                compressed_sizes[filename] = os.path.getsize(output_path) / 1024  # Convert to KB
    
    total_time = 0
    total_space_saved = 0
    ordered_filenames = []
    
    if scheduler_name == "fcfs":
        if time_budget is not None:
            items = [(i, info[0], info[2]) for i, info in enumerate(image_info)]
            selected = []
            
            for i, filename, proc_time in items:
                if total_time + proc_time <= time_budget:
                    selected.append((i, filename))
                    total_time += proc_time
                    for fname, fsize, _ in image_info:
                        if fname == filename:
                            compressed_size = compressed_sizes.get(fname, 0)
                            total_space_saved += fsize - compressed_size
                            break
            
            ordered_filenames = [item[1] for item in selected]
            
            print(f"\nScheduling order ({len(ordered_filenames)} of {len(image_info)} image(s)):")
            print(f"Time Budget: {time_budget:.2f}s, Used: {total_time:.2f}s")
            print(f"Total space saved: {total_space_saved:.2f}KB")
            for i, filename in enumerate(ordered_filenames):
                print(f"{i+1}. {filename}")
        else:
            ordered_filenames = fcfs.schedule(image_info)
            total_space_saved = sum(info[1] - compressed_sizes.get(info[0], 0) for info in image_info)
  
    elif scheduler_name == "greedy":
        if time_budget is not None:
            print(f"Greedy Knapsack (Time Budget: {time_budget:.2f}s)")
            ordered_filenames = greedy.schedule(image_info, time_budget)
        else:
            print("Shortest Processing Time First")
            ordered_filenames = greedy.schedule(image_info)
            total_space_saved = sum(info[1] - compressed_sizes.get(info[0], 0) for info in image_info)
  
    elif scheduler_name == "dp":
        if time_budget is not None:
            print(f"DP Knapsack (Time Budget: {time_budget:.2f}s)")
            ordered_filenames = dp.schedule(image_info, time_budget)
        else:
            print("Highest Ratio First (Smith's Rule)")
            ordered_filenames = dp.schedule(image_info)
            total_space_saved = sum(info[1] - compressed_sizes.get(info[0], 0) for info in image_info)
    else:
        print(f"Error: Unknown scheduling algorithm {{{scheduler_name}}}")
        return 0, {}

    print("\nScheduling Completed.")
    print(f"Scheduling for {scheduler_name} completed with:")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    print(f"Total space saved: {total_space_saved:.2f} KB")

    return total_time, compressed_sizes


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

def run_initialization(input_folder, output_folder, quality):
    """Run the initialization phase with minimal output"""
    print("Initialization in progress...")
    image_info = compressor.get_image_info(input_folder)
    
    if not image_info:
        print("Error: No valid images found in the folder!")
        return None, None, 0, 0, 0, {}
    
    start_time = time.time()
    processed, skipped, _, measured_times = compressor.compress_images(
        input_folder, output_folder, quality
    )
    total_time = time.time() - start_time
    
    total_original_size = sum(info[1] for info in image_info)
    
    total_compressed_size = 0
    compressed_sizes = {}
    for file_name in os.listdir(output_folder):
        output_path = os.path.join(output_folder, file_name)
        if os.path.isfile(output_path):
            total_compressed_size += os.path.getsize(output_path) / 1024
            compressed_sizes[file_name] = os.path.getsize(output_path) / 1024  # Store the compressed size
    
    updated_image_info = update_image_info_with_times(image_info, measured_times)
    
    return updated_image_info, measured_times, total_original_size, total_compressed_size, total_time, compressed_sizes


def display_menu(image_count, total_original_size, total_compressed_size, total_time, time_budget=None):
    """Display the main menu with basic info"""
    print("=" * 50)
    print("IMAGE COMPRESSION SCHEDULER")
    print("=" * 50)
    print(f"Found images: {image_count}")
    print(f"Total original size: {total_original_size:.2f} KB")
    print(f"Total compressed size: {total_compressed_size:.2f} KB")
    print(f"Compression ratio: {(1 - total_compressed_size/total_original_size) * 100:.2f}%")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Time budget: {time_budget if time_budget else 'None'}")
    print("=" * 50)
    print("1. Set the time budget (deadline)")
    print("2. FCFS Scheduling")
    print("3. Greedy Scheduling")
    print("4. DP Scheduling")
    print("0. Exit")
    print("=" * 50)

def main():
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"\nError: Input folder '{INPUT_FOLDER}' does not exist!")
        return
    
    # Run initialization with minimal output
    image_info, measured_times, total_original_size, total_compressed_size, total_time, processed = run_initialization(
        INPUT_FOLDER, OUTPUT_FOLDER, QUALITY
    )
    
    if not image_info:
        return
    
    clear_terminal()
    
    # Main menu loop
    time_budget = None
    while True:
        display_menu(len(image_info), total_original_size, total_compressed_size, total_time, time_budget)
        
        try:
            choice = int(input("\nEnter your choice (0-4): "))
            
            if choice == 0:
                print("Exiting program. Goodbye!")
                break
                
            elif choice == 1:
                time_budget = get_time_budget()
                
            elif choice == 2:
                process_with_scheduler("fcfs", image_info, INPUT_FOLDER, OUTPUT_FOLDER, QUALITY, measured_times, time_budget)
                input("\nPress Enter to return to menu...")
                
            elif choice == 3:
                process_with_scheduler("greedy", image_info, INPUT_FOLDER, OUTPUT_FOLDER, QUALITY, measured_times, time_budget)
                input("\nPress Enter to return to menu...")
                
            elif choice == 4:
                process_with_scheduler("dp", image_info, INPUT_FOLDER, OUTPUT_FOLDER, QUALITY, measured_times, time_budget)
                input("\nPress Enter to return to menu...")
                
            else:
                print("Error: Please enter a number between 0-4!")
                
            clear_terminal()
            
        except ValueError:
            print("Error: Please enter a valid number!")
            input("\nPress Enter to continue...")
            clear_terminal()

if __name__ == "__main__":
    main()