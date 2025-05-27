import os
from PIL import Image
import shutil
import time
import stat

def compress_single_image(input_path:str, output_path:str, quality:int=50):
    """
    Compress a single image

    Args:
    input_path(str): Path to input image
    output_path(str): Path to save compressed image
    quality(int): Compression quality (1-100)

    Returns:
        tuple: (success, processing_time) => success is boolean and time in seconds
    """
    start_time = time.time()

    try:
        with Image.open(input_path) as img:
            file_ext = os.path.splitext(output_path)[1].lower()
            image_exts = ('.jpg', '.jpeg', '.png', '.webp')
            if file_ext in image_exts:
                img_format = img.format

            if img_format:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path, format=img_format, optimize=True, quality=quality)
                end_time = time.time()
                return True, end_time - start_time
            else:
                return False, 0

    except Exception as e:
        end_time = time.time()
        return False, end_time - start_time


def compress_images(input_folder:str, output_folder:str, quality=50, files_to_process=None, processing_times=None):
    """
    Compresses images from input_folder and saves them to output_folder.

    Args:
        input_folder(str): Path to folder containing images to compress
        output_folder(str): Path to folder where compressed images will be saved
        quality(int): Compression quality (1-100, lower = more compression)
        files_to_process(list): Optional list of filenames to process in specific order (If None, all files are processed in directory order)
        processing_times(dict): Optional dictionary of processing times for each file

    Returns:
        tuple: (processed_count, skipped_count, total_time, measured_times)
        *measured_times is a dictionary of actual processing times
    """
    overall_start_time = time.time()

    # Check if folders exist:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Clear output folder if it exists
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                os.chmod(file_path, stat.S_IWRITE)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error: {e}")

    # Process each file
    processed_count = 0
    skipped_count = 0
    measured_times = {}

    # Get the list of files to process
    if files_to_process is None:
        files_to_process = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    for filename in files_to_process:
        input_path = os.path.join(input_folder, filename)

        # Skip if not a file or not found
        if not os.path.isfile(input_path):
            skipped_count += 1
            continue

        output_path = os.path.join(output_folder, filename)
        
        # During initialization, show minimal output
        if processing_times is None:
            print(f"Processing: {filename}")
        
        # Compress the image
        success, process_time = compress_single_image(input_path, output_path, quality)

        if success:
            processed_count += 1
            measured_times[filename] = process_time
            
            # During initialization, show minimal output
            if processing_times is None:
                print(f">>> Compressed: {filename}")
            elif processing_times:
                # During scheduling runs, show more detailed output
                time_info = f" (expected time: {processing_times[filename]:.2f}s)"
                print(f"Processing: {filename}{time_info}")
                print(f">>> Compressed: {filename} (took {process_time:.2f}s)")
        else:
            skipped_count += 1

    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time

    return processed_count, skipped_count, total_time, measured_times


def get_image_info(input_folder:str):
    """
    Get information about all images in the input folder Returns a list of tuples (filename, file_size, estimated_processing_time)
    """
    image_info = []
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)

        # Skip if not a file
        if not os.path.isfile(input_path):
            continue

        try:
            # Get file size in KB
            file_size = os.path.getsize(input_path) / 1024
            estimated_time = 0  # Just a placeholder (this will be replaced with actual times once measured)

            image_info.append((file_name, file_size, estimated_time))

        except Exception:
            # Skip non-image files
            pass

    return image_info


if __name__ == "__main__":
    # If run directly, these values will be used
    # Value can be customized
    INPUT_FOLDER = "./before/"
    OUTPUT_FOLDER = "./after/"
    QUALITY = 50  # 1-100, lower = more compression

    print("\n" + "=" * 50)
    print("Starting image compression...")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Compression quality: {QUALITY}")
    print("=" * 50)

    print(f"Found: {len(get_image_info(INPUT_FOLDER))} Images")

    processed, skipped, total_time, _ = compress_images(INPUT_FOLDER, OUTPUT_FOLDER, QUALITY)

    print("=" * 50)
    print(f"Compression Completed!")
    print(f"Successful compression >>> {processed}")
    print(f"Skipped files >>> {skipped}")
    print(f"Overall Time >>> {total_time:.2f} seconds")