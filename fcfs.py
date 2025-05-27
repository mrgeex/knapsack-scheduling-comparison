import compressor

def schedule(image_info):
  """First-Come, First-Served scheduling algorithm"""

  # Process files in the order the were found (FCFS)
  ordered_filenames = [info[0] for info in image_info]

  # Calculate total space saved
  total_space_saved = sum(info[3] for info in image_info)

  print(f"\nScheduling order ({len(image_info)} image(s)):")
  print(f"Total space saved: {total_space_saved:.2f}KB")
  for i, (filename, _, _, space_saved, _) in enumerate(image_info):
    print(f"{i+1}. {filename} (Space saved: {space_saved:.2f}KB)")

  return ordered_filenames