import compressor

def schedule(image_info):
  """First-Come, First-Served scheduling algorithm"""

  # Process files in the order the were found (FCFS)
  ordered_filenames = [info[0] for info in image_info]

  print(f"\nScheduling order ({len(image_info)} image(s)):")
  for i, filename in enumerate(ordered_filenames):
    print(f"{i+1}. {filename}")

  return ordered_filenames