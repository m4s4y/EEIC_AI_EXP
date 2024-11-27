import os

def process_lab_files(input_dir, output_dir, frame_count):
    """
    Process .lab files to divide their time into frame_count frames and output the phonemes.
    In case of an error, the function will print the file name for debugging.

    Args:
        input_dir (str): Directory containing input .lab files.
        output_dir (str): Directory to save processed .lab files.
        frame_count (int): Number of frames to divide the time into.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_index = 1  # Start from file 1 and increment for each file
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.lab'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            try:
                # Read .lab file
                with open(input_path, 'r') as f:
                    lines = f.readlines()

                phonemes = []
                end_time = 0

                # Parse lines
                for line in lines:
                    line = line.strip()

                    # Skip empty lines or lines that do not contain exactly 3 parts
                    if not line or len(line.split()) != 3:
                        continue

                    start, end, phoneme = line.split()
                    start, end = int(start), int(end)
                    phonemes.append((start, end, phoneme))
                    end_time = max(end_time, end)

                # Divide time into frames
                frame_duration = end_time / frame_count
                frame_phonemes = ['sil'] * frame_count

                # Process frames (backward)
                for i in range(frame_count - 1, -1, -1):
                    start_time = int(i * frame_duration)
                    end_time = int((i + 1) * frame_duration)

                    # Find all phonemes that overlap with the current frame
                    overlapping_phonemes = [
                        phoneme for start, end, phoneme in phonemes
                        if start < end_time and end > start_time
                    ]

                    # Determine the phoneme for the current frame (backward logic)
                    if overlapping_phonemes:
                        if i + 1 < frame_count and frame_phonemes[i + 1] != overlapping_phonemes[-1]:
                            # If next frame differs, take the last phoneme
                            frame_phonemes[i] = overlapping_phonemes[-1]
                        else:
                            # Otherwise, take the second-to-last phoneme
                            frame_phonemes[i] = overlapping_phonemes[-2] if len(overlapping_phonemes) > 1 else overlapping_phonemes[0]

                # Write output .lab file
                with open(output_path, 'w', newline='') as f:
                    for i, phoneme in enumerate(frame_phonemes):
                        start_time = int(i * frame_duration)
                        end_time = int((i + 1) * frame_duration)
                        f.write(f"{start_time} {end_time} {phoneme}\n")

            except Exception as e:
                print(f"Error processing file {file_index}: {file_name}")
                print(f"Error: {e}")
        
        file_index += 1

# Example usage
input_directory = "./data/annotation"
output_directory = "./data/processed"
frames = 200  # Example frame count

process_lab_files(input_directory, output_directory, frames)