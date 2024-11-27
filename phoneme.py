# import os
# import re
# from collections import defaultdict

# def extract_phonemes(annotation_folder):
#     phoneme_count = defaultdict(int)

#     # Regular expression to match the phoneme part of each line
#     phoneme_pattern = re.compile(r'\d+\s+\d+\s+(\S+)')

#     # Iterate over all files in the annotation folder
#     for file_name in os.listdir(annotation_folder):
#         if file_name.endswith('.lab'):
#             file_path = os.path.join(annotation_folder, file_name)

#             # Open and read the annotation file
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 for line in file:
#                     # Extract phoneme using regex
#                     match = phoneme_pattern.match(line.strip())
#                     if match:
#                         phoneme = match.group(1)
#                         phoneme_count[phoneme] += 1

#     # Sort the phonemes by name
#     sorted_phoneme_count = dict(sorted(phoneme_count.items()))
#     return sorted_phoneme_count

# # Usage example
# annotation_folder = 'data/annotation'
# phoneme_counts = extract_phonemes(annotation_folder)

# # Output the phoneme counts
# print("Phoneme counts:")
# for phoneme, count in phoneme_counts.items():
#     print(f"{phoneme}: {count}")

# # Total number of unique phonemes
# print(f"\nTotal unique phonemes: {len(phoneme_counts)}")
# print(phoneme_counts)

import os
import re
from collections import defaultdict

def extract_phonemes_and_next(annotation_folder):
    phoneme_count = defaultdict(int)
    next_phoneme_count = defaultdict(lambda: defaultdict(int))  # To count next phoneme occurrences

    # Regular expression to match the phoneme part of each line
    phoneme_pattern = re.compile(r'\d+\s+\d+\s+(\S+)')

    # Iterate over all files in the annotation folder
    for file_name in os.listdir(annotation_folder):
        if file_name.endswith('.lab'):
            file_path = os.path.join(annotation_folder, file_name)

            # Open and read the annotation file
            with open(file_path, 'r', encoding='utf-8') as file:
                prev_phoneme = None  # Variable to track the previous phoneme
                for line in file:
                    # Extract phoneme using regex
                    match = phoneme_pattern.match(line.strip())
                    if match:
                        phoneme = match.group(1)
                        phoneme_count[phoneme] += 1
                        
                        # If there's a previous phoneme, count the transition to the current one
                        if prev_phoneme:
                            next_phoneme_count[prev_phoneme][phoneme] += 1
                        
                        # Update the previous phoneme
                        prev_phoneme = phoneme

    # Sort phonemes by name
    sorted_phoneme_count = dict(sorted(phoneme_count.items()))

    # Sort transitions by the first phoneme and then by the second
    sorted_next_phoneme_count = {k: dict(sorted(v.items())) for k, v in sorted(next_phoneme_count.items())}

    return sorted_phoneme_count, sorted_next_phoneme_count

# Usage example
annotation_folder = 'data/annotation'
phoneme_counts, next_phoneme_counts = extract_phonemes_and_next(annotation_folder)

# Output the phoneme counts
print("Phoneme counts:")
for phoneme, count in phoneme_counts.items():
    print(f"{phoneme}: {count}")

# Output the next phoneme counts
print("\nNext phoneme counts:")
for phoneme, next_phonemes in next_phoneme_counts.items():
    print(f"{phoneme} transitions:")
    for next_phoneme, count in next_phonemes.items():
        print(f"  {next_phoneme}: {count}")

# Total number of unique phonemes
print(f"\nTotal unique phonemes: {len(phoneme_counts)}")
