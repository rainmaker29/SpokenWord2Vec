import re
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot correlation values over epochs from a file.')
parser.add_argument('input_file', type=str, help='Path to the experiment output file')
args = parser.parse_args()


file_content =''

with open(args.input_file, 'r') as file:
    file_content = file.read()
 

# Regular expressions to find the cosine distance and edit distance values
cosine_pattern = re.compile(r"w\. target model cosine dist: ([\d.]+)")
edit_pattern = re.compile(r"w\. edit distance: ([\d.]+)")

# Extracting the values
cosine_values = [float(match) for match in cosine_pattern.findall(file_content)]
edit_values = [float(match) for match in edit_pattern.findall(file_content)]

epochs = list(range(1, len(cosine_values) + 1))


plt.figure(figsize=(5, 5))
plt.plot(epochs, cosine_values, 'k-', label='cosine', linewidth=2)  
plt.plot(epochs, edit_values, 'm--', label='edit', linewidth=2)  # Magenta dashed line, thicker
plt.xlabel('Epochs')
plt.ylabel('Correlation Values')
plt.legend()
plt.grid(False)  
plt.savefig('correlation_plot.png')  
plt.close()
