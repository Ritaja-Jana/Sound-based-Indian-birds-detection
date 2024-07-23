import os
import pandas as pd
import requests

# Define the output directory
output_dir = 'bird_calls1'

# Define the bird species and the number of audio files to download for each
species_counts = {
    'Oriental Magpie-Robin': 170,
    'Asian Koel': 117,
    'Common Tailorbird': 116,
    'Rufous Treepie': 102,
    'Black-hooded Oriole': 93,
    'White-cheeked Barbet': 88,
    'Ashy Prinia': 83,
    'Puff-throated Babbler': 82,
    'White-throated Kingfisher': 81,
    'Red-vented Bulbul': 81,
    'Jungle Babbler': 79,
    'Common Hawk-Cuckoo': 73,
    'Indian Scimitar Babbler': 72,
    'Red-whiskered Bulbul': 71,
    'Red-wattled Lapwing': 70,
    'Common Iora': 69,
    'Purple Sunbird': 68,
    'Greater Coucal': 65,
    'Blyth\'s Reed Warbler': 64,
    'Orange-headed Thrush': 64,
    'House Crow': 63,
    'Greater Racket-tailed Drongo': 62,
    'Malabar Whistling Thrush': 62
}

# Read the CSV file
csv_file = csv_file_path = os.path.join('data', 'birds_india.csv')
df = pd.read_csv(csv_file)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Download audio files
def download_audio(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {save_path}")
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")

# Process each species
for species, count in species_counts.items():
    # Create directory for species if it doesn't exist
    species_dir = os.path.join(output_dir, species.replace(' ', '_'))
    if not os.path.exists(species_dir):
        os.makedirs(species_dir)
    
    # Filter and download audio files for this species
    filtered_df = df[df['en'] == species]
    urls = filtered_df[['file', 'file-name']].head(count)  # Get URLs and file names for the specified count
    
    for i, row in urls.iterrows():
        url = row['file']
        file_name = row['file-name']
        if not isinstance(file_name, str) or pd.isna(file_name):
            continue  # Skip if file_name is not a string or is NaN
        if not url.startswith('http'):
            url = 'https:' + url  # Ensure the URL has the correct scheme
        file_path = os.path.join(species_dir, file_name)
        download_audio(url, file_path)

print("Download completed.")