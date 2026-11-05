import xml.etree.ElementTree as ET
import os
import re
from icrawler.builtin import GoogleImageCrawler # You can also try BingImageCrawler

# --- Configuration ---
ROOT_OUTPUT_FOLDER = "connection_chain_images"
IMAGES_PER_LINK = 10 # How many images to download for each link
XML_INPUT_FILE = "input.xml" # Name of the input XML file

def sanitize_filename(name):
    """
    Sanitizes a string to be suitable for a filename or directory name.
    Removes or replaces invalid characters.
    """
    name = "_".join([part.strip() for part in name.split('→')])
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(' ', '_')
    name = re.sub(r'_+', '_', name)
    return name

def fetch_images_for_link(subjects_str, google_query_str, parent_folder, num_images):
    """
    Fetches images for a given link and saves them in a subfolder.
    """
    folder_name = sanitize_filename(subjects_str)
    link_output_folder = os.path.join(parent_folder, folder_name)

    if not os.path.exists(link_output_folder):
        os.makedirs(link_output_folder)
        print(f"Created folder: {link_output_folder}")
    else:
        print(f"Folder exists: {link_output_folder}")

    print(f"  Searching for: '{google_query_str}'")
    print(f"  Saving up to {num_images} images to: {link_output_folder}")

    google_crawler = GoogleImageCrawler(
        parser_threads=2,
        downloader_threads=4,
        storage={'root_dir': link_output_folder}
    )
    try:
        google_crawler.crawl(
            keyword=google_query_str,
            max_num=num_images,
            min_size=(200, 200),
            file_idx_offset=0
        )
        print(f"  Finished downloading for '{subjects_str}'. Check folder for results.")
    except Exception as e:
        print(f"  ERROR crawling for '{google_query_str}': {e}")
        print("  This might be due to Google blocking requests or other network issues.")

def main():
    # Create the root output folder if it doesn't exist
    if not os.path.exists(ROOT_OUTPUT_FOLDER):
        os.makedirs(ROOT_OUTPUT_FOLDER)
        print(f"Created root output folder: {ROOT_OUTPUT_FOLDER}")

    # Read XML from file
    try:
        with open(XML_INPUT_FILE, 'r', encoding='utf-8') as file:
            xml_input_string = file.read()
    except FileNotFoundError:
        print(f"Error: Input file '{XML_INPUT_FILE}' not found.")
        return
    except Exception as e:
        print(f"Error reading input file '{XML_INPUT_FILE}': {e}")
        return

    # Parse the XML
    try:
        root = ET.fromstring(xml_input_string)
    except ET.ParseError as e:
        print(f"Error parsing XML from '{XML_INPUT_FILE}': {e}")
        return

    # Iterate over each <link> element
    for link_element in root.findall('link'):
        link_id = link_element.get('id')
        subjects_element = link_element.find('subjects')
        google_query_element = link_element.find('google')

        if subjects_element is not None and google_query_element is not None:
            subjects_str = subjects_element.text
            google_query_str = google_query_element.text
            print(f"\nProcessing Link ID: {link_id} ({subjects_str})")
            fetch_images_for_link(subjects_str, google_query_str, ROOT_OUTPUT_FOLDER, IMAGES_PER_LINK)
        else:
            print(f"Skipping Link ID: {link_id} - missing 'subjects' or 'google' tag.")

    print("\nAll links processed.")

if __name__ == "__main__":
    main()