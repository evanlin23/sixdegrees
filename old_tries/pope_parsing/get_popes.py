import json
import os
import requests
import re
import urllib.parse # For URL encoding/decoding

# --- Configuration ---
JSON_FILE_PATH = "data.json"  # Path to your JSON file
BASE_DOWNLOAD_DIR = "downloaded_meeting_images"
# --- End Configuration ---

def sanitize_filename(name_component):
    """
    Sanitizes a string component to be a valid part of a filename.
    It should not be used on full filenames with extensions.
    """
    if not isinstance(name_component, str):
        name_component = str(name_component) # Ensure it's a string
    # Remove characters not suitable for filenames (excluding dot if it's part of an extension)
    # This regex keeps alphanumeric, underscore, whitespace, hyphen.
    name_component = re.sub(r'[^\w\s-]', '', name_component)
    # Replace spaces and hyphens with a single underscore, trim leading/trailing underscores
    name_component = re.sub(r'[-\s]+', '_', name_component).strip('_')
    return name_component if name_component else "unknown" # Ensure there's always some name

def get_direct_wikimedia_url(page_url):
    """
    Attempts to get the direct image URL from a Wikimedia Commons File page URL.
    """
    if not page_url or not isinstance(page_url, str) or not page_url.startswith("https://commons.wikimedia.org/wiki/File:"):
        return page_url

    try:
        title = page_url.split("https://commons.wikimedia.org/wiki/", 1)[1]
        encoded_title = urllib.parse.quote(title)
        api_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={encoded_title}&prop=imageinfo&iiprop=url&format=json"
        headers = {'User-Agent': 'MyImageDownloaderScript/1.1 (compatible; Python Requests)'}
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id in pages:
            page_info = pages[page_id]
            if "imageinfo" in page_info and page_info["imageinfo"]:
                direct_url = page_info["imageinfo"][0].get("url")
                if direct_url:
                    return direct_url
        print(f"  Warning: Could not find direct image URL for {page_url} via API.")
        return page_url
    except Exception as e:
        print(f"  Warning: Error processing Wikimedia URL {page_url}: {e}. Using original URL.")
        return page_url

def get_filename_from_url(url, entry_id, prefix="image"):
    """
    Extracts a sanitized filename with a proper extension from a URL.
    """
    if not url or not isinstance(url, str):
        sanitized_prefix = sanitize_filename(prefix)
        return f"{sanitized_prefix}_{entry_id}_fallback.jpg"

    # Common image extensions to check against
    allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff'}

    try:
        path_part = urllib.parse.unquote(url.split('?')[0]) # Get path and unquote (e.g. %20 -> space)
        original_filename_from_url = os.path.basename(path_part)

        final_base_name = ""
        final_extension = ""

        if original_filename_from_url and '.' in original_filename_from_url:
            name_part, dot_char, ext_part = original_filename_from_url.rpartition('.')
            # Check if the part after the last dot is a known image extension
            # and if there's a name part before the dot.
            if name_part and ext_part.lower() in allowed_extensions:
                final_base_name = sanitize_filename(name_part)
                final_extension = ext_part.lower()
                if final_extension == "jpeg": # Normalize
                    final_extension = "jpg"
            # If not a recognized structure (e.g. "file.dat", ".config", "filename_no_ext"),
            # final_base_name and final_extension will remain empty.
            # We'll then try to construct from prefix and content-type.

        if final_base_name and final_extension:
            return f"{final_base_name}.{final_extension}"
        else:
            # Fallback: if URL didn't yield a clear name.ext, or extension wasn't recognized.
            # Construct base name from prefix and entry_id.
            base_name_from_prefix = f"{sanitize_filename(prefix)}_{entry_id}"
            derived_extension = "jpg" # Default

            try:
                headers = {'User-Agent': 'MyImageDownloaderScript/1.1'}
                # Use HEAD for efficiency, but be ready for it to fail or not give Content-Type
                head_response = requests.head(url, timeout=7, headers=headers, allow_redirects=True)
                # No raise_for_status here, as a 404 might still have a URL we tried.
                # The download function will handle actual download errors.
                content_type = head_response.headers.get('content-type')

                if content_type and '/' in content_type:
                    content_ext_candidate = content_type.split('/')[-1].split(';')[0].strip().lower()
                    if content_ext_candidate == "jpeg": content_ext_candidate = "jpg" # Normalize
                    if content_ext_candidate in allowed_extensions:
                        derived_extension = content_ext_candidate
            except requests.exceptions.Timeout:
                print(f"    Timeout while fetching headers for {url}. Defaulting extension to .jpg.")
            except requests.exceptions.RequestException:
                # Silently proceed with default extension if header fetch fails
                pass
            
            return f"{base_name_from_prefix}.{derived_extension}"

    except Exception as e:
        print(f"    Major error in get_filename_from_url for URL '{url}': {e}. Using generic fallback.")
        sanitized_prefix = sanitize_filename(prefix)
        return f"{sanitized_prefix}_{entry_id}_error.jpg"


def download_image(url, save_path):
    """Downloads an image from a URL and saves it to save_path."""
    if not url:
        print(f"  Skipping download: URL is empty/null for destination {os.path.basename(save_path)}")
        return False

    original_url = url
    if isinstance(url, str) and "commons.wikimedia.org/wiki/File:" in url:
        print(f"  Resolving Wikimedia page URL: {url}")
        url = get_direct_wikimedia_url(url)
        if url != original_url:
            print(f"  Resolved to direct URL: {url}")
        else:
            print(f"  Could not resolve or resolution returned same URL. Will try original: {url}")
            if "commons.wikimedia.org/wiki/File:" in url: # Still a page URL
                 print(f"  Skipping download of unresolved Wikimedia page (likely HTML): {url}")
                 return False
    try:
        headers = {'User-Agent': 'MyImageDownloaderScript/1.1'}
        response = requests.get(url, stream=True, timeout=20, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        if not content_type.startswith('image/'):
            print(f"  Warning: Content-Type is '{content_type}', not an image. Skipping download for {url} -> {save_path}")
            if 'text/html' in content_type and "commons.wikimedia.org/wiki/" in url:
                 print(f"  This appears to be an HTML page from Wikimedia, not a direct image link.")
            return False

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Successfully downloaded: {url} -> {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading {url}: {e}")
    except IOError as e:
        print(f"  Error saving file {save_path}: {e}")
    return False


# --- Main Script ---
if __name__ == "__main__":
    os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)

    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {JSON_FILE_PATH}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {JSON_FILE_PATH}. Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading {JSON_FILE_PATH}: {e}")
        exit(1)

    for entry_id, entry_details in data.items():
        print(f"\nProcessing entry: {entry_id}")
        person1_name = entry_details.get("person1", "UnknownPerson1")
        person2_name = entry_details.get("person2", "UnknownPerson2")
        images_info = entry_details.get("images", {})

        if not images_info:
            print(f"  No image information found for entry {entry_id}.")
            continue

        if "joint" in images_info:
            joint_image_url = images_info["joint"]
            if joint_image_url:
                # Construct a descriptive prefix for joint images
                prefix_for_filename = f"joint_{sanitize_filename(person1_name)}_{sanitize_filename(person2_name)}"
                filename = get_filename_from_url(joint_image_url, entry_id, prefix_for_filename)
                save_path = os.path.join(BASE_DOWNLOAD_DIR, filename)
                print(f"  Found joint image for '{person1_name}' and '{person2_name}'. Attempting download...")
                download_image(joint_image_url, save_path)
            else:
                print(f"  Joint image URL is null or empty for entry {entry_id}.")

        elif "person1" in images_info or "person2" in images_info:
            person1_image_url = images_info.get("person1")
            person2_image_url = images_info.get("person2")

            if person1_image_url or person2_image_url:
                s_person1_name = sanitize_filename(person1_name)
                s_person2_name = sanitize_filename(person2_name)
                subfolder_name = f"{entry_id}_{s_person1_name}_and_{s_person2_name}"
                pair_folder_path = os.path.join(BASE_DOWNLOAD_DIR, subfolder_name)
                os.makedirs(pair_folder_path, exist_ok=True)
                print(f"  Processing individual image(s) for '{person1_name}' and '{person2_name}' in folder: {subfolder_name}")

                if person1_image_url:
                    prefix_p1 = f"{s_person1_name}" # More direct prefix for individual file
                    filename1 = get_filename_from_url(person1_image_url, entry_id, prefix_p1)
                    save_path1 = os.path.join(pair_folder_path, filename1)
                    download_image(person1_image_url, save_path1)
                else:
                    print(f"  Person1 ('{person1_name}') image URL is missing or null for entry {entry_id}.")

                if person2_image_url:
                    prefix_p2 = f"{s_person2_name}" # More direct prefix for individual file
                    filename2 = get_filename_from_url(person2_image_url, entry_id, prefix_p2)
                    save_path2 = os.path.join(pair_folder_path, filename2)
                    download_image(person2_image_url, save_path2)
                else:
                    print(f"  Person2 ('{person2_name}') image URL is missing or null for entry {entry_id}.")
            else:
                print(f"  Both person1 and person2 image URLs are missing or null for entry {entry_id}, though keys might exist.")
        else:
            print(f"  No 'joint' image and no 'person1' or 'person2' image keys found for entry {entry_id}.")

    print("\nImage processing complete.")