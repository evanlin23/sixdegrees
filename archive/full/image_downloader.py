# image_downloader.py
import os
import re
from icrawler.builtin import GoogleImageCrawler
import logging

def sanitize_filename(name):
    if not name: return "unknown_subject"
    # If '→' is present, it's a link; otherwise, assume it's a single person for reference.
    if '→' in name:
        name = "_".join([part.strip() for part in name.split('→')])
    else: # Single person name for reference folder
        name = name.strip()

    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(' ', '_')
    name = re.sub(r'_+', '_', name)
    name = re.sub(r'^_|_$', '', name)
    return name if name else "unknown_subject"

def fetch_images_for_link(subjects_str_or_person_name, google_query_str, root_dl_folder, num_images_to_dl, logger, is_reference_download=False):
    logger.info(f"Preparing to download images. Target: '{subjects_str_or_person_name}', Query: '{google_query_str}', Num: {num_images_to_dl}, Reference: {is_reference_download}")

    folder_name_sanitized = sanitize_filename(subjects_str_or_person_name)

    if is_reference_download:
        # For reference downloads, root_dl_folder IS the person-specific folder.
        link_specific_output_folder = root_dl_folder
        # Ensure the folder exists (image_verifier.py should also do this, but good to be robust)
        if not os.path.exists(link_specific_output_folder):
             try:
                os.makedirs(link_specific_output_folder)
                logger.info(f"Created reference image folder (by downloader): {link_specific_output_folder}")
             except OSError as e:
                logger.error(f"Failed to create directory {link_specific_output_folder} (by downloader): {e}")
                return None
    else:
        # For non-reference (event specific), create a subfolder within root_dl_folder
        link_specific_output_folder = os.path.join(root_dl_folder, folder_name_sanitized)
        if not os.path.exists(link_specific_output_folder):
            try:
                os.makedirs(link_specific_output_folder)
                logger.info(f"Created image download folder: {link_specific_output_folder}")
            except OSError as e:
                logger.error(f"Failed to create directory {link_specific_output_folder}: {e}")
                return None
        else:
            logger.info(f"Image download folder exists: {link_specific_output_folder}")

    logger.info(f"  Saving up to {num_images_to_dl} images to: {link_specific_output_folder} for query: '{google_query_str}'")

    if num_images_to_dl <= 0:
        logger.info(f"  Number of images to download is {num_images_to_dl}. Skipping crawl.")
        return link_specific_output_folder 

    google_crawler = GoogleImageCrawler(
        parser_threads=2,
        downloader_threads=4,
        storage={'root_dir': link_specific_output_folder}
    )
    try:
        logger.debug(f"Starting icrawler.crawl for: {google_query_str} (max_num={num_images_to_dl})")
        google_crawler.crawl(
            keyword=google_query_str,
            max_num=num_images_to_dl,
            min_size=(200, 200),
            file_idx_offset='auto'
        )
        crawl_msg_target = subjects_str_or_person_name if not is_reference_download else f"reference for {subjects_str_or_person_name}"
        logger.info(f"  Finished icrawler.crawl for '{crawl_msg_target}'.")

        downloaded_files = []
        if os.path.exists(link_specific_output_folder):
            try:
                downloaded_files = [f for f in os.listdir(link_specific_output_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            except Exception as e:
                logger.error(f"Error listing files in {link_specific_output_folder} after crawl: {e}")
        
        logger.info(f"  Found {len(downloaded_files)} image files in {link_specific_output_folder} after crawl attempt.")
        if not downloaded_files and num_images_to_dl > 0 : 
            logger.warning(f"  No new images appear to have been downloaded by icrawler for query: '{google_query_str}' for '{crawl_msg_target}' despite requesting {num_images_to_dl}.")

        return link_specific_output_folder
    except Exception as e:
        logger.error(f"  ERROR during icrawler.crawl for '{google_query_str}': {e}", exc_info=True)
        return None

if __name__ == "__main__":
    print("This script is intended to be called by main_orchestrator.py")