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

    # Removed the skip logic for is_reference_download if folder already has files.
    # image_verifier.py now controls whether to call this based on existing file counts.

    if num_images_to_dl <= 0:
        logger.info(f"  Number of images to download is {num_images_to_dl}. Skipping crawl.")
        return link_specific_output_folder # Return folder path even if no download, as it might exist

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
            file_idx_offset='auto' # CRUCIAL: To avoid overwriting when adding more images
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
        if not downloaded_files and num_images_to_dl > 0 : # Only warn if we expected to download
            logger.warning(f"  No new images appear to have been downloaded by icrawler for query: '{google_query_str}' for '{crawl_msg_target}' despite requesting {num_images_to_dl}.")
            # If it was a reference download and no files were obtained, and we tried to get some,
            # this might be an issue, but image_verifier will re-check the folder.
            # Returning the folder path is still generally correct.

        return link_specific_output_folder # Return the folder where images were (or should have been) saved
    except Exception as e:
        logger.error(f"  ERROR during icrawler.crawl for '{google_query_str}': {e}", exc_info=True)
        return None # Indicate a more significant failure during the crawl itself

if __name__ == "__main__":
    print("This script is intended to be called by main_orchestrator.py")

    # # Example test for image_downloader.py
    # logging.basicConfig(level=logging.DEBUG)
    # test_logger_downloader = logging.getLogger("downloader_test")
    # test_person = "Example Person"
    # test_ref_folder_base = "test_reference_images"
    # person_sanitized = sanitize_filename(test_person)
    # specific_person_ref_folder = os.path.join(test_ref_folder_base, person_sanitized)

    # if not os.path.exists(test_ref_folder_base): os.makedirs(test_ref_folder_base)
    # if not os.path.exists(specific_person_ref_folder): os.makedirs(specific_person_ref_folder)

    # test_logger_downloader.info(f"--- Test 1: Download 2 images for {test_person} ---")
    # fetch_images_for_link(
    #     subjects_str_or_person_name=test_person,
    #     google_query_str=f'"{test_person}" portrait photo',
    #     root_dl_folder=specific_person_ref_folder, # This is the direct folder for refs
    #     num_images_to_dl=2,
    #     logger=test_logger_downloader,
    #     is_reference_download=True
    # )

    # test_logger_downloader.info(f"--- Test 2: Download 2 MORE images for {test_person} (should use file_idx_offset='auto') ---")
    # fetch_images_for_link(
    #     subjects_str_or_person_name=test_person,
    #     google_query_str=f'"{test_person}" clear face',
    #     root_dl_folder=specific_person_ref_folder, # This is the direct folder for refs
    #     num_images_to_dl=2,
    #     logger=test_logger_downloader,
    #     is_reference_download=True
    # )
    # test_logger_downloader.info(f"Check folder: {specific_person_ref_folder}. Should have ~4 images if downloads were successful.")