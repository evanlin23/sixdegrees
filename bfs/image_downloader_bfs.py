# bfs/image_downloader_bfs.py
import os
from icrawler.builtin import GoogleImageCrawler
import time

def fetch_images_for_link_bfs(
    subjects_str_or_person_name,
    google_query_str,
    root_dl_folder,  # This is already the specific target folder for this link
    num_images_to_dl,
    logger,
    max_retries,
    retry_delay,
    is_reference_download=False
    ):
    logger.info(
        f"BFS DL: Preparing to download. Target: '{subjects_str_or_person_name}', "
        f"Query: '{google_query_str}', Num: {num_images_to_dl}, Ref: {is_reference_download}, "
        f"MaxRetries: {max_retries}, RetryDelay: {retry_delay}s"
    )

    target_specific_output_folder = root_dl_folder

    if not os.path.exists(target_specific_output_folder):
        try:
            os.makedirs(target_specific_output_folder)
            logger.info(f"BFS DL: Created image download folder: {target_specific_output_folder}")
        except OSError as e:
            logger.error(f"BFS DL: Failed to create directory {target_specific_output_folder}: {e}")
            return None # Indicate failure to prepare download location
    else:
        logger.debug(f"BFS DL: Image download folder exists: {target_specific_output_folder}")

    if num_images_to_dl <= 0:
        logger.info(f"BFS DL: Number of images to download is {num_images_to_dl}. Skipping crawl.")
        return target_specific_output_folder # Return folder path as per existing logic

    attempt_count = 0
    images_found_in_folder = 0
    # Ensure max_retries is at least 1 for the initial attempt
    actual_max_attempts = max(1, max_retries)


    while attempt_count < actual_max_attempts:
        attempt_count += 1
        logger.info(f"BFS DL: Download attempt {attempt_count}/{actual_max_attempts} for query: '{google_query_str}'")

        # Ensure the folder exists for the crawler (might be redundant if created above, but safe)
        if not os.path.exists(target_specific_output_folder):
            try:
                os.makedirs(target_specific_output_folder)
            except OSError as e:
                 logger.error(f"BFS DL: Failed to ensure directory {target_specific_output_folder} for attempt {attempt_count}: {e}")
                 # If critical, could 'continue' to next retry or 'return None'
                 # For now, let icrawler try and potentially fail

        google_crawler = GoogleImageCrawler(
            parser_threads=2,
            downloader_threads=4,
            storage={'root_dir': target_specific_output_folder}
        )

        try:
            logger.debug(
                f"BFS DL: Starting icrawler.crawl (Attempt {attempt_count}) for: '{google_query_str}' "
                f"(requesting up to {num_images_to_dl} total images in folder)"
            )

            google_crawler.crawl(
                keyword=google_query_str,
                max_num=num_images_to_dl,
                min_size=(200, 200), # Consider making this configurable
                file_idx_offset='auto'
            )
            crawl_msg_target = subjects_str_or_person_name
            logger.info(f"BFS DL: Finished icrawler.crawl (Attempt {attempt_count}) for '{crawl_msg_target}'.")

            current_downloaded_paths = []
            if os.path.exists(target_specific_output_folder):
                try:
                    for root, _, files in os.walk(target_specific_output_folder):
                        for file_name in files:
                            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')):
                                 current_downloaded_paths.append(os.path.join(root, file_name))
                    images_found_in_folder = len(current_downloaded_paths)
                except Exception as e_list:
                    logger.error(
                        f"BFS DL: Error listing files in {target_specific_output_folder} "
                        f"after crawl (Attempt {attempt_count}): {e_list}"
                    )

            logger.info(
                f"BFS DL: Found {images_found_in_folder} image(s) in {target_specific_output_folder} "
                f"after crawl attempt {attempt_count} for query '{google_query_str}'."
            )

            if images_found_in_folder >= num_images_to_dl:
                logger.info(
                    f"BFS DL: Sufficient images ({images_found_in_folder}/{num_images_to_dl}) found. "
                    "Download considered successful for this stage."
                )
                break # Exit retry loop

            if attempt_count < actual_max_attempts:
                logger.warning(
                    f"BFS DL: Downloaded {images_found_in_folder}/{num_images_to_dl} images. "
                    f"Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                logger.warning(
                    f"BFS DL: Max attempts ({actual_max_attempts}) reached. "
                    f"Downloaded {images_found_in_folder}/{num_images_to_dl} images for query '{google_query_str}'."
                )

        except Exception as e_crawl:
            logger.error(
                f"BFS DL: ERROR during icrawler.crawl (Attempt {attempt_count}) for '{google_query_str}': {e_crawl}",
                exc_info=True
            )
            if attempt_count < actual_max_attempts:
                logger.info(f"BFS DL: Crawl error on attempt {attempt_count}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"BFS DL: Crawl error on final attempt ({attempt_count}). No more retries for this link."
                )

    # Final status logging after all attempts
    if images_found_in_folder == 0 and num_images_to_dl > 0:
        logger.warning(
            f"BFS DL: Ultimately downloaded 0 images for query: '{google_query_str}' "
            f"for '{subjects_str_or_person_name}' after {attempt_count} attempts, "
            f"despite requesting {num_images_to_dl}."
        )
    elif images_found_in_folder < num_images_to_dl and num_images_to_dl > 0:
         logger.warning(
            f"BFS DL: Ultimately downloaded only {images_found_in_folder}/{num_images_to_dl} requested images "
            f"for query: '{google_query_str}' after {attempt_count} attempts."
        )
    elif images_found_in_folder >= num_images_to_dl:
        logger.info(
            f"BFS DL: Successfully obtained {images_found_in_folder} images for query '{google_query_str}' "
            f"after {attempt_count} attempt(s)."
        )

    return target_specific_output_folder

if __name__ == "__main__":
    # Example usage for direct testing (requires a logger setup)
    import logging
    test_logger = logging.getLogger("TestImageDownloader")
    test_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    test_logger.addHandler(ch)

    # Create a dummy temp folder for testing
    temp_test_folder = "temp_image_test_dl"
    if not os.path.exists(temp_test_folder):
        os.makedirs(temp_test_folder)

    fetch_images_for_link_bfs(
        subjects_str_or_person_name="Test Subject",
        google_query_str="cat", # A query likely to return images
        root_dl_folder=temp_test_folder,
        num_images_to_dl=2,
        logger=test_logger,
        max_retries=2,
        retry_delay=3,
        is_reference_download=False
    )
    print(f"Check the folder: {temp_test_folder}")