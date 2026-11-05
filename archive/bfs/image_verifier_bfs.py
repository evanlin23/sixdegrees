# bfs/image_verifier_bfs.py
import os
import shutil
import time # For potential delays or unique naming
import logging # For standalone testing

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    print("WARNING: deepface library not found. Image verification will be disabled.")
    print("Please install it: pip install deepface")
    DEEPFACE_AVAILABLE = False

# Assuming image_downloader_bfs is in the same directory or package
# Adjust import based on how you run your scripts (as a package or directly)
try:
    from . import image_downloader_bfs # Use this if running as part of a package
except ImportError:
    import image_downloader_bfs # Fallback for direct script execution

# --- Constants ---
# These constants serve as defaults primarily for standalone testing or direct use of this module.
# The main_bfs.py script will typically override these by passing specific arguments.
BFS_REFERENCE_IMAGES_BASE_DIR = os.path.join(os.path.dirname(__file__), "output_bfs", "reference_faces_bfs")
DEFAULT_NUM_TARGET_REF_IMAGES = 5 # Target number of validated reference images per person
DEFAULT_NUM_DOWNLOAD_FOR_REF_POOL = 10 # Download this many to have a pool for validation if refs are needed

# Default models to use for verification, in order of preference or desired sequence
DEFAULT_VERIF_MODELS = ['VGG-Face', 'Facenet', 'ArcFace', 'SFace']

# Custom thresholds for verification (distance). If distance <= threshold, it's a match.
CUSTOM_VERIFICATION_THRESHOLDS = {
    "VGG-Face": 0.68,
    "Facenet": 0.40,
    "ArcFace": 0.68,
    "SFace": 0.55,
}
# Default detector backend for DeepFace
DEFAULT_DETECTOR_BACKEND = 'retinaface'

def _ensure_dir_bfs(directory_path, logger_obj):
    """Ensures a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logger_obj.debug(f"Created directory: {directory_path}")
        except OSError as e:
            logger_obj.error(f"Failed to create directory {directory_path}: {e}")
            return False
    return True

def _validate_and_filter_reference_images(person_name, potential_ref_image_paths, detector_backend_to_use, logger_obj):
    """
    Validates potential reference images by checking for a detectable face.
    Returns a list of paths to valid reference images.
    """
    if not DEEPFACE_AVAILABLE:
        logger_obj.warning("DeepFace not available, skipping reference image validation. Using all provided.")
        return potential_ref_image_paths

    valid_ref_images = []
    logger_obj.debug(f"Ref Img Validation: Validating {len(potential_ref_image_paths)} potential reference images for '{person_name}' using '{detector_backend_to_use}'.")
    for img_path in potential_ref_image_paths:
        try:
            _ = DeepFace.extract_faces(img_path=img_path, enforce_detection=True, detector_backend=detector_backend_to_use, align=True)
            valid_ref_images.append(img_path)
            logger_obj.debug(f"  Ref Img Validation: '{os.path.basename(img_path)}' is VALID for '{person_name}' (face detected).")
        except ValueError as e:
            logger_obj.warning(f"  Ref Img Validation: '{os.path.basename(img_path)}' is INVALID for '{person_name}' (no face detected or other issue with {detector_backend_to_use}): {e}")
        except Exception as e_other:
            logger_obj.error(f"  Ref Img Validation: Error processing '{os.path.basename(img_path)}' for '{person_name}' with {detector_backend_to_use}: {e_other}")

    logger_obj.info(f"Ref Img Validation: For '{person_name}', {len(valid_ref_images)} valid images remain out of {len(potential_ref_image_paths)} initially processed for validation.")
    return valid_ref_images

def find_reference_images_bfs(
    person_name,
    num_target_images_to_keep, # Number of reference images to ideally find and keep
    num_images_to_download_if_needed, # How many to download into the pool if existing are not enough
    detector_backend_to_use, # Detector backend for validation
    logger_obj,
    download_max_retries,
    download_retry_delay
    ):
    """
    Finds or downloads and validates reference images for a given person.
    Aims to return `num_target_images_to_keep` validated image paths.
    """
    logger_obj.info(f"BFS Finding ref images for: '{person_name}' (target: {num_target_images_to_keep}, download pool: {num_images_to_download_if_needed})")
    person_ref_folder_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in person_name).replace(' ', '_')
    person_ref_folder_path = os.path.join(BFS_REFERENCE_IMAGES_BASE_DIR, person_ref_folder_name)

    if not _ensure_dir_bfs(person_ref_folder_path, logger_obj):
        logger_obj.error(f"Could not create or access reference folder for {person_name}. Cannot get reference images.")
        return []

    raw_existing_refs = []
    if os.path.exists(person_ref_folder_path):
        raw_existing_refs = [
            os.path.join(person_ref_folder_path, f)
            for f in os.listdir(person_ref_folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
    logger_obj.info(f"Initially found {len(raw_existing_refs)} potential reference images in folder for '{person_name}'.")

    validated_existing_refs = _validate_and_filter_reference_images(person_name, raw_existing_refs, detector_backend_to_use, logger_obj)
    logger_obj.info(f"{len(validated_existing_refs)} existing reference images for '{person_name}' are valid after filtering.")

    if len(validated_existing_refs) < num_target_images_to_keep:
        logger_obj.info(
            f"Need more reference images for '{person_name}'. Have {len(validated_existing_refs)}, "
            f"target {num_target_images_to_keep}. Attempting to download {num_images_to_download_if_needed} more."
        )
        download_query = f'"{person_name}" face portrait photo'
        image_downloader_bfs.fetch_images_for_link_bfs(
            subjects_str_or_person_name=f"{person_name}_reference",
            google_query_str=download_query,
            root_dl_folder=person_ref_folder_path,
            num_images_to_dl=num_images_to_download_if_needed,
            logger=logger_obj,
            max_retries=download_max_retries,
            retry_delay=download_retry_delay,
            is_reference_download=True
        )
        all_potential_refs_after_dl = [
            os.path.join(person_ref_folder_path, f)
            for f in os.listdir(person_ref_folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        logger_obj.info(f"Total {len(all_potential_refs_after_dl)} potential reference images for '{person_name}' after download attempt.")
        validated_refs = _validate_and_filter_reference_images(person_name, all_potential_refs_after_dl, detector_backend_to_use, logger_obj)
    else:
        validated_refs = validated_existing_refs

    if not validated_refs:
        logger_obj.warning(f"No validated reference images found for '{person_name}' after all attempts.")
        return []

    logger_obj.info(f"Using {min(len(validated_refs), num_target_images_to_keep)} reference images for '{person_name}'.")
    return validated_refs[:num_target_images_to_keep]

def verify_image_with_deepface_models_bfs(
    candidate_image_path,
    person_name,
    person_ref_image_paths,
    models_to_use, # List of model names
    detector_backend_to_use, # Specific detector backend
    logger_obj
    ):
    """
    Verifies if a person is present in a candidate image using specified DeepFace models
    and detector backend against a list of reference images for that person.
    """
    if not DEEPFACE_AVAILABLE:
        logger_obj.warning(f"DeepFace not available. Cannot verify {person_name} in {os.path.basename(candidate_image_path)}.")
        return False, "N/A", float('inf'), float('inf')

    if not person_ref_image_paths:
        logger_obj.warning(f"No reference images provided for {person_name}. Cannot verify.")
        return False, "N/A", float('inf'), float('inf')

    if not models_to_use:
        logger_obj.warning(f"No verification models specified for {person_name}. Cannot verify.")
        return False, "N/A", float('inf'), float('inf')

    logger_obj.debug(f"    BFS DeepFace: Verifying for '{person_name}' in '{os.path.basename(candidate_image_path)}' using {len(person_ref_image_paths)} refs, models: {models_to_use}, detector: {detector_backend_to_use}.")

    for model_name in models_to_use:
        custom_threshold_for_model = CUSTOM_VERIFICATION_THRESHOLDS.get(model_name)
        if custom_threshold_for_model is None:
            logger_obj.warning(f"Custom threshold not defined for model '{model_name}'. Skipping this model for {person_name}.")
            continue

        for ref_idx, ref_img_path in enumerate(person_ref_image_paths):
            try:
                result = DeepFace.verify(
                    img1_path=candidate_image_path,
                    img2_path=ref_img_path,
                    model_name=model_name,
                    enforce_detection=True,
                    detector_backend=detector_backend_to_use,
                    align=True,
                )

                distance = result.get("distance", float('inf'))
                model_internal_threshold = result.get("threshold", float('inf'))

                logger_obj.debug(
                    f"      P: {person_name}, M: {model_name}, Det: {detector_backend_to_use}, Ref: {ref_idx+1} ({os.path.basename(ref_img_path)}), "
                    f"Dist: {distance:.4f}, ModelThr: {model_internal_threshold:.4f}, CustThr: {custom_threshold_for_model:.4f}"
                )

                if distance <= custom_threshold_for_model:
                    logger_obj.info(
                        f"      ---> VERIFIED '{person_name}' with {model_name} (Detector: {detector_backend_to_use}, Ref {ref_idx+1}, Dist: {distance:.4f} <= CustThr: {custom_threshold_for_model:.4f})"
                    )
                    return True, model_name, distance, custom_threshold_for_model

            except ValueError as ve:
                logger_obj.debug(f"      P: {person_name}, M: {model_name}, Det: {detector_backend_to_use}, Ref: {ref_idx+1}: ValueError (likely no face): {ve}")
            except Exception as e:
                logger_obj.error(
                    f"      P: {person_name}, M: {model_name}, Det: {detector_backend_to_use}, Ref: {ref_idx+1}: Error during DeepFace.verify: {e}",
                    exc_info=False
                )

    logger_obj.debug(f"    '{person_name}' NOT conclusively verified in '{os.path.basename(candidate_image_path)}' by any specified model/reference combination.")
    return False, "N/A", float('inf'), float('inf')


def verify_single_link_attempt(
    person1_name, person2_name,
    google_query_for_link,
    link_attempt_images_dl_folder,
    logger_obj,
    num_images_to_dl, # Number of candidate images to download for this link
    download_max_retries,
    download_retry_delay,
    # New parameters controlling specific verifier behaviors, supplied by main_bfs
    num_target_ref_images_to_acquire_per_person,
    ref_images_download_batch_size_for_person_pool,
    list_of_verification_models_to_use,
    deepface_detector_backend_to_use_for_verification
):
    logger_obj.info(f"--- BFS Verifying Link: '{person1_name}' <-> '{person2_name}' ---")
    logger_obj.info(f"    Link Img DLs: {num_images_to_dl}, Ref Imgs Target: {num_target_ref_images_to_acquire_per_person}, Ref DL Pool: {ref_images_download_batch_size_for_person_pool}")
    logger_obj.info(f"    Verif Models: {list_of_verification_models_to_use}, Verif Detector: {deepface_detector_backend_to_use_for_verification}")
    logger_obj.info(f"    Image Download Config: MaxRetries={download_max_retries}, RetryDelay={download_retry_delay}s")


    if not DEEPFACE_AVAILABLE:
        logger_obj.critical("DeepFace library is not available. Cannot perform image verification.")
        return "FAILED_VERIFICATION", "DeepFace library unavailable", None

    folder_after_download = image_downloader_bfs.fetch_images_for_link_bfs(
        subjects_str_or_person_name=f"{person1_name} to {person2_name}",
        google_query_str=google_query_for_link,
        root_dl_folder=link_attempt_images_dl_folder,
        num_images_to_dl=num_images_to_dl, # For candidate images
        logger=logger_obj,
        max_retries=download_max_retries,
        retry_delay=download_retry_delay,
        is_reference_download=False
    )

    if not folder_after_download:
        logger_obj.error(f"  BFS Image download preparation failed for link '{person1_name}' <-> '{person2_name}' (directory issue).")
        return "FAILED_VERIFICATION", "Download folder error", None

    candidate_images_paths = []
    if os.path.exists(link_attempt_images_dl_folder):
        for f_name in os.listdir(link_attempt_images_dl_folder):
            if f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                candidate_images_paths.append(os.path.join(link_attempt_images_dl_folder, f_name))

    if not candidate_images_paths:
        logger_obj.warning(f"  BFS No images found in '{link_attempt_images_dl_folder}' to verify (after all download attempts).")
        return "FAILED_VERIFICATION", "No images downloaded", None

    logger_obj.info(f"  BFS Found {len(candidate_images_paths)} candidate images for link. Will check all.")

    p1_ref_imgs = find_reference_images_bfs(
        person1_name, num_target_ref_images_to_acquire_per_person,
        ref_images_download_batch_size_for_person_pool,
        deepface_detector_backend_to_use_for_verification, # Use same detector for ref validation
        logger_obj, download_max_retries, download_retry_delay
    )
    p2_ref_imgs = find_reference_images_bfs(
        person2_name, num_target_ref_images_to_acquire_per_person,
        ref_images_download_batch_size_for_person_pool,
        deepface_detector_backend_to_use_for_verification, # Use same detector for ref validation
        logger_obj, download_max_retries, download_retry_delay
    )

    if not p1_ref_imgs:
        logger_obj.warning(f"  BFS Could not obtain sufficient validated reference images for '{person1_name}'. Verification might be inaccurate.")
    if not p2_ref_imgs:
        logger_obj.warning(f"  BFS Could not obtain sufficient validated reference images for '{person2_name}'. Verification might be inaccurate.")

    for idx, cand_img_path in enumerate(candidate_images_paths):
        logger_obj.info(f"  BFS Processing candidate image {idx+1}/{len(candidate_images_paths)}: {os.path.basename(cand_img_path)}")

        p1_verified, p1_model, p1_dist, p1_thr = verify_image_with_deepface_models_bfs(
            cand_img_path, person1_name, p1_ref_imgs,
            list_of_verification_models_to_use,
            deepface_detector_backend_to_use_for_verification,
            logger_obj
        )

        if p1_verified:
            logger_obj.info(f"    Found '{person1_name}' in '{os.path.basename(cand_img_path)}'. Now checking for '{person2_name}'.")
            p2_verified, p2_model, p2_dist, p2_thr = verify_image_with_deepface_models_bfs(
                cand_img_path, person2_name, p2_ref_imgs,
                list_of_verification_models_to_use,
                deepface_detector_backend_to_use_for_verification,
                logger_obj
            )

            if p2_verified:
                logger_obj.info(
                    f"    >>> SUCCESS: Both '{person1_name}' (by {p1_model}, D:{p1_dist:.4f}) AND "
                    f"'{person2_name}' (by {p2_model}, D:{p2_dist:.4f}) VERIFIED in image '{os.path.basename(cand_img_path)}'!"
                )
                verification_details = {
                    "verified_image_filename": os.path.basename(cand_img_path),
                    f"{person1_name}_verified_by_model": p1_model,
                    f"{person1_name}_distance": round(p1_dist, 4),
                    f"{person1_name}_threshold_used": round(p1_thr, 4),
                    f"{person2_name}_verified_by_model": p2_model,
                    f"{person2_name}_distance": round(p2_dist, 4),
                    f"{person2_name}_threshold_used": round(p2_thr, 4),
                    "verification_detector_backend": deepface_detector_backend_to_use_for_verification,
                }
                return "VERIFIED_OK", cand_img_path, verification_details
            else:
                logger_obj.info(f"    '{person1_name}' found, but '{person2_name}' NOT verified in '{os.path.basename(cand_img_path)}'.")
        else:
            logger_obj.info(f"    '{person1_name}' NOT verified in '{os.path.basename(cand_img_path)}'. Link verification for this image fails.")

    logger_obj.warning(
        f"  BFS Link Verification FAILED: None of the {len(candidate_images_paths)} downloaded images "
        f"could be verified for both '{person1_name}' and '{person2_name}'."
    )
    return "FAILED_VERIFICATION", "No image passed checks for both persons", None


if __name__ == '__main__':
    test_logger_verifier = logging.getLogger("TestImageVerifierStandalone")
    test_logger_verifier.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    if not test_logger_verifier.handlers:
        test_logger_verifier.addHandler(ch)
    test_logger_verifier.propagate = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dummy_output_bfs_root = os.path.join(script_dir, "output_bfs_test_verifier")

    original_bfs_ref_dir = BFS_REFERENCE_IMAGES_BASE_DIR
    BFS_REFERENCE_IMAGES_BASE_DIR = os.path.join(dummy_output_bfs_root, "reference_faces_bfs")

    temp_link_folder_name = "temp_link_attempt_standalone_test"
    temp_link_folder_path = os.path.join(dummy_output_bfs_root, "temp_files", temp_link_folder_name)

    _ensure_dir_bfs(dummy_output_bfs_root, test_logger_verifier)
    _ensure_dir_bfs(BFS_REFERENCE_IMAGES_BASE_DIR, test_logger_verifier)
    _ensure_dir_bfs(os.path.join(dummy_output_bfs_root, "temp_files"), test_logger_verifier)

    if os.path.exists(temp_link_folder_path):
        shutil.rmtree(temp_link_folder_path)
    _ensure_dir_bfs(temp_link_folder_path, test_logger_verifier)

    test_person1 = "Elon Musk"
    test_person2 = "Mark Zuckerberg"
    test_query = f'"{test_person1}" "{test_person2}" event photo'

    test_logger_verifier.info(f"--- Starting Standalone Test for image_verifier_bfs ---")
    test_logger_verifier.info(f"DeepFace Available: {DEEPFACE_AVAILABLE}")
    test_logger_verifier.info(f"Reference Image Base Directory (for this test): {BFS_REFERENCE_IMAGES_BASE_DIR}")
    test_logger_verifier.info(f"Temporary Link Images Directory (for this test): {temp_link_folder_path}")

    if not DEEPFACE_AVAILABLE:
        test_logger_verifier.error("DeepFace is not available. Cannot run a meaningful test. Please install DeepFace.")
    else:
        status, data, details = verify_single_link_attempt(
            person1_name=test_person1,
            person2_name=test_person2,
            google_query_for_link=test_query,
            link_attempt_images_dl_folder=temp_link_folder_path,
            logger_obj=test_logger_verifier,
            num_images_to_dl=3,
            download_max_retries=2,
            download_retry_delay=5,
            # Provide values for the new mandatory args, using module defaults for test:
            num_target_ref_images_to_acquire_per_person=DEFAULT_NUM_TARGET_REF_IMAGES,
            ref_images_download_batch_size_for_person_pool=DEFAULT_NUM_DOWNLOAD_FOR_REF_POOL,
            list_of_verification_models_to_use=DEFAULT_VERIF_MODELS,
            deepface_detector_backend_to_use_for_verification=DEFAULT_DETECTOR_BACKEND
        )
        print(f"\n--- Standalone Verification Test Result ---")
        print(f"Status: {status}")
        print(f"Data (Verified Image Path or Msg): {data}")
        print(f"Details: {details}")
        print(f"Check reference image folder: {BFS_REFERENCE_IMAGES_BASE_DIR}/{test_person1.replace(' ','_')} and {test_person2.replace(' ','_')}")
        print(f"Check candidate images folder: {temp_link_folder_path}")

    BFS_REFERENCE_IMAGES_BASE_DIR = original_bfs_ref_dir