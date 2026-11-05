# image_verifier.py
import google.generativeai as genai
import os
import xml.etree.ElementTree as ET
# from PIL import Image # Retained for future use, not directly for DeepFace verify
from deepface import DeepFace
import logging
import time
import cv2 # Added for collage detection
import numpy as np # Added for collage detection

import image_downloader 

TEXT_MODEL_NAME_GEMINI = "gemini-1.5-flash-latest"

REFERENCE_IMAGES_BASE_DIR = "reference_faces"
REFERENCE_IMAGE_QUERY_PATTERNS_PATH = os.path.join("prompts", "reference_image_query_patterns.txt")


DEEPFACE_MODELS_TO_USE = ["ArcFace", "VGG-Face", "Facenet"] 
DEEPFACE_DETECTOR_BACKEND = "retinaface"
DEEPFACE_DISTANCE_METRIC = "cosine" 

DEEPFACE_CUSTOM_THRESHOLDS = {
    "ArcFace": {
        "cosine": 0.68, 
        "euclidean_l2": 1.13
    },
    "VGG-Face": {
        "cosine": 0.45, 
        "euclidean_l2": 0.80 
    },
    "Facenet": {
        "cosine": 0.40, 
        "euclidean_l2": 1.0 
    },
    "Facenet512": { 
        "cosine": 0.30, 
        "euclidean_l2": 0.80 
    },
    "SFace": { 
        "cosine": 0.593 
    }
}
DEEPFACE_VOTING_THRESHOLD_PERCENT = 50


DEEPFACE_REFERENCE_DOWNLOAD_DELAY = 3 
NUM_REFERENCE_IMAGES_TO_USE = 5 

API_CALL_DELAY_SECONDS_CONFIG = {
    "gemini_text_retry": 15,
    "default": 5
}

# --- Collage Detection Heuristic Parameters ---
COLLAGE_CANNY_THRESH1 = 50
COLLAGE_CANNY_THRESH2 = 150
COLLAGE_HOUGH_THRESHOLD_RATIO = 0.10 # Ratio of min(height,width) for Hough accumulator votes
COLLAGE_HOUGH_MIN_LINE_LENGTH_RATIO = 0.3 # Ratio of min(height,width) for min line length
COLLAGE_HOUGH_MAX_LINE_GAP_RATIO = 0.05  # Ratio of min(height,width) for max line gap
COLLAGE_SIGNIFICANT_LINE_LENGTH_RATIO = 0.6 # A line is significant if its length > this ratio * image dimension
COLLAGE_LINE_ANGLE_TOLERANCE_DEGREES = 5.0 # Tolerance for classifying line as horizontal/vertical
COLLAGE_MIN_SIGNIFICANT_LINES_TO_FLAG = 1 # If this many significant dividing lines are found, flag as collage


def is_likely_collage_heuristic(image_path, logger):
    """
    Heuristic check to see if an image is likely a collage by detecting strong dividing lines.
    Returns True if likely a collage, False otherwise.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Collage Check: Could not read image {image_path}")
            return False 

        height, width = img.shape[:2]
        if height == 0 or width == 0:
            logger.warning(f"Collage Check: Invalid image dimensions for {image_path}")
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, COLLAGE_CANNY_THRESH1, COLLAGE_CANNY_THRESH2)

        # Adaptive thresholds for HoughLinesP
        min_dim = min(height, width)
        hough_thresh = int(min_dim * COLLAGE_HOUGH_THRESHOLD_RATIO)
        hough_min_len = int(min_dim * COLLAGE_HOUGH_MIN_LINE_LENGTH_RATIO)
        hough_max_gap = int(min_dim * COLLAGE_HOUGH_MAX_LINE_GAP_RATIO)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thresh,
                                minLineLength=hough_min_len, maxLineGap=hough_max_gap)

        if lines is None:
            return False # No lines detected, unlikely to be a grid-like collage

        num_significant_vertical_lines = 0
        num_significant_horizontal_lines = 0

        for line_segment in lines:
            x1, y1, x2, y2 = line_segment[0]
            line_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)

            # Normalize angle to be between 0 and 180 for easier checking
            if angle_deg < 0:
                angle_deg += 180
            
            # Check for Vertical lines (around 90 degrees)
            if abs(angle_deg - 90) < COLLAGE_LINE_ANGLE_TOLERANCE_DEGREES:
                if line_len > height * COLLAGE_SIGNIFICANT_LINE_LENGTH_RATIO:
                    # Further check: not too close to image borders
                    if min(x1, x2) > width * 0.05 and max(x1, x2) < width * 0.95:
                        num_significant_vertical_lines += 1
            # Check for Horizontal lines (around 0 or 180 degrees)
            elif angle_deg < COLLAGE_LINE_ANGLE_TOLERANCE_DEGREES or \
                 abs(angle_deg - 180) < COLLAGE_LINE_ANGLE_TOLERANCE_DEGREES:
                if line_len > width * COLLAGE_SIGNIFICANT_LINE_LENGTH_RATIO:
                     # Further check: not too close to image borders
                    if min(y1, y2) > height * 0.05 and max(y1, y2) < height * 0.95:
                        num_significant_horizontal_lines += 1
        
        if num_significant_vertical_lines >= COLLAGE_MIN_SIGNIFICANT_LINES_TO_FLAG or \
           num_significant_horizontal_lines >= COLLAGE_MIN_SIGNIFICANT_LINES_TO_FLAG:
            logger.info(f"Collage Check: Image {os.path.basename(image_path)} flagged as likely collage. "
                        f"Found {num_significant_vertical_lines} vertical and {num_significant_horizontal_lines} horizontal significant lines.")
            return True

        return False
    except Exception as e:
        logger.error(f"Collage Check: Error processing image {image_path}: {e}", exc_info=False) # exc_info=False to avoid too much noise for a heuristic
        return False # Err on the side of caution, treat as not a collage if error occurs

def _load_reference_query_patterns(logger_obj):
    try:
        with open(REFERENCE_IMAGE_QUERY_PATTERNS_PATH, "r", encoding="utf-8") as f:
            patterns = [line.strip() for line in f if line.strip()]
        if not patterns:
            logger_obj.warning(f"Reference image query patterns file '{REFERENCE_IMAGE_QUERY_PATTERNS_PATH}' is empty or contains no valid patterns. Falling back to default.")
            return ['"{person_name}" clear face photo', '"{person_name}" headshot'] # Fallback
        logger_obj.info(f"Loaded {len(patterns)} reference image query patterns from {REFERENCE_IMAGE_QUERY_PATTERNS_PATH}")
        return patterns
    except FileNotFoundError:
        logger_obj.error(f"Reference image query patterns file not found: {REFERENCE_IMAGE_QUERY_PATTERNS_PATH}. Falling back to default.")
        return ['"{person_name}" clear face photo', '"{person_name}" headshot'] # Fallback
    except Exception as e:
        logger_obj.error(f"Error loading reference image query patterns from {REFERENCE_IMAGE_QUERY_PATTERNS_PATH}: {e}. Falling back to default.")
        return ['"{person_name}" clear face photo', '"{person_name}" headshot'] # Fallback


def clean_gemini_xml_response_verifier(ai_output, logger):
    logger.debug(f"Raw AI output before cleaning (verifier response) (first 300 chars): {ai_output[:300]}")
    ai_output = ai_output.strip()
    if ai_output.startswith("```xml"): ai_output = ai_output[len("```xml"):]; ai_output = ai_output[:-3] if ai_output.endswith("```") else ai_output; ai_output = ai_output.strip()
    elif ai_output.startswith("```"): ai_output = ai_output[3:]; ai_output = ai_output[:-3] if ai_output.endswith("```") else ai_output; ai_output = ai_output.strip()
    first_angle_bracket = ai_output.find('<')
    if first_angle_bracket > 0: logger.debug(f"Stripping leading non-XML: '{ai_output[:first_angle_bracket]}'"); ai_output = ai_output[first_angle_bracket:]
    elif first_angle_bracket == -1 and ai_output.strip() and not ai_output.startswith("<"): logger.warning("Verifier AI resp no XML start"); return f"<error><type>MalformedResponse</type><message>AI resp (verifier) no XML start.</message><raw_response_snippet>{ai_output[:200]}</raw_response_snippet></error>"
    logger.debug(f"AI output after cleaning (verifier resp) (first 300 chars): {ai_output[:300]}"); return ai_output

def query_gemini_text_for_retry(system_prompt, user_prompt_xml, logger_obj):
    logger = logger_obj; logger.info("Sending text prompt to Gemini for retry (from image_verifier)...")
    if not system_prompt: logger.error("System prompt missing for query_gemini_text_for_retry."); return "<error><message>System prompt is missing.</message></error>"
    full_prompt = f"{system_prompt}\n{user_prompt_xml}"; logger.debug(f"User part of prompt for Gemini text (verifier retry):\n{user_prompt_xml[:500]}")
    try:
        time.sleep(API_CALL_DELAY_SECONDS_CONFIG.get("gemini_text_retry", 15))
        model = genai.GenerativeModel(TEXT_MODEL_NAME_GEMINI); response = model.generate_content(full_prompt)
        ai_output = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        cleaned_output = clean_gemini_xml_response_verifier(ai_output, logger); logger.debug(f"Cleaned Gemini response from text query (verifier retry): {cleaned_output[:300]}"); return cleaned_output
    except Exception as e:
        logger.error(f"Error querying Gemini (text) in verifier for retry: {e}", exc_info=True)
        prompt_feedback_info = "";_ = locals()
        if 'response' in _ and hasattr(_['response'], 'prompt_feedback') and _['response'].prompt_feedback: logger.warning(f"Prompt Feedback (verifier text retry): {response.prompt_feedback}"); prompt_feedback_info = f"<prompt_feedback>{response.prompt_feedback}</prompt_feedback>"
        return f"<error><type>APIError</type><message>Gemini text API call (verifier retry) failed: {str(e)}</message>{prompt_feedback_info}</error>"

def _ensure_reference_folder(person_name, logger):
    if not os.path.exists(REFERENCE_IMAGES_BASE_DIR):
        try:
            os.makedirs(REFERENCE_IMAGES_BASE_DIR)
            logger.info(f"Created base reference directory: {REFERENCE_IMAGES_BASE_DIR}")
        except OSError as e:
            logger.error(f"Could not create base reference directory {REFERENCE_IMAGES_BASE_DIR}: {e}")
            return None

    person_folder_name_sanitized = image_downloader.sanitize_filename(person_name)
    person_ref_folder_path = os.path.join(REFERENCE_IMAGES_BASE_DIR, person_folder_name_sanitized)

    if not os.path.exists(person_ref_folder_path):
        try:
            os.makedirs(person_ref_folder_path)
            logger.info(f"Created reference folder for {person_name}: {person_ref_folder_path}")
        except OSError as e:
            logger.error(f"Could not create reference folder {person_ref_folder_path} for {person_name}: {e}")
            return None
    return person_ref_folder_path

def _get_existing_reference_images(person_folder_path, logger):
    if not os.path.isdir(person_folder_path):
        return []
    try:
        image_files = [os.path.join(person_folder_path, f)
                       for f in os.listdir(person_folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort() 
        return image_files
    except Exception as e:
        logger.error(f"Error listing files in {person_folder_path}: {e}")
        return []


def _download_additional_reference_images(person_name, person_ref_folder_path, num_to_download, logger_obj, query_patterns):
    logger = logger_obj
    if num_to_download <= 0:
        logger.debug(f"No additional reference images needed for {person_name}.")
        return True 

    logger.info(f"Attempting download of {num_to_download} additional reference image(s) for {person_name} into {person_ref_folder_path}.")
    
    formatted_query_terms = [pattern.format(person_name=person_name) for pattern in query_patterns]
    ref_query = " OR ".join(formatted_query_terms)
    
    logger.info(f"Using diversified reference query for {person_name}: {ref_query}")

    download_result_folder = image_downloader.fetch_images_for_link(
        subjects_str_or_person_name=person_name, 
        google_query_str=ref_query,
        root_dl_folder=person_ref_folder_path, 
        num_images_to_dl=num_to_download,
        logger=logger,
        is_reference_download=True
    )

    if download_result_folder and os.path.isdir(download_result_folder):
        logger.info(f"Download process potentially added images for {person_name}. Folder: {download_result_folder}.")
        time.sleep(DEEPFACE_REFERENCE_DOWNLOAD_DELAY) 
        return True
    else:
        logger.warning(f"Failed to download additional reference images for {person_name} or download folder not confirmed by downloader.")
        return False

def find_reference_image(person_name, logger_obj, auto_download=True):
    logger = logger_obj
    logger.info(f"Finding reference images for: {person_name} (target count: {NUM_REFERENCE_IMAGES_TO_USE})")

    person_ref_folder_path = _ensure_reference_folder(person_name, logger)
    if not person_ref_folder_path:
        logger.error(f"Could not ensure reference folder for '{person_name}'. Cannot provide reference images.")
        return []

    existing_images = _get_existing_reference_images(person_ref_folder_path, logger)
    logger.debug(f"Found {len(existing_images)} existing reference(s) for '{person_name}' in {person_ref_folder_path}.")

    if len(existing_images) >= NUM_REFERENCE_IMAGES_TO_USE:
        logger.info(f"Sufficient reference images ({len(existing_images)}) already exist for '{person_name}'. Using top {NUM_REFERENCE_IMAGES_TO_USE}.")
        return existing_images[:NUM_REFERENCE_IMAGES_TO_USE]

    if auto_download:
        num_needed_to_download = NUM_REFERENCE_IMAGES_TO_USE - len(existing_images)
        logger.info(f"Need {num_needed_to_download} more reference images for '{person_name}'. Triggering download.")
        
        query_patterns = _load_reference_query_patterns(logger)

        download_successful = _download_additional_reference_images(
            person_name, person_ref_folder_path, num_needed_to_download, logger, query_patterns
        )

        all_images_after_attempt = _get_existing_reference_images(person_ref_folder_path, logger)
        if download_successful:
            logger.info(f"Found {len(all_images_after_attempt)} total reference(s) for '{person_name}' after download attempt.")
        else:
            logger.warning(f"Download attempt for additional references may have failed or yielded no new images for '{person_name}'. Using {len(all_images_after_attempt)} available images.")
        return all_images_after_attempt[:NUM_REFERENCE_IMAGES_TO_USE]
    else: 
        if not existing_images:
            logger.warning(f"No local reference images for '{person_name}' at '{person_ref_folder_path}' and auto-download is OFF.")
            return [] 
        logger.info(f"Auto-download is OFF. Using {len(existing_images)} existing reference(s) for '{person_name}' (up to {NUM_REFERENCE_IMAGES_TO_USE}).")
        return existing_images[:NUM_REFERENCE_IMAGES_TO_USE]


def verify_image_with_deepface_models(
    image_path, person1_name, person2_name, logger_obj
    ):
    logger = logger_obj
    logger.info(f"  Verifying image: {os.path.basename(image_path)} for '{person1_name}' and '{person2_name}' using DeepFace (multiple models: {', '.join(DEEPFACE_MODELS_TO_USE)}).")

    if not DEEPFACE_MODELS_TO_USE:
        logger.error("  DEEPFACE_MODELS_TO_USE list is empty. Cannot perform DeepFace verification.")
        return "ERROR"

    try:
        ref1_paths = find_reference_image(person1_name, logger, auto_download=True)
        ref2_paths = find_reference_image(person2_name, logger, auto_download=True)

        if not ref1_paths:
            logger.error(f"No reference images available for DeepFace verification for P1 '{person1_name}' after auto-download attempt.")
            return "ERROR"
        if not ref2_paths:
            logger.error(f"No reference images available for DeepFace verification for P2 '{person2_name}' after auto-download attempt.")
            return "ERROR"

        p1_model_votes = {} 
        logger.info(f"  DeepFace: Verifying P1 '{person1_name}' using {len(ref1_paths)} reference image(s) across {len(DEEPFACE_MODELS_TO_USE)} model(s).")
        for model_idx, current_model_name in enumerate(DEEPFACE_MODELS_TO_USE):
            logger.debug(f"    P1 Model {model_idx+1}/{len(DEEPFACE_MODELS_TO_USE)}: '{current_model_name}'")
            person1_verified_by_current_model = False
            
            custom_threshold_to_use = None
            use_default_verified_flag_logic = True
            model_specific_custom_thresholds = DEEPFACE_CUSTOM_THRESHOLDS.get(current_model_name)
            if model_specific_custom_thresholds:
                custom_threshold_to_use = model_specific_custom_thresholds.get(DEEPFACE_DISTANCE_METRIC)
            
            if custom_threshold_to_use is not None:
                use_default_verified_flag_logic = False
                logger.debug(f"      Using custom threshold {custom_threshold_to_use} for model '{current_model_name}' with metric '{DEEPFACE_DISTANCE_METRIC}'.")
            else:
                logger.debug(f"      No custom threshold for model '{current_model_name}' / metric '{DEEPFACE_DISTANCE_METRIC}'. Using DeepFace default 'verified' flag.")

            for i, ref1_path in enumerate(ref1_paths):
                if not isinstance(ref1_path, str) or not os.path.exists(ref1_path):
                    logger.warning(f"      Invalid or non-existent reference path for {person1_name} with {current_model_name}: {ref1_path}. Skipping.")
                    continue
                try:
                    logger.debug(f"        P1 Check (Model: {current_model_name}, Ref {i+1}/{len(ref1_paths)}): {os.path.basename(image_path)} vs {os.path.basename(ref1_path)}")
                    result1 = DeepFace.verify(
                        img1_path=image_path, img2_path=ref1_path,
                        model_name=current_model_name, 
                        detector_backend=DEEPFACE_DETECTOR_BACKEND,
                        distance_metric=DEEPFACE_DISTANCE_METRIC,
                        enforce_detection=True, align=True
                    )
                    
                    current_check_verified = False
                    distance = result1.get('distance', float('inf'))
                    default_model_threshold = result1.get('threshold', 'N/A')

                    if use_default_verified_flag_logic:
                        if result1.get("verified", False): current_check_verified = True
                        log_suffix = f"(Dist: {distance:.4f}, ModelThr: {default_model_threshold}, Decision: DeepFace default)"
                    else: 
                        if distance <= custom_threshold_to_use: current_check_verified = True
                        log_suffix = f"(Dist: {distance:.4f}, ModelThr: {default_model_threshold}, CustomThr: {custom_threshold_to_use}, Decision: Custom)"
                    
                    if current_check_verified:
                        person1_verified_by_current_model = True
                        logger.info(f"      P1 verification with {current_model_name} SUCCEEDED (Ref: '{os.path.basename(ref1_path)}') {log_suffix}")
                        break 
                    else:
                        logger.debug(f"      P1 verification with {current_model_name} FAILED (Ref: '{os.path.basename(ref1_path)}') {log_suffix}")
                except Exception as e_df_p1_model:
                    logger.warning(f"      DeepFace.verify call failed for P1 {person1_name} with model {current_model_name} (Ref: {os.path.basename(ref1_path)}) against {image_path}: {str(e_df_p1_model)[:200]}")
            
            p1_model_votes[current_model_name] = person1_verified_by_current_model
            logger.info(f"    P1 Result for model '{current_model_name}': {'VERIFIED' if person1_verified_by_current_model else 'NOT VERIFIED'}")

        num_p1_yes_votes = sum(1 for vote in p1_model_votes.values() if vote)
        final_p1_verified = (num_p1_yes_votes * 100.0 / len(DEEPFACE_MODELS_TO_USE)) > DEEPFACE_VOTING_THRESHOLD_PERCENT if DEEPFACE_MODELS_TO_USE else False
        logger.info(f"  DeepFace P1 '{person1_name}' Voting: {num_p1_yes_votes}/{len(DEEPFACE_MODELS_TO_USE)} models voted YES. Final P1 Verified: {final_p1_verified} (Threshold: >{DEEPFACE_VOTING_THRESHOLD_PERCENT}%)")

        p2_model_votes = {}
        final_p2_verified = False
        if final_p1_verified:
            logger.info(f"  DeepFace: P1 '{person1_name}' verified by vote. Verifying P2 '{person2_name}' using {len(ref2_paths)} reference image(s) across {len(DEEPFACE_MODELS_TO_USE)} model(s).")
            for model_idx, current_model_name in enumerate(DEEPFACE_MODELS_TO_USE):
                logger.debug(f"    P2 Model {model_idx+1}/{len(DEEPFACE_MODELS_TO_USE)}: '{current_model_name}'")
                person2_verified_by_current_model = False

                custom_threshold_to_use = None
                use_default_verified_flag_logic = True
                model_specific_custom_thresholds = DEEPFACE_CUSTOM_THRESHOLDS.get(current_model_name)
                if model_specific_custom_thresholds:
                    custom_threshold_to_use = model_specific_custom_thresholds.get(DEEPFACE_DISTANCE_METRIC)
                
                if custom_threshold_to_use is not None:
                    use_default_verified_flag_logic = False 
                
                for i, ref2_path in enumerate(ref2_paths):
                    if not isinstance(ref2_path, str) or not os.path.exists(ref2_path):
                        logger.warning(f"      Invalid or non-existent reference path for {person2_name} with {current_model_name}: {ref2_path}. Skipping.")
                        continue
                    try:
                        logger.debug(f"        P2 Check (Model: {current_model_name}, Ref {i+1}/{len(ref2_paths)}): {os.path.basename(image_path)} vs {os.path.basename(ref2_path)}")
                        result2 = DeepFace.verify(
                            img1_path=image_path, img2_path=ref2_path,
                            model_name=current_model_name, 
                            detector_backend=DEEPFACE_DETECTOR_BACKEND,
                            distance_metric=DEEPFACE_DISTANCE_METRIC,
                            enforce_detection=True, align=True
                        )

                        current_check_verified_p2 = False
                        distance_p2 = result2.get('distance', float('inf'))
                        default_model_threshold_p2 = result2.get('threshold', 'N/A')

                        if use_default_verified_flag_logic:
                            if result2.get("verified", False): current_check_verified_p2 = True
                            log_suffix_p2 = f"(Dist: {distance_p2:.4f}, ModelThr: {default_model_threshold_p2}, Decision: DeepFace default)"
                        else: 
                            if distance_p2 <= custom_threshold_to_use: current_check_verified_p2 = True
                            log_suffix_p2 = f"(Dist: {distance_p2:.4f}, ModelThr: {default_model_threshold_p2}, CustomThr: {custom_threshold_to_use}, Decision: Custom)"

                        if current_check_verified_p2:
                            person2_verified_by_current_model = True
                            logger.info(f"      P2 verification with {current_model_name} SUCCEEDED (Ref: '{os.path.basename(ref2_path)}') {log_suffix_p2}")
                            break
                        else:
                            logger.debug(f"      P2 verification with {current_model_name} FAILED (Ref: '{os.path.basename(ref2_path)}') {log_suffix_p2}")
                    except Exception as e_df_p2_model:
                        logger.warning(f"      DeepFace.verify call failed for P2 {person2_name} with model {current_model_name} (Ref: {os.path.basename(ref2_path)}) against {image_path}: {str(e_df_p2_model)[:200]}")
                
                p2_model_votes[current_model_name] = person2_verified_by_current_model
                logger.info(f"    P2 Result for model '{current_model_name}': {'VERIFIED' if person2_verified_by_current_model else 'NOT VERIFIED'}")

            num_p2_yes_votes = sum(1 for vote in p2_model_votes.values() if vote)
            final_p2_verified = (num_p2_yes_votes * 100.0 / len(DEEPFACE_MODELS_TO_USE)) > DEEPFACE_VOTING_THRESHOLD_PERCENT if DEEPFACE_MODELS_TO_USE else False
            logger.info(f"  DeepFace P2 '{person2_name}' Voting: {num_p2_yes_votes}/{len(DEEPFACE_MODELS_TO_USE)} models voted YES. Final P2 Verified: {final_p2_verified} (Threshold: >{DEEPFACE_VOTING_THRESHOLD_PERCENT}%)")
        else:
            logger.info(f"  DeepFace: Skipping P2 '{person2_name}' check as P1 '{person1_name}' was not verified by vote.")

        if final_p1_verified and final_p2_verified:
            logger.info(f"  DeepFace Multi-Model Verification: Both {person1_name} and {person2_name} successfully verified by vote in {os.path.basename(image_path)}.")
            return "YES"
        else:
            logger.info(f"  DeepFace Multi-Model Verification: One or both not verified by vote. P1_verified: {final_p1_verified}, P2_verified: {final_p2_verified}")
            return "NO"

    except FileNotFoundError: 
        logger.error(f"  Image file not found for DeepFace verification: {image_path}"); return "ERROR"
    except Exception as e:
        logger.error(f"  Error during DeepFace Multi-Model verification for {image_path}: {e}", exc_info=True)
        return "ERROR"


def verify_and_potentially_reprompt_link(
    person1_in_link, person2_in_link, images_folder_path,
    original_link_xml_node, system_prompt_content,
    retry_user_input_template_str, 
    logger_obj, max_images_to_check_vision = 3
    ):
    logger = logger_obj
    logger.info(f"--- Verifying link segment: '{person1_in_link}' <-> '{person2_in_link}' (using DeepFace Multi-Model & Collage Check) ---")

    def format_retry_prompt(p1, p2, failed_details_dict, template_str):
        try: p1_str_fmt = str(p1); p2_str_fmt = str(p2); return template_str.format(person1_name=p1_str_fmt, person2_name=p2_str_fmt, failed_event_description=failed_details_dict.get('event_description', 'Previously suggested event'), failed_google_query=failed_details_dict.get('google_query', 'Previously suggested query'))
        except KeyError as e_format: logger.error(f"Failed to format retry user input template. Missing key: {e_format}. P1: {p1}, P2: {p2}. Template snippet: {template_str[:200]}"); return None

    if not images_folder_path or not os.path.exists(images_folder_path) or not os.listdir(images_folder_path):
        logger.warning(f"  No images found in the folder '{images_folder_path}' to verify for '{person1_in_link}' <-> '{person2_in_link}'.")
        failed_event_desc = original_link_xml_node.findtext('evidence', 'Unknown Event'); failed_google_query = original_link_xml_node.findtext('google', 'Unknown Query'); failed_details = {"event_description": failed_event_desc, "google_query": failed_google_query}
        formatted_retry_user_prompt_xml = format_retry_prompt(person1_in_link, person2_in_link, failed_details, retry_user_input_template_str)
        if not formatted_retry_user_prompt_xml: return "FAILED_VERIFICATION_NO_ALTERNATIVE", ET.tostring(original_link_xml_node, encoding='unicode')
        logger.info(f"  Reprompting Gemini for an alternative link (due to no images).")
        new_link_suggestion_xml = query_gemini_text_for_retry(system_prompt_content, formatted_retry_user_prompt_xml, logger)
        try:
            if not isinstance(new_link_suggestion_xml, str) or new_link_suggestion_xml.strip().startswith("<error>"):
                 logger.error(f"Reprompt for 'no images' returned an error XML or invalid type: {new_link_suggestion_xml}"); return "FAILED_VERIFICATION_NO_ALTERNATIVE", ET.tostring(original_link_xml_node, encoding='unicode')
            temp_root = ET.fromstring(new_link_suggestion_xml)
            if temp_root.tag == "no_alternative_link_found": reason = temp_root.get("reason", "N/A"); logger.info(f"  Gemini indicated no alternative link found (reprompt after no images): {reason}"); return "FAILED_VERIFICATION_NO_ALTERNATIVE", ET.tostring(original_link_xml_node, encoding='unicode')
        except ET.ParseError as e: logger.error(f"  Malformed XML response from Gemini during reprompt (after no images): {e}"); logger.debug(f"  Malformed XML: {new_link_suggestion_xml}"); return "FAILED_VERIFICATION_NO_ALTERNATIVE", ET.tostring(original_link_xml_node, encoding='unicode')
        logger.info(f"  Received new link suggestion XML after 'no images' reprompt."); logger.debug(f"  New link XML snippet: {new_link_suggestion_xml[:300]}..."); return "NEEDS_REPROMPT_NEW_LINK_PROVIDED", new_link_suggestion_xml

    image_files = [os.path.join(images_folder_path, f) for f in os.listdir(images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    image_files_to_check = image_files[:max_images_to_check_vision]
    logger.info(f"  Found {len(image_files)} images in '{images_folder_path}'. Will check up to {len(image_files_to_check)}.")

    verified_image_path = None
    for img_path in image_files_to_check:
        logger.info(f"  Processing image for verification: {os.path.basename(img_path)}")

        # Collage Check
        if is_likely_collage_heuristic(img_path, logger):
            logger.warning(f"    Image {os.path.basename(img_path)} flagged as a likely collage. Skipping DeepFace verification for this image.")
            continue # Skip to the next image if it's likely a collage

        verification_result = verify_image_with_deepface_models(
            img_path, person1_in_link, person2_in_link, logger
        )
        if verification_result == "YES":
            verified_image_path = img_path; logger.info(f"  SUCCESS: Verified '{person1_in_link}' and '{person2_in_link}' in {os.path.basename(img_path)} using DeepFace Multi-Model"); break
        elif verification_result == "ERROR": 
            logger.warning(f"  DeepFace Multi-Model error or file issue for image {os.path.basename(img_path)}. This image will be skipped.")
        # If "NO", continue to the next image

    original_link_xml_string = ET.tostring(original_link_xml_node, encoding='unicode')
    if verified_image_path:
        return "VERIFIED_OK", (verified_image_path, original_link_xml_string)
    else:
        logger.warning(f"  Verification FAILED for all {len(image_files_to_check)} checked images (or images were skipped as collages) for '{person1_in_link}' and '{person2_in_link}'.")
        failed_event_desc = original_link_xml_node.findtext('evidence', 'Previously suggested event'); failed_google_query = original_link_xml_node.findtext('google', 'Previously suggested query'); failed_details = {"event_description": failed_event_desc, "google_query": failed_google_query}
        formatted_retry_user_prompt_xml = format_retry_prompt(person1_in_link, person2_in_link, failed_details, retry_user_input_template_str)
        if not formatted_retry_user_prompt_xml: return "FAILED_VERIFICATION_NO_ALTERNATIVE", original_link_xml_string
        logger.info(f"  Reprompting Gemini for an alternative link (due to image verification failure or no suitable images).")
        new_link_suggestion_xml = query_gemini_text_for_retry(system_prompt_content, formatted_retry_user_prompt_xml, logger)
        try:
            if not isinstance(new_link_suggestion_xml, str) or new_link_suggestion_xml.strip().startswith("<error>"):
                 logger.error(f"Reprompt for 'verification failure' returned an error XML or invalid type: {new_link_suggestion_xml}"); return "FAILED_VERIFICATION_NO_ALTERNATIVE", original_link_xml_string
            temp_root = ET.fromstring(new_link_suggestion_xml)
            if temp_root.tag == "no_alternative_link_found": reason = temp_root.get("reason", "N/A"); logger.info(f"  Gemini indicated no alternative link found (reprompt after verification failure): {reason}"); return "FAILED_VERIFICATION_NO_ALTERNATIVE", original_link_xml_string
        except ET.ParseError as e: logger.error(f"  Malformed XML response from Gemini during reprompt (after verification failure): {e}"); logger.debug(f"  Malformed XML: {new_link_suggestion_xml}"); return "FAILED_VERIFICATION_NO_ALTERNATIVE", original_link_xml_string
        except Exception as e_xml_other: logger.error(f"  Unexpected error processing reprompt XML ({e_xml_other}): {new_link_suggestion_xml}"); return "FAILED_VERIFICATION_NO_ALTERNATIVE", original_link_xml_string
        logger.info(f"  Received new link suggestion XML after verification failure."); logger.debug(f"  New link XML snippet: {new_link_suggestion_xml[:300]}..."); return "NEEDS_REPROMPT_NEW_LINK_PROVIDED", new_link_suggestion_xml

if __name__ == "__main__":
    print("This script is intended to be called by main_orchestrator.py")