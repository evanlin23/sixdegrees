import google.generativeai as genai
import os
import xml.etree.ElementTree as ET
from PIL import Image
from deepface import DeepFace
import logging
import time

import image_downloader # Assuming image_downloader.py is in the same directory or accessible

TEXT_MODEL_NAME_GEMINI = "gemini-1.5-flash-latest"
VISION_MODEL_NAME_GEMINI = "gemini-1.5-flash-latest"

REFERENCE_IMAGES_BASE_DIR = "reference_faces"
DEEPFACE_MODEL_NAME = "ArcFace"
DEEPFACE_DETECTOR_BACKEND = "retinaface"
DEEPFACE_DISTANCE_METRIC = "cosine" # Explicitly define, ArcFace typically uses cosine
# To make DeepFace "less strict", we accept a slightly HIGHER distance value as a match.
# Default ArcFace cosine threshold is ~0.68. A value like 0.72 here is more lenient.
# Lower distance = more similar. We check if distance <= threshold.
DEEPFACE_CUSTOM_THRESHOLDS = {
    "ArcFace": { # Model name
        "cosine": 0.74,  # distance_metric: threshold_value for match
        "euclidean_l2": 1.18 # Example if euclidean_l2 was used (ArcFace default is ~1.13)
    },
    "VGG-Face": { # Example for another model
        "cosine": 0.45 # Default is ~0.40
    }
    # Add other models and metrics as needed if you change DEEPFACE_MODEL_NAME
}

DEEPFACE_REFERENCE_DOWNLOAD_DELAY = 5
NUM_REFERENCE_IMAGES_TO_USE = 5 # Number of reference images to try and use

# Placeholder for API call delays - ensure this is defined appropriately in your main execution context
API_CALL_DELAY_SECONDS_CONFIG = {
    "gemini_vision": 10,
    "gemini_text_retry": 15, # Added for query_gemini_text_for_retry
    "default": 5
}

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
    """Ensures the base and person-specific reference folders exist. Returns person_ref_folder_path or None."""
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
    """Lists existing valid image files in a given folder, sorted."""
    if not os.path.isdir(person_folder_path):
        return []
    try:
        image_files = [os.path.join(person_folder_path, f)
                       for f in os.listdir(person_folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Ensure consistent order
        return image_files
    except Exception as e:
        logger.error(f"Error listing files in {person_folder_path}: {e}")
        return []


def _download_additional_reference_images(person_name, person_ref_folder_path, num_to_download, logger_obj):
    """Attempts to download a specific number of additional reference images."""
    logger = logger_obj
    if num_to_download <= 0:
        logger.debug(f"No additional reference images needed for {person_name}.")
        return True # Nothing to download

    logger.info(f"Attempting download of {num_to_download} additional reference image(s) for {person_name} into {person_ref_folder_path}.")
    
    # --- MODIFIED QUERY for diverse poses ---
    # This query attempts to get a mix of frontal, profile, and 3/4 views.
    # "headshot" is also a good general term.
    # The effectiveness of "OR" depends on how Google interprets it in image search via icrawler.
    # Often, Google might prioritize the first terms more.
    # For truly diverse sets, multiple separate crawls with specific pose queries might be needed,
    # but that would require more significant logic changes. This is a good first step.
    ref_query_terms = [
        f'"{person_name}" clear face photo',
        f'"{person_name}" profile face photo', # Tries to get side views
        f'"{person_name}" 3/4 face view',    # Tries to get angled views
        f'"{person_name}" headshot',
        f'"{person_name}" official photo' # Often yields good quality frontal shots
    ]
    # Join with OR. You might experiment with just one or two of these if the "OR" isn't effective.
    # For now, let's try combining them.
    ref_query = " OR ".join(ref_query_terms)
    
    logger.info(f"Using diversified reference query for {person_name}: {ref_query}")
    # --- END MODIFICATION ---

    # image_downloader.fetch_images_for_link will download into person_ref_folder_path
    # It's crucial that image_downloader.py uses file_idx_offset='auto'
    download_result_folder = image_downloader.fetch_images_for_link(
        subjects_str_or_person_name=person_name, # For logging inside downloader
        google_query_str=ref_query,
        root_dl_folder=person_ref_folder_path, # Downloader puts files directly here for ref
        num_images_to_dl=num_to_download,
        logger=logger,
        is_reference_download=True
    )

    if download_result_folder and os.path.isdir(download_result_folder):
        logger.info(f"Download process potentially added images for {person_name}. Folder: {download_result_folder}.")
        # The actual number of new files will be checked by re-listing in find_reference_image
        time.sleep(DEEPFACE_REFERENCE_DOWNLOAD_DELAY) # Still useful
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

        download_successful = _download_additional_reference_images(
            person_name, person_ref_folder_path, num_needed_to_download, logger
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


def verify_image_with_vision_api(
    image_path, person1_name, person2_name,
    vision_prompt_template_str, logger_obj,
    api_provider="gemini"
    ):
    logger = logger_obj

    if api_provider == "gemini":
        time.sleep(API_CALL_DELAY_SECONDS_CONFIG.get("gemini_vision", 10))
        if not vision_prompt_template_str: logger.error(f"Vision API prompt template is missing for {api_provider}."); return "ERROR"
        try: p1_str = str(person1_name); p2_str = str(person2_name); vision_api_formatted_prompt = vision_prompt_template_str.format(person1_name=p1_str, person2_name=p2_str)
        except KeyError as e_format: logger.error(f"Failed to format Vision API prompt for {api_provider}. Missing key: {e_format}. P1: {p1_str}, P2: {p2_str}"); return "ERROR"
        logger.debug(f"  Formatted Vision API prompt for {api_provider}: {vision_api_formatted_prompt}")
    elif api_provider == "local_deepface":
        pass 
    else:
        logger.error(f"Unsupported API provider '{api_provider}' for prompt formatting step.")
        return "ERROR"

    logger.info(f"  Verifying image: {os.path.basename(image_path)} for '{person1_name}' and '{person2_name}' using {api_provider.upper()} API.")
    api_response_text = "" # Initialize for Gemini case

    try:
        if api_provider == "local_deepface":
            ref1_paths = find_reference_image(person1_name, logger, auto_download=True)
            ref2_paths = find_reference_image(person2_name, logger, auto_download=True)

            if not ref1_paths:
                logger.error(f"No reference images available for DeepFace verification for P1 '{person1_name}' after auto-download attempt.")
                return "ERROR" # Changed from NO to ERROR as it's a prerequisite failure
            if not ref2_paths:
                logger.error(f"No reference images available for DeepFace verification for P2 '{person2_name}' after auto-download attempt.")
                return "ERROR" # Changed from NO to ERROR

            # Determine the custom threshold to use for DeepFace
            model_specific_custom_thresholds = DEEPFACE_CUSTOM_THRESHOLDS.get(DEEPFACE_MODEL_NAME)
            custom_threshold_to_use = None
            use_default_verified_flag_logic = True # Fallback to original logic

            if model_specific_custom_thresholds:
                custom_threshold_to_use = model_specific_custom_thresholds.get(DEEPFACE_DISTANCE_METRIC)
            
            if custom_threshold_to_use is not None:
                use_default_verified_flag_logic = False
                logger.info(f"Using custom DeepFace threshold {custom_threshold_to_use} for model '{DEEPFACE_MODEL_NAME}' with distance metric '{DEEPFACE_DISTANCE_METRIC}'.")
            else:
                logger.warning(f"No custom DeepFace threshold defined for model '{DEEPFACE_MODEL_NAME}' and metric '{DEEPFACE_DISTANCE_METRIC}'. Will rely on DeepFace's default 'verified' flag.")

            person1_verified = False
            logger.debug(f"DeepFace: Verifying P1 '{person1_name}' using {len(ref1_paths)} reference image(s).")
            for i, ref1_path in enumerate(ref1_paths):
                if not isinstance(ref1_path, str) or not os.path.exists(ref1_path):
                    logger.warning(f"Invalid or non-existent reference path for {person1_name}: {ref1_path}. Skipping.")
                    continue
                try:
                    logger.debug(f"  P1 Check {i+1}/{len(ref1_paths)}: {os.path.basename(image_path)} vs {os.path.basename(ref1_path)}")
                    result1 = DeepFace.verify(
                        img1_path=image_path, img2_path=ref1_path,
                        model_name=DEEPFACE_MODEL_NAME, 
                        detector_backend=DEEPFACE_DETECTOR_BACKEND,
                        distance_metric=DEEPFACE_DISTANCE_METRIC, # Specify for consistency
                        enforce_detection=True, align=True
                    )
                    
                    current_check_verified = False
                    distance = result1.get('distance', float('inf'))
                    default_model_threshold = result1.get('threshold', 'N/A') # DeepFace's own threshold for this model/metric

                    if use_default_verified_flag_logic:
                        if result1.get("verified", False):
                            current_check_verified = True
                        log_suffix = f"(Dist: {distance:.4f}, Model Thr: {default_model_threshold}, Decision: DeepFace default)"
                    else: # Use custom threshold
                        if distance <= custom_threshold_to_use:
                            current_check_verified = True
                        log_suffix = f"(Dist: {distance:.4f}, Model Thr: {default_model_threshold}, Custom Thr: {custom_threshold_to_use}, Decision: Custom)"
                    
                    if current_check_verified:
                        person1_verified = True
                        logger.info(f"  DeepFace verification for {person1_name} SUCCEEDED with ref '{os.path.basename(ref1_path)}' {log_suffix}")
                        break 
                    else:
                        logger.debug(f"  DeepFace verification for {person1_name} FAILED with ref '{os.path.basename(ref1_path)}' {log_suffix}")

                except Exception as e_df_p1:
                    logger.warning(f"  DeepFace.verify call failed for {person1_name} (ref: {os.path.basename(ref1_path)}) against {image_path}: {str(e_df_p1)[:200]}")
            
            if not person1_verified:
                logger.info(f"DeepFace: {person1_name} NOT verified with any of their {len(ref1_paths)} reference images.")

            person2_verified = False
            if person1_verified: # Only check P2 if P1 was verified
                logger.debug(f"DeepFace: P1 '{person1_name}' verified. Verifying P2 '{person2_name}' using {len(ref2_paths)} reference image(s).")
                for i, ref2_path in enumerate(ref2_paths):
                    if not isinstance(ref2_path, str) or not os.path.exists(ref2_path):
                        logger.warning(f"Invalid or non-existent reference path for {person2_name}: {ref2_path}. Skipping.")
                        continue
                    try:
                        logger.debug(f"  P2 Check {i+1}/{len(ref2_paths)}: {os.path.basename(image_path)} vs {os.path.basename(ref2_path)}")
                        result2 = DeepFace.verify(
                            img1_path=image_path, img2_path=ref2_path,
                            model_name=DEEPFACE_MODEL_NAME, 
                            detector_backend=DEEPFACE_DETECTOR_BACKEND,
                            distance_metric=DEEPFACE_DISTANCE_METRIC,
                            enforce_detection=True, align=True
                        )

                        current_check_verified_p2 = False
                        distance_p2 = result2.get('distance', float('inf'))
                        default_model_threshold_p2 = result2.get('threshold', 'N/A')

                        if use_default_verified_flag_logic:
                            if result2.get("verified", False):
                                current_check_verified_p2 = True
                            log_suffix_p2 = f"(Dist: {distance_p2:.4f}, Model Thr: {default_model_threshold_p2}, Decision: DeepFace default)"
                        else: # Use custom threshold
                            if distance_p2 <= custom_threshold_to_use:
                                current_check_verified_p2 = True
                            log_suffix_p2 = f"(Dist: {distance_p2:.4f}, Model Thr: {default_model_threshold_p2}, Custom Thr: {custom_threshold_to_use}, Decision: Custom)"

                        if current_check_verified_p2:
                            person2_verified = True
                            logger.info(f"  DeepFace verification for {person2_name} SUCCEEDED with ref '{os.path.basename(ref2_path)}' {log_suffix_p2}")
                            break
                        else:
                            logger.debug(f"  DeepFace verification for {person2_name} FAILED with ref '{os.path.basename(ref2_path)}' {log_suffix_p2}")
                            
                    except Exception as e_df_p2:
                        logger.warning(f"  DeepFace.verify call failed for {person2_name} (ref: {os.path.basename(ref2_path)}) against {image_path}: {str(e_df_p2)[:200]}")
                
                if not person2_verified:
                    logger.info(f"DeepFace: {person2_name} NOT verified with any of their {len(ref2_paths)} reference images (though {person1_name} was).")
            else: # P1 not verified
                 logger.info(f"DeepFace: Skipping P2 '{person2_name}' check as P1 '{person1_name}' was not verified.")

            if person1_verified and person2_verified:
                logger.info(f"DeepFace: Both {person1_name} and {person2_name} successfully verified in {os.path.basename(image_path)} using their respective reference sets.")
                return "YES"
            else:
                logger.info(f"DeepFace: One or both not verified. P1_verified: {person1_verified}, P2_verified: {person2_verified}")
                return "NO"

        elif api_provider == "gemini":
            img = Image.open(image_path); model = genai.GenerativeModel(VISION_MODEL_NAME_GEMINI)
            response = model.generate_content([vision_api_formatted_prompt, img])
            
            # Extract text response
            api_response_text = "" # Initialize
            if response.parts:
                for part in response.parts:
                    if hasattr(part, 'text'): 
                        api_response_text = part.text # Keep original for logging
                        break
            elif hasattr(response, 'text'): 
                api_response_text = response.text
            else: 
                logger.warning(f"    {api_provider.upper()} Vision API returned no parsable text part for {os.path.basename(image_path)}.")
                # api_response_text remains ""

            logger.info(f"    {api_provider.upper()} Vision API raw response for {os.path.basename(image_path)}: '{api_response_text}'")
            
            processed_response_text = api_response_text.strip().upper()
            
            if processed_response_text == "YES":
                return "YES"
            if processed_response_text == "NO":
                return "NO"
            
            # Allow for simple variations like "YES." or "NO," but not longer phrases
            if processed_response_text.startswith("YES") and len(processed_response_text) <= 5: 
                 if all(c in "YES.!, " for c in processed_response_text): # check for only allowed chars
                    logger.debug(f"Interpreting '{api_response_text}' as YES due to simple fuzzy match.")
                    return "YES"
            if processed_response_text.startswith("NO") and len(processed_response_text) <= 4: 
                 if all(c in "NO.!, " for c in processed_response_text):
                    logger.debug(f"Interpreting '{api_response_text}' as NO due to simple fuzzy match.")
                    return "NO"

            # If still ambiguous, treat as NO (less strict than ERROR which might halt chain)
            logger.warning(f"    {api_provider.upper()} Vision API gave ambiguous answer: '{api_response_text}' for {os.path.basename(image_path)} (did not clearly parse to YES/NO). Treating as NO."); 
            return "NO" 
        else:
            logger.error(f"Unsupported API provider during call: {api_provider}"); return "ERROR"

    except FileNotFoundError: logger.error(f"  Image file not found for Vision API: {image_path}"); return "ERROR"
    except Exception as e:
        logger.error(f"  Error during {api_provider.upper()} Vision call for {image_path}: {e}", exc_info=True)
        if api_provider == "gemini" and 'response' in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            logger.warning(f"    Prompt Feedback from Gemini Vision API: {response.prompt_feedback}")
        return "ERROR" # Keep as ERROR for unexpected exceptions

def verify_and_potentially_reprompt_link(
    person1_in_link, person2_in_link, images_folder_path,
    original_link_xml_node, system_prompt_content,
    retry_user_input_template_str, vision_api_prompt_template_str,
    logger_obj, max_images_to_check_vision = 3,
    vision_api_provider = "gemini" 
    ):
    logger = logger_obj
    logger.info(f"--- Verifying link segment: '{person1_in_link}' <-> '{person2_in_link}' (using {vision_api_provider.upper()}) ---")

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
    logger.info(f"  Found {len(image_files)} images in '{images_folder_path}'. Will check up to {len(image_files_to_check)} with {vision_api_provider.upper()} Vision API.")

    verified_image_path = None
    for img_path in image_files_to_check:
        verification_result = verify_image_with_vision_api(img_path, person1_in_link, person2_in_link, vision_api_prompt_template_str, logger, api_provider=vision_api_provider)
        if verification_result == "YES":
            verified_image_path = img_path; logger.info(f"  SUCCESS: Verified '{person1_in_link}' and '{person2_in_link}' in {os.path.basename(img_path)} using {vision_api_provider.upper()}"); break
        elif verification_result == "ERROR": 
            logger.warning(f"  Vision API error or file issue for image {os.path.basename(img_path)} using {vision_api_provider.upper()}. This image will be skipped. An alternative link might be sought if all images fail.")
        # Implicit: if "NO", continue to next image

    original_link_xml_string = ET.tostring(original_link_xml_node, encoding='unicode')
    if verified_image_path:
        return "VERIFIED_OK", (verified_image_path, original_link_xml_string)
    else:
        logger.warning(f"  Verification FAILED for all {len(image_files_to_check)} checked images for '{person1_in_link}' and '{person2_in_link}' using {vision_api_provider.upper()}.")
        failed_event_desc = original_link_xml_node.findtext('evidence', 'Previously suggested event'); failed_google_query = original_link_xml_node.findtext('google', 'Previously suggested query'); failed_details = {"event_description": failed_event_desc, "google_query": failed_google_query}
        formatted_retry_user_prompt_xml = format_retry_prompt(person1_in_link, person2_in_link, failed_details, retry_user_input_template_str)
        if not formatted_retry_user_prompt_xml: return "FAILED_VERIFICATION_NO_ALTERNATIVE", original_link_xml_string
        logger.info(f"  Reprompting Gemini for an alternative link (due to image verification failure with {vision_api_provider.upper()}).")
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
