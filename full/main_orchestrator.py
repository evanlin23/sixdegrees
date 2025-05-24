# main_orchestrator.py

import os
import argparse
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import shutil
import logging
import time

import initial_gemini_query
import image_downloader
import image_verifier

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

PROMPTS_DIR = "prompts"
SYSTEM_PROMPT_FILE_PATH = os.path.join(PROMPTS_DIR, "system_prompt.txt")
INITIAL_USER_INPUT_TEMPLATE_PATH = os.path.join(PROMPTS_DIR, "initial_user_input_template.xml")
INITIAL_CHAIN_RETRY_USER_INPUT_TEMPLATE_PATH = os.path.join(PROMPTS_DIR, "initial_chain_retry_user_input_template.xml")
RETRY_USER_INPUT_TEMPLATE_PATH = os.path.join(PROMPTS_DIR, "retry_user_input_template.xml")
SUB_CHAIN_EXCLUSION_TEMPLATE_PATH = os.path.join(PROMPTS_DIR, "sub_chain_exclusion_instruction_template.txt") 

OUTPUT_DIR = "output"
RAW_IMAGES_DIR = os.path.join(OUTPUT_DIR, "connection_chain_images")
VERIFIED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "connection_chain_images_verified")
TEMP_FILES_DIR = os.path.join(OUTPUT_DIR, "temp_files")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

IMAGES_TO_DOWNLOAD_PER_LINK = 5
MAX_IMAGES_TO_CHECK_PER_LINK_VERIFIER = 3 
MAX_ORCHESTRATOR_SUBCHAIN_ATTEMPTS = 2
MAX_INITIAL_CHAIN_RETRY_ATTEMPTS = 1
INITIAL_CHAIN_RETRY_DELAY_SECONDS = 20

def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            if 'logger' in globals() and logger:
                logger.info(f"Created directory: {directory_path}")
            else:
                print(f"Created directory: {directory_path}")
        except OSError as e:
            if 'logger' in globals() and logger:
                logger.error(f"Failed to create directory {directory_path}: {e}")
            else:
                print(f"ERROR: Failed to create directory {directory_path}: {e}")

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "visual_chain_run.log")

logger = logging.getLogger("VisualChainApp")
if not logger.handlers: 
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_FILE_PATH, mode='w') 
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

def clean_output_directory(dir_path, logger_obj):
    if not os.path.exists(dir_path): 
        logger_obj.info(f"Directory to clean does not exist, skipping: {dir_path}")
        return
    if not os.path.isdir(dir_path): 
        logger_obj.warning(f"Path to clean is not a directory, skipping: {dir_path}")
        return
    
    logger_obj.info(f"Cleaning contents of directory: {dir_path}")
    for item_name in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path): 
                os.unlink(item_path)
                logger_obj.debug(f"  Deleted file/link: {item_path}")
            elif os.path.isdir(item_path): 
                shutil.rmtree(item_path)
                logger_obj.debug(f"  Deleted directory and its contents: {item_path}")
        except Exception as e: 
            logger_obj.error(f"  Failed to delete {item_path}. Reason: {e}")
    logger_obj.info(f"Finished cleaning directory: {dir_path}")

def load_prompt_template(filepath, is_critical=True): 
    logger.debug(f"Attempting to load prompt template from: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f: 
            content = f.read()
        logger.info(f"Prompt template loaded successfully from {filepath}")
        return content
    except FileNotFoundError: 
        log_level = logger.error if is_critical else logger.warning
        log_level(f"{'FATAL ERROR' if is_critical else 'Warning'}: Prompt template file '{filepath}' not found.")
        return None
    except Exception as e: 
        log_level = logger.error if is_critical else logger.warning
        log_level(f"Error loading prompt template file '{filepath}': {e}", exc_info=True)
        return None

def extract_persons_from_subjects(subjects_text):
    logger.debug(f"Attempting to extract persons from subjects_text: '{subjects_text}'")
    if not subjects_text or '→' not in subjects_text:
        # If it's a single name (often from reference download subjects or malformed link)
        if subjects_text and '→' not in subjects_text:
            p1_sanitized = image_downloader.sanitize_filename(subjects_text) # Sanitize for consistency
            logger.warning(f"Only one part found in subjects: '{subjects_text}'. Extracted P1: '{p1_sanitized}', using fallback for P2.")
            return p1_sanitized, "UnknownTarget_From_Single_Subject"
        logger.warning(f"Could not parse subjects: '{subjects_text}'. Using fallbacks.")
        return "UnknownPerson1", "UnknownPerson2" 
    
    parts = [p.strip() for p in subjects_text.split('→')]
    
    if len(parts) >= 2:
        p1 = parts[0]
        p2 = parts[1]
        if not p1: 
            logger.warning(f"Extracted P1 is empty from '{subjects_text}'. Using fallback.")
            p1 = "UnknownPerson1_from_empty"
        if not p2: 
            logger.warning(f"Extracted P2 is empty from '{subjects_text}'. Using fallback.")
            p2 = "UnknownPerson2_from_empty"
        logger.debug(f"Extracted P1: '{p1}', P2: '{p2}'")
        return p1, p2
    elif len(parts) == 1: 
        p1 = parts[0]
        if not p1:
            logger.warning(f"Extracted P1 is empty from single part '{subjects_text}'. Using fallback.")
            p1 = "UnknownPerson1_from_empty_single"
        logger.warning(f"Only one person found in subjects '{subjects_text}' after split. Extracted P1: '{p1}', using fallback for P2.")
        return p1, "UnknownTarget_From_Split_Single" 
    else: 
        logger.warning(f"Unexpected parsing of subjects '{subjects_text}' after split (0 parts). Using fallbacks.")
        return "UnknownPerson1", "UnknownPerson2"

def main(person1_arg, person2_arg, no_cleanup_arg=False):
    logger.info(f"--- Starting Visual Chain Generation (Verification: DeepFace Multi-Model) ---")
    logger.info(f"Person A: {person1_arg}")
    logger.info(f"Person B: {person2_arg}")

    if not no_cleanup_arg:
        logger.info("--- Cleaning up previous output directories ---")
        dirs_to_clean = [RAW_IMAGES_DIR, VERIFIED_IMAGES_DIR, TEMP_FILES_DIR]
        for d in dirs_to_clean: 
            clean_output_directory(d, logger)
    else: 
        logger.info("--- Skipping output directory cleanup as per --no-cleanup flag ---")

    ensure_dir(RAW_IMAGES_DIR)
    ensure_dir(VERIFIED_IMAGES_DIR)
    ensure_dir(TEMP_FILES_DIR)

    system_prompt_content = load_prompt_template(SYSTEM_PROMPT_FILE_PATH)
    initial_user_input_template = load_prompt_template(INITIAL_USER_INPUT_TEMPLATE_PATH)
    initial_chain_retry_template = load_prompt_template(INITIAL_CHAIN_RETRY_USER_INPUT_TEMPLATE_PATH)
    retry_user_input_template = load_prompt_template(RETRY_USER_INPUT_TEMPLATE_PATH)
    sub_chain_exclusion_template = load_prompt_template(SUB_CHAIN_EXCLUSION_TEMPLATE_PATH, is_critical=False)


    if not all([system_prompt_content, initial_user_input_template, initial_chain_retry_template, 
                retry_user_input_template]): 
        logger.fatal("One or more core prompt templates could not be loaded. Exiting.")
        return
    if not sub_chain_exclusion_template: 
        logger.warning(f"Sub-chain exclusion template not loaded from {SUB_CHAIN_EXCLUSION_TEMPLATE_PATH}. Sub-chain retries might be less effective.")


    logger.info("--- Step 1: Requesting Initial Full Connection Chain from Gemini ---")
    current_chain_xml_str = None
    current_chain_root = None
    
    for attempt in range(MAX_INITIAL_CHAIN_RETRY_ATTEMPTS + 1):
        logger.info(f"Initial full chain generation attempt {attempt + 1}/{MAX_INITIAL_CHAIN_RETRY_ATTEMPTS + 1}")
        user_input_xml_for_gemini_payload = ""
        if attempt == 0:
            current_chain_xml_str = initial_gemini_query.get_initial_chain(
                person1_arg, person2_arg, system_prompt_content, 
                initial_user_input_template, GOOGLE_API_KEY, logger, exclusion_instruction=""
            )
        else:
            logger.info(f"Retrying initial full chain generation. Waiting for {INITIAL_CHAIN_RETRY_DELAY_SECONDS}s...")
            time.sleep(INITIAL_CHAIN_RETRY_DELAY_SECONDS)
            previous_response_snippet = current_chain_xml_str[:200] if current_chain_xml_str else "N/A"
            try:
                user_input_xml_for_gemini_payload = initial_chain_retry_template.format(
                    person1_name=person1_arg, person2_name=person2_arg,
                    previous_response_snippet=previous_response_snippet
                )
            except KeyError as e_fmt: 
                logger.error(f"Formatting initial_chain_retry_template failed: {e_fmt}. Skipping retry attempt.")
                continue 
            
            current_chain_xml_str = initial_gemini_query.get_initial_chain_from_gemini_direct(
                 system_prompt_content, user_input_xml_for_gemini_payload, GOOGLE_API_KEY, logger
            )
        
        attempt_file_path = os.path.join(TEMP_FILES_DIR, f"initial_full_chain_attempt_{attempt+1}.xml")
        try:
            with open(attempt_file_path, "w", encoding="utf-8") as f: 
                f.write(current_chain_xml_str if isinstance(current_chain_xml_str, str) else str(current_chain_xml_str))
            logger.info(f"Initial full chain attempt {attempt+1} output saved to {attempt_file_path}")
        except Exception as e_write: 
            logger.error(f"Failed to write initial full chain attempt {attempt+1} to file: {e_write}")

        if not isinstance(current_chain_xml_str, str) or not current_chain_xml_str.strip():
            logger.warning(f"Attempt {attempt+1}: Received empty or non-string response for full chain.")
            if attempt == MAX_INITIAL_CHAIN_RETRY_ATTEMPTS:
                logger.error("Max retries for initial full chain. Last response was empty/invalid.")
                return
            continue

        try:
            parsed_root = ET.fromstring(current_chain_xml_str)
            if parsed_root.tag == "connection_chain":
                current_chain_root = parsed_root
                logger.info(f"Successfully parsed <connection_chain> on attempt {attempt+1}.")
                break 
            elif parsed_root.tag == "research_failure":
                logger.warning(f"Attempt {attempt+1}: Gemini reported <research_failure> for full chain. XML: {current_chain_xml_str}")
                if attempt == MAX_INITIAL_CHAIN_RETRY_ATTEMPTS: 
                    logger.error("Max retries for full chain, last was <research_failure>.")
                    try: ET.ElementTree(parsed_root).write(os.path.join(TEMP_FILES_DIR, "final_full_chain_research_failure.xml"))
                    except Exception as e_xmlw: logger.error(f"Could not write research_failure XML: {e_xmlw}")
                    return 
            elif parsed_root.tag == "error": 
                error_type = parsed_root.findtext("type", "UnknownErrorType")
                logger.warning(f"Attempt {attempt+1}: Received error XML (type: {error_type}) for full chain from get_initial_chain_...: {current_chain_xml_str[:300]}")
            else: 
                logger.warning(f"Attempt {attempt+1}: Unexpected root tag '{parsed_root.tag}' for full chain. XML: {current_chain_xml_str[:300]}")
        except ET.ParseError as e: 
            logger.warning(f"Attempt {attempt+1}: Failed to parse XML for full chain: {e}. Raw XML snippet: {current_chain_xml_str[:500]}")
        
        if attempt == MAX_INITIAL_CHAIN_RETRY_ATTEMPTS: 
            logger.error("Max retries for initial full chain. Could not get valid <connection_chain>.")
            return

    if not current_chain_root: 
        logger.fatal("Failed to obtain initial full connection chain. Exiting.")
        return

    master_links_to_process = list(current_chain_root.findall("link")) 
    verified_chain_links_data = [] 
    last_verified_person = person1_arg
    chain_broken = False
    link_idx = -1 
    failed_links_signatures = set() # Stores (sanitized_subjects_text, google_query)

    while (link_idx + 1) < len(master_links_to_process):
        link_idx += 1
        current_processing_link_node = master_links_to_process[link_idx]

        logger.info(f"--- Processing Link {link_idx + 1}/{len(master_links_to_process)} (Overall) ---")
        logger.debug(f"Current link to process XML: {ET.tostring(current_processing_link_node, encoding='unicode')[:300]}")
        
        p1_current_segment, p2_current_segment = "", "" 
        
        subjects_text = current_processing_link_node.findtext("subjects")
        google_query = current_processing_link_node.findtext("google")
        
        if not subjects_text or not google_query:
            logger.error(f"  Link node {link_idx+1} is missing 'subjects' or 'google' tag. Segment failed critically.")
            chain_broken = True; break 

        p1_current_segment, p2_current_segment = extract_persons_from_subjects(subjects_text)
        logger.info(f"  Attempting segment: '{p1_current_segment}' → '{p2_current_segment}'")

        current_link_signature = (image_downloader.sanitize_filename(subjects_text), google_query)
        
        is_first_link_overall = not verified_chain_links_data and p1_current_segment == person1_arg
        is_subsequent_link_correct = verified_chain_links_data and p1_current_segment == last_verified_person
        
        if not (is_first_link_overall or is_subsequent_link_correct):
            logger.error(f"  Chain integrity error! Expected segment to start with '{last_verified_person}' (or '{person1_arg}' if first link), but got '{p1_current_segment}'. Chain broken.")
            logger.debug(f"    Context: last_verified_person='{last_verified_person}', p1_current_segment='{p1_current_segment}', person1_arg='{person1_arg}', verified_chain_links_data length: {len(verified_chain_links_data)}")
            chain_broken = True; break
        
        verifier_status = None
        verifier_data = None
        download_folder_path = None

        if current_link_signature in failed_links_signatures:
            logger.info(f"  Link '{subjects_text}' (Query: '{google_query}') was previously tried and failed verification. Skipping download and verification.")
            verifier_status = "ALREADY_FAILED_IN_ORCHESTRATOR" # Custom status for orchestrator
        else:
            download_folder_path = image_downloader.fetch_images_for_link(
                subjects_text, google_query, RAW_IMAGES_DIR, IMAGES_TO_DOWNLOAD_PER_LINK, logger
            )
            
            verifier_status, verifier_data = image_verifier.verify_and_potentially_reprompt_link(
                person1_in_link=p1_current_segment, person2_in_link=p2_current_segment,
                images_folder_path=download_folder_path if download_folder_path else "", 
                original_link_xml_node=current_processing_link_node,
                system_prompt_content=system_prompt_content,
                retry_user_input_template_str=retry_user_input_template,
                logger_obj=logger, max_images_to_check_vision=MAX_IMAGES_TO_CHECK_PER_LINK_VERIFIER
            )

        # --- Handle Verification Outcome ---
        should_regenerate_subchain = False
        original_link_failed_definitively_this_iteration = False

        if verifier_status == "VERIFIED_OK":
            verified_image_path_raw, verified_link_xml_str = verifier_data
            logger.info(f"  SUCCESS: Segment '{p1_current_segment}' → '{p2_current_segment}' verified with image: {verified_image_path_raw}")
            
            destination_path_copied_image = None 
            try:
                sanitized_p1 = image_downloader.sanitize_filename(p1_current_segment)
                sanitized_p2 = image_downloader.sanitize_filename(p2_current_segment)
                
                _, file_extension = os.path.splitext(verified_image_path_raw)
                if not file_extension: 
                    file_extension = ".jpg" 
                    logger.warning(f"Verified image {verified_image_path_raw} had no extension, defaulting to .jpg for copy.")

                verified_image_index = len(verified_chain_links_data) + 1
                new_filename = f"{verified_image_index:02d}_{sanitized_p1}_to_{sanitized_p2}{file_extension}"
                destination_path_copied_image = os.path.join(VERIFIED_IMAGES_DIR, new_filename)
                
                ensure_dir(VERIFIED_IMAGES_DIR) 
                shutil.copy(verified_image_path_raw, destination_path_copied_image)
                logger.info(f"  Copied verified image to: {destination_path_copied_image}")
                
            except Exception as e_copy:
                logger.error(f"  Error copying verified image {verified_image_path_raw} to {VERIFIED_IMAGES_DIR}: {e_copy}")
            
            verified_chain_links_data.append((verified_link_xml_str, verified_image_path_raw, destination_path_copied_image))
            last_verified_person = p2_current_segment 
            
            if last_verified_person == person2_arg: 
                logger.info(f"  Target '{person2_arg}' reached and verified!")
                break 
        
        # Handle cases where verification did not result in VERIFIED_OK
        else: 
            if verifier_status == "FAILED_VERIFICATION_NO_ALTERNATIVE":
                logger.warning(f"  Segment '{p1_current_segment}' → '{p2_current_segment}' failed verification, and verifier found no alternative. Query: '{google_query}'.")
                original_link_failed_definitively_this_iteration = True
                should_regenerate_subchain = True
            
            elif verifier_status == "NEEDS_REPROMPT_NEW_LINK_PROVIDED":
                new_link_suggestion_xml_str = verifier_data 
                logger.info(f"  Orchestrator received new link suggestion from verifier. Attempting to parse and integrate.")
                try:
                    new_link_node = ET.fromstring(new_link_suggestion_xml_str)
                    if new_link_node.tag == "link":
                        new_subjects_text = new_link_node.findtext("subjects")
                        new_google_query = new_link_node.findtext("google")
                        if not new_subjects_text or not new_google_query:
                            logger.error("    New link suggestion from verifier is missing subjects or google query. Discarding.")
                            original_link_failed_definitively_this_iteration = True
                            should_regenerate_subchain = True
                        else:
                            new_link_signature = (image_downloader.sanitize_filename(new_subjects_text), new_google_query)
                            if new_link_signature in failed_links_signatures:
                                logger.warning(f"    Verifier suggested new link '{new_subjects_text}', but this link/query combination was also previously tried and failed. Original link '{subjects_text}' is now considered definitively failed.")
                                original_link_failed_definitively_this_iteration = True
                                should_regenerate_subchain = True
                            else:
                                logger.info(f"    Successfully parsed new <link> node. Replacing current link and re-processing.")
                                master_links_to_process[link_idx] = new_link_node 
                                link_idx -=1 
                                continue # Restart loop for the new link suggestion
                    else:
                        logger.error(f"    New link suggestion was not a single <link> tag, but <{new_link_node.tag}>. Original link '{subjects_text}' considered failed.")
                        original_link_failed_definitively_this_iteration = True
                        should_regenerate_subchain = True
                except ET.ParseError as e_parse:
                    logger.error(f"    Failed to parse new link XML from verifier: {e_parse}. Raw: {new_link_suggestion_xml_str[:300]}. Original link '{subjects_text}' considered failed.")
                    original_link_failed_definitively_this_iteration = True
                    should_regenerate_subchain = True
                except Exception as e_unexpected:
                    logger.error(f"    Unexpected error processing new link XML from verifier: {e_unexpected}. Original link '{subjects_text}' considered failed.")
                    original_link_failed_definitively_this_iteration = True
                    should_regenerate_subchain = True

            elif verifier_status == "ALREADY_FAILED_IN_ORCHESTRATOR":
                logger.info(f"  Link '{subjects_text}' (Query: '{google_query}') was identified as ALREADY FAILED by orchestrator. Proceeding to sub-chain regeneration.")
                # No need to add to failed_links_signatures again, it's already there.
                should_regenerate_subchain = True
            
            else: # Other unknown failure from verifier
                logger.warning(f"  Segment '{p1_current_segment}' → '{p2_current_segment}' failed with unhandled verifier status: {verifier_status}. Query: '{google_query}'.")
                original_link_failed_definitively_this_iteration = True
                should_regenerate_subchain = True

            if original_link_failed_definitively_this_iteration:
                 logger.info(f"  Adding signature for failed link '{subjects_text}' (Query: '{google_query}') to failed_links_signatures.")
                 failed_links_signatures.add(current_link_signature)


        if should_regenerate_subchain:
            logger.info(f"  Attempting orchestrator-level sub-chain re-generation from '{last_verified_person}' to '{person2_arg}'.")
            logger.debug(f"    Failed segment was '{p1_current_segment}' → '{p2_current_segment}'.")
            
            chain_regenerated_from_here = False
            current_verified_path_people = [person1_arg] + [extract_persons_from_subjects(ET.fromstring(link_data[0]).findtext("subjects"))[1] for link_data in verified_chain_links_data]
            current_verified_path_people = list(dict.fromkeys(current_verified_path_people)) 
            
            people_to_avoid_as_intermediaries = [p for p in current_verified_path_people if p != last_verified_person and p != person2_arg]
            immediate_next_step_exclusion = p2_current_segment 

            for sub_chain_attempt in range(MAX_ORCHESTRATOR_SUBCHAIN_ATTEMPTS + 1):
                logger.info(f"    Orchestrator sub-chain attempt {sub_chain_attempt + 1}/{MAX_ORCHESTRATOR_SUBCHAIN_ATTEMPTS + 1} from '{last_verified_person}' to '{person2_arg}'")
                if sub_chain_attempt > 0: time.sleep(INITIAL_CHAIN_RETRY_DELAY_SECONDS)

                exclusion_instr = ""
                if sub_chain_exclusion_template:
                    avoid_intermediaries_instr_part = ""
                    if people_to_avoid_as_intermediaries:
                        exclusion_list_str = ", ".join(f"'{p}'" for p in people_to_avoid_as_intermediaries)
                        avoid_intermediaries_instr_part = (f"2. If possible, also try to AVOID using any of these people as *new intermediaries* in this sub-chain, "
                                                           f"as they are already part of the path leading to '{last_verified_person}': [{exclusion_list_str}]. "
                                                           f"Focus on new, previously unused intermediaries.\n")
                    
                    failed_direct_links_instr_part = ""
                    failed_subject_query_strings_for_prompt = set()
                    if failed_links_signatures:
                        for sig_subj, sig_query in failed_links_signatures:
                            parts = sig_subj.split('_')
                            human_readable_failed_link = " → ".join(parts) if len(parts) > 1 else (parts[0] if parts else sig_subj)
                            failed_subject_query_strings_for_prompt.add(f"'{human_readable_failed_link}' (related to query: '{sig_query[:75]}...')")
                    
                    if failed_subject_query_strings_for_prompt:
                        failed_direct_links_instr_part = (f"3. CRITICAL: AVOID suggesting any of the following specific connections (SUBJECTS and associated general query context) that have already been attempted and FAILED verification in previous steps: "
                                                          f"[{', '.join(sorted(list(failed_subject_query_strings_for_prompt)))}]. Focus on entirely new connection ideas.\n")

                    try:
                        exclusion_instr = sub_chain_exclusion_template.format(
                            last_verified_person_name=last_verified_person,
                            target_person_name=person2_arg,
                            immediate_next_step_to_avoid=immediate_next_step_exclusion,
                            avoid_intermediaries_instruction=avoid_intermediaries_instr_part,
                            previously_failed_direct_links_instruction=failed_direct_links_instr_part
                        )
                    except KeyError as e_fmt:
                        logger.error(f"Failed to format sub-chain exclusion template: {e_fmt}. Using minimal exclusion.")
                        exclusion_instr = f"IMPORTANT: Do not suggest '{immediate_next_step_exclusion}' as next from '{last_verified_person}'."
                        if failed_direct_links_instr_part: exclusion_instr += "\n" + failed_direct_links_instr_part.replace("3. ", "") # Add if available

                else: 
                     exclusion_instr = (
                        f"IMPORTANT: You are building a new sub-chain from '{last_verified_person}' to '{person2_arg}'.\n"
                        f"1. Crucially, DO NOT suggest '{immediate_next_step_exclusion}' as the immediate next connection from '{last_verified_person}'. That path just failed.\n")
                     if people_to_avoid_as_intermediaries:
                        exclusion_list_str = ", ".join(f"'{p}'" for p in people_to_avoid_as_intermediaries)
                        exclusion_instr += (f"2. If possible, also try to AVOID using any of these people as *new intermediaries* in this sub-chain, as they are already part of the path leading to '{last_verified_person}': [{exclusion_list_str}]. Focus on new, previously unused intermediaries.\n")
                     
                     failed_subject_query_strings_for_prompt_fallback = set() # Re-calculate for fallback
                     if failed_links_signatures:
                        for sig_subj, sig_query in failed_links_signatures:
                            parts = sig_subj.split('_')
                            human_readable_failed_link = " → ".join(parts) if len(parts) > 1 else (parts[0] if parts else sig_subj)
                            failed_subject_query_strings_for_prompt_fallback.add(f"'{human_readable_failed_link}' (query: '{sig_query[:75]}...')")
                     if failed_subject_query_strings_for_prompt_fallback:
                         exclusion_instr += (f"3. CRITICAL: AVOID suggesting any of the following specific connections (SUBJECTS and associated general query context) that have already been attempted and FAILED verification in previous steps: "
                                          f"[{', '.join(sorted(list(failed_subject_query_strings_for_prompt_fallback)))}]. Focus on entirely new connection ideas.\n")
                     exclusion_instr += f"Goal: Find a NEW, fresh path from '{last_verified_person}' to '{person2_arg}'."
                
                logger.debug(f"    Sub-chain exclusion instruction: {exclusion_instr}")

                new_sub_chain_xml_str = initial_gemini_query.get_initial_chain(
                    last_verified_person, person2_arg, system_prompt_content,
                    initial_user_input_template, GOOGLE_API_KEY, logger, exclusion_instruction=exclusion_instr
                )
                
                sub_chain_attempt_file = os.path.join(TEMP_FILES_DIR, f"sub_chain_attempt_{image_downloader.sanitize_filename(last_verified_person)}_to_{image_downloader.sanitize_filename(person2_arg)}_{sub_chain_attempt+1}.xml")
                try:
                    with open(sub_chain_attempt_file, "w", encoding="utf-8") as f: f.write(new_sub_chain_xml_str if isinstance(new_sub_chain_xml_str, str) else str(new_sub_chain_xml_str))
                    logger.info(f"    Sub-chain attempt output saved to {sub_chain_attempt_file}")
                except Exception as e_sfw: logger.error(f"Error writing sub-chain file: {e_sfw}")

                if not isinstance(new_sub_chain_xml_str, str) or not new_sub_chain_xml_str.strip():
                    logger.warning(f"    Sub-chain attempt {sub_chain_attempt+1} returned empty or invalid response.")
                    if sub_chain_attempt == MAX_ORCHESTRATOR_SUBCHAIN_ATTEMPTS:
                        logger.error(f"    Max sub-chain regeneration attempts from '{last_verified_person}' reached. Last response was empty.")
                    continue

                try:
                    new_sub_chain_root = ET.fromstring(new_sub_chain_xml_str)
                    if new_sub_chain_root.tag == "connection_chain":
                        new_links = new_sub_chain_root.findall("link")
                        if new_links:
                            logger.info(f"    Successfully got new sub-chain from '{last_verified_person}' with {len(new_links)} link(s). Replacing rest of old chain.")
                            master_links_to_process = master_links_to_process[:link_idx] + new_links 
                            link_idx -=1 
                            chain_regenerated_from_here = True; break 
                        else: logger.warning(f"    Sub-chain attempt {sub_chain_attempt+1} from '{last_verified_person}' returned <connection_chain> but with no links.")
                    elif new_sub_chain_root.tag == "research_failure": logger.warning(f"    Sub-chain attempt {sub_chain_attempt+1} from '{last_verified_person}' resulted in <research_failure>.")
                    elif new_sub_chain_root.tag == "error": logger.warning(f"    Sub-chain attempt {sub_chain_attempt+1} from '{last_verified_person}' returned an error XML (type: {new_sub_chain_root.findtext('type','N/A')}).")
                    else: logger.warning(f"    Sub-chain attempt {sub_chain_attempt+1} from '{last_verified_person}' returned unexpected XML: {new_sub_chain_root.tag}")
                except ET.ParseError as e_parse: logger.warning(f"    Failed to parse sub-chain XML from '{last_verified_person}': {e_parse}. Raw: {new_sub_chain_xml_str[:300]}")

                if sub_chain_attempt == MAX_ORCHESTRATOR_SUBCHAIN_ATTEMPTS: 
                    logger.error(f"    Max sub-chain regeneration attempts from '{last_verified_person}' reached. This path is broken.")
                    break 
            
            if not chain_regenerated_from_here: 
                chain_broken = True
                break 

    if not chain_broken and last_verified_person == person2_arg and verified_chain_links_data:
        logger.info("--- Assembling Final Verified Chain ---")
        final_verified_chain_root = ET.Element("connection_chain")
        chain_summary_node = ET.SubElement(final_verified_chain_root, "chain_summary")
        ET.SubElement(chain_summary_node, "total_links").text = str(len(verified_chain_links_data))
        
        subjects_path_list = [person1_arg]
        temp_last_person = person1_arg
        
        for i, (link_xml_str, raw_image_path, copied_image_path) in enumerate(verified_chain_links_data):
            try:
                link_node = ET.fromstring(link_xml_str)
                current_link_subjects_text = link_node.findtext("subjects", "")
                link_p1, link_p2 = extract_persons_from_subjects(current_link_subjects_text)
                
                if link_p1 == temp_last_person : 
                    if link_p2 not in subjects_path_list: subjects_path_list.append(link_p2)
                    temp_last_person = link_p2
                else: 
                    logger.warning(f"Chain integrity issue during final summary: Link {i+1} P1 '{link_p1}' does not match expected '{temp_last_person}'. Path: {subjects_path_list}")
                    if not subjects_path_list or link_p1 != subjects_path_list[-1]:
                         subjects_path_list.append(link_p1) 
                    if link_p2 not in subjects_path_list: subjects_path_list.append(link_p2)
                    temp_last_person = link_p2
                
                if copied_image_path and os.path.exists(copied_image_path):
                    ET.SubElement(link_node, "verified_image_filename").text = os.path.basename(copied_image_path)
                elif raw_image_path: 
                    ET.SubElement(link_node, "verified_image_filename").text = f"NOT_COPIED_RAW_({os.path.basename(raw_image_path)})"
                else:
                    ET.SubElement(link_node, "verified_image_filename").text = "IMAGE_PATH_UNKNOWN"
                
                final_verified_chain_root.append(link_node)

            except ET.ParseError as e_parse_final: 
                logger.error(f"Error parsing stored link_xml_str during final assembly: {e_parse_final}. Link {i+1} might be corrupt."); 
                error_link_node = ET.Element("link", id=str(i+1))
                ET.SubElement(error_link_node, "error_parsing_link").text = f"Original XML unparsable: {link_xml_str[:100]}"
                final_verified_chain_root.append(error_link_node)
                continue 
            except Exception as e_final_assembly:
                 logger.error(f"Unexpected error during final assembly of link {i+1}: {e_final_assembly}")
                 error_link_node = ET.Element("link", id=str(i+1))
                 ET.SubElement(error_link_node, "error_processing_link").text = f"Unexpected error: {str(e_final_assembly)}"
                 final_verified_chain_root.append(error_link_node)

        if subjects_path_list and subjects_path_list[-1] != person2_arg:
             logger.warning(f"Final assembled path '{' → '.join(subjects_path_list)}' does not end with target '{person2_arg}'. This may occur if the chain was incomplete.")
        elif not subjects_path_list and person1_arg: 
            subjects_path_list = [person1_arg] if verified_chain_links_data else [] 
        elif not subjects_path_list: 
            logger.warning("Subjects path list is empty after processing verified links.")

        ET.SubElement(chain_summary_node, "subjects_connected").text = " → ".join(subjects_path_list)
        ET.SubElement(chain_summary_node, "chain_type").text = "Direct" if len(verified_chain_links_data) == 1 and len(subjects_path_list) == 2 else "Extended"
        ET.SubElement(chain_summary_node, "intermediary_count").text = str(max(0, len(subjects_path_list) - 2))
        ET.SubElement(chain_summary_node, "research_confidence").text = "Medium" 

        final_xml_path = os.path.join(TEMP_FILES_DIR, "verified_connection_chain.xml")
        try:
            tree = ET.ElementTree(final_verified_chain_root)
            # ET.indent(tree, space="  ") # Python 3.9+
            tree.write(final_xml_path, encoding="utf-8", xml_declaration=True)
            logger.info(f"SUCCESS: Verified chain XML written to: {final_xml_path}")
            logger.info(f"Verified images (if any were successfully copied) are in: {VERIFIED_IMAGES_DIR}")
        except Exception as e_write_final_xml: 
            logger.error(f"Failed to write final verified XML: {e_write_final_xml}")
        
        final_image_paths_for_video = [data[2] for data in verified_chain_links_data if data[2] and os.path.exists(data[2])]
        if final_image_paths_for_video:
            logger.info("--- Next Steps: ---")
            logger.info("1. Review images in VERIFIED_IMAGES_DIR and the final XML.")
            logger.info(f"   Paths to successfully copied images: {final_image_paths_for_video}")
            logger.info("2. If images need annotation, do so. Then feed to your video generator.")
        else: 
            logger.info("No images were successfully verified and/or copied for the video, or an error occurred during copy.")

    elif chain_broken: 
        logger.info("--- Process Stopped: Chain was broken during verification or re-generation. Check logs for details. ---")
    elif not verified_chain_links_data:
        logger.info("--- Process Finished: No verifiable links were established. ---")
    elif last_verified_person != person2_arg:
        logger.info(f"--- Process Finished: Chain was formed but did not reach the final target '{person2_arg}'. Last verified person: '{last_verified_person}'. ---")
    else: 
        logger.info("--- Process Finished: Outcome undetermined or no verifiable chain produced. ---")
        
    logger.info("--- Visual Chain Generation Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a visual chain connecting two people.")
    parser.add_argument("--person1", required=True, help="Name of the first person.")
    parser.add_argument("--person2", required=True, help="Name of the second person (the target).")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not clean output directories before running.")
    args = parser.parse_args()
    main(args.person1, args.person2, args.no_cleanup)