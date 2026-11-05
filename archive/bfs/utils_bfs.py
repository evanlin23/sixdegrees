# bfs/utils_bfs.py
import os
import shutil
import xml.etree.ElementTree as ET
import logging
import re

# --- Directory and File System ---
def ensure_dir_bfs(directory_path, logger):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            raise
    else:
        logger.debug(f"Directory already exists: {directory_path}")


def clean_output_sub_dir_bfs(dir_path, logger):
    if not os.path.exists(dir_path):
        logger.info(f"Directory to clean does not exist, skipping: {dir_path}")
        return
    if not os.path.isdir(dir_path):
        logger.warning(f"Path to clean is not a directory, skipping: {dir_path}")
        return
    logger.info(f"Cleaning contents of directory: {dir_path}")
    for item_name in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            logger.error(f"Failed to delete {item_path}. Reason: {e}")

# --- Prompt Loading ---
def load_prompt_template_bfs(filepath, logger, is_critical=True):
    logger.debug(f"Attempting to load BFS prompt template from: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"BFS Prompt template loaded successfully from {filepath}")
        return content
    except FileNotFoundError:
        log_level = logger.error if is_critical else logger.warning
        log_level(f"{'FATAL ERROR' if is_critical else 'Warning'}: BFS Prompt template file '{filepath}' not found.")
        if is_critical: raise FileNotFoundError(f"Critical BFS prompt file missing: {filepath}")
        return None
    except Exception as e:
        log_level = logger.error if is_critical else logger.warning
        log_level(f"Error loading BFS prompt template file '{filepath}': {e}", exc_info=True)
        if is_critical: raise
        return None

# --- XML Processing ---
def extract_connection_details_from_suggestion(suggestion_node, logger):
    details = {}
    connection_details_node = suggestion_node.find("connection_details")
    if connection_details_node is None:
        logger.warning("No <connection_details> found in suggestion.")
        # Attempt to find fields directly under suggestion_node as a fallback for simpler structures
        fields_to_try = ["evidence", "source", "google", "source_type", "context", "date_range", "strength", "verification_notes"]
        found_any_direct = False
        for field in fields_to_try:
            value = suggestion_node.findtext(field, "").strip()
            if value:
                details[field] = value
                found_any_direct = True
        if found_any_direct:
            logger.info("Found some connection details directly under <suggestion> node.")
        return details

    # Standard extraction from <connection_details>
    fields = ["evidence", "source", "google_query", "source_type", "context", "date_range", "strength", "verification_notes"] # "google" -> "google_query"
    # Also include fields that might come from image_verifier
    verifier_fields = [
        "verified_image_filename",
        f"{suggestion_node.findtext('intermediary_person_name', 'personX')}_verified_by_model", # Placeholder if name not found
        f"{suggestion_node.findtext('intermediary_person_name', 'personX')}_distance",
        f"{suggestion_node.findtext('intermediary_person_name', 'personX')}_threshold_used",
        "verification_detector_backend"
    ]
    all_fields_to_check = fields + verifier_fields # Check all known possible fields

    for field_key in all_fields_to_check:
        # Try to find the field, if it's a dynamic field name, it would have been constructed above
        # For static fields, just use the key. This part needs to be careful if dynamic field names were intended.
        # The current verifier_fields list above uses static strings from example persons.
        # This is complex. Let's simplify: connection_details_dict from verifier is merged in main_bfs.
        # So, here, we only extract what Gemini provides.
        pass # This logic becomes complex due to dynamic keys. The merging in main_bfs handles verifier details.

    # Simplified for what Gemini provides
    gemini_provided_fields = ["evidence", "source", "google_query_suggestion", "source_type", "context", "date_range", "strength", "verification_notes"]
    for field in gemini_provided_fields:
        details[field] = connection_details_node.findtext(field, "").strip()

    # If google_query (specific for link) is set by main_bfs, it will overwrite google_query_suggestion here.
    # This is fine. `extract_connection_details_from_suggestion` gets Gemini's view.
    return details


def create_final_link_xml_node(p1_name, p2_name, connection_details_dict, verified_image_filename_relative, link_id_counter):
    link_node = ET.Element("link", id=str(link_id_counter))
    ET.SubElement(link_node, "subjects").text = f"{p1_name} → {p2_name}"

    # Ensure specific important keys are present, even if empty, for consistent XML structure
    # These are keys typically found in connection_details_dict
    standard_keys = [
        "google_query", "evidence", "source", "source_type", "context",
        "date_range", "strength", "verification_notes",
        f"{sanitize_filename_bfs(p1_name)}_verified_by_model", # Verifier dynamic keys
        f"{sanitize_filename_bfs(p1_name)}_distance",
        f"{sanitize_filename_bfs(p1_name)}_threshold_used",
        f"{sanitize_filename_bfs(p2_name)}_verified_by_model",
        f"{sanitize_filename_bfs(p2_name)}_distance",
        f"{sanitize_filename_bfs(p2_name)}_threshold_used",
        "verification_detector_backend"
    ]

    for key in standard_keys:
        value = connection_details_dict.get(key, "") # Get value or empty string if not present
        # Handle cases where value might be None or not a string
        if not isinstance(value, str):
            value = str(value) if value is not None else ""
        ET.SubElement(link_node, key).text = value

    # Add any other keys from connection_details_dict not in standard_keys (e.g. Gemini's own fields)
    for key, value in connection_details_dict.items():
        if key not in standard_keys:
            # Handle cases where value might be None or not a string
            if not isinstance(value, str):
                value = str(value) if value is not None else ""
            ET.SubElement(link_node, key).text = value

    if verified_image_filename_relative:
        ET.SubElement(link_node, "verified_image_filename").text = verified_image_filename_relative
    else:
        ET.SubElement(link_node, "verified_image_filename").text = "IMAGE_NOT_FOUND_OR_COPIED"
    return link_node

def save_chain_to_xml(person1_start, final_path_tuples, output_dir_root, logger, no_path_person2=None, error_message=None):
    final_verified_chain_root = ET.Element("connection_chain")

    # Add metadata about the query
    query_info_node = ET.SubElement(final_verified_chain_root, "query_information")
    ET.SubElement(query_info_node, "start_person").text = person1_start
    if no_path_person2:
        ET.SubElement(query_info_node, "target_person").text = no_path_person2
    elif final_path_tuples:
        ET.SubElement(query_info_node, "target_person").text = final_path_tuples[-1][1] # p_to of the last link
    else: # Case: person1_start == person2_target, no links
         ET.SubElement(query_info_node, "target_person").text = person1_start


    chain_summary_node = ET.SubElement(final_verified_chain_root, "chain_summary")

    if error_message:
        ET.SubElement(chain_summary_node, "status").text = "Error"
        ET.SubElement(chain_summary_node, "error_details").text = error_message
    elif not final_path_tuples:
        if person1_start == (no_path_person2 or person1_start): # Handling A to A or if no_path_person2 is None
             ET.SubElement(chain_summary_node, "status").text = "Start and Target are the same"
        else:
            ET.SubElement(chain_summary_node, "status").text = "No verifiable path found"
    else:
        ET.SubElement(chain_summary_node, "status").text = "Path found"


    subjects_path_list = [person1_start]
    p_end_final = person1_start

    if final_path_tuples:
        for i, (p_from, p_to, conn_details, _, rel_img_path) in enumerate(final_path_tuples):
            if not subjects_path_list or p_from == subjects_path_list[-1]:
                if p_to not in subjects_path_list:
                     subjects_path_list.append(p_to)
            else:
                logger.error(f"Chain integrity error during XML save: Link {i+1} p_from '{p_from}' does not match last in path '{subjects_path_list[-1]}'. Appending.")
                if p_from not in subjects_path_list: subjects_path_list.append(p_from)
                if p_to not in subjects_path_list: subjects_path_list.append(p_to)
            p_end_final = p_to
            link_node = create_final_link_xml_node(p_from, p_to, conn_details, rel_img_path, i + 1)
            final_verified_chain_root.append(link_node)

    ET.SubElement(chain_summary_node, "total_links").text = str(len(final_path_tuples))
    ET.SubElement(chain_summary_node, "subjects_connected_path").text = " → ".join(subjects_path_list)
    chain_type_text = "Direct" if len(final_path_tuples) == 1 else "Extended"
    if len(final_path_tuples) == 0 : chain_type_text = "N/A"
    ET.SubElement(chain_summary_node, "chain_type").text = chain_type_text
    ET.SubElement(chain_summary_node, "intermediary_count").text = str(max(0, len(subjects_path_list) - 2))
    ET.SubElement(chain_summary_node, "research_confidence").text = "Medium (BiBFS)"

    target_person_for_filename = p_end_final
    if no_path_person2: # If search was for a specific person but no path found
        target_person_for_filename = no_path_person2

    final_xml_filename = f"verified_chain_bibfs_{sanitize_filename_bfs(person1_start)}_to_{sanitize_filename_bfs(target_person_for_filename)}.xml"
    final_xml_path = os.path.join(output_dir_root, final_xml_filename)

    try:
        tree = ET.ElementTree(final_verified_chain_root)
        # ET.indent(tree, space="  ") # Python 3.9+ for pretty print
        # Manual indent for compatibility
        _pretty_print_xml(final_verified_chain_root)
        tree.write(final_xml_path, encoding="utf-8", xml_declaration=True)
        logger.info(f"SUCCESS: BiBFS Verified chain XML written to: {final_xml_path}")
        return final_xml_path
    except Exception as e_write_final_xml:
        logger.error(f"Failed to write final BiBFS verified XML to {final_xml_path}: {e_write_final_xml}")
        return None

def _pretty_print_xml(element, indent_level=0, indent_str="  "):
    """Helper to manually pretty print XML for wider Python compatibility."""
    i = "\n" + indent_level * indent_str
    if len(element):
        if not element.text or not element.text.strip():
            element.text = i + indent_str
        if not element.tail or not element.tail.strip():
            element.tail = i
        for subelement_idx, subelement in enumerate(element): # Iterate with index
            _pretty_print_xml(subelement, indent_level + 1, indent_str)
            if subelement_idx == len(element) -1: # Last child
                if not subelement.tail or not subelement.tail.strip():
                     subelement.tail = i
    elif not element.tail or not element.tail.strip():
        element.tail = i


def sanitize_filename_bfs(name):
    if not name: return "unknown_subject"
    # Normalize spaces and common path-breaking characters
    name = str(name).strip() # Ensure it's a string
    name = re.sub(r'\s+', '_', name) # Replace whitespace sequences with single underscore
    name = re.sub(r'[<>:"/\\|?*&^%$#@!`~=+{}\[\];,\']', '', name) # Remove problematic characters
    name = re.sub(r'_+', '_', name) # Collapse multiple underscores
    name = re.sub(r'^_|_$', '', name) # Remove leading/trailing underscores
    return name if name else "unknown_subject"