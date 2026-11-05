# bfs/main_bfs.py
import os
import argparse
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import shutil
import logging
import time
import collections # For deque
import networkx as nx

# Local BFS imports
import gemini_bfs_query
import image_downloader_bfs
import image_verifier_bfs
import utils_bfs

# --- Configuration ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

load_dotenv(dotenv_path=dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR_BFS = os.path.join(BASE_DIR, "prompts")
OUTPUT_DIR_BFS_ROOT = os.path.join(BASE_DIR, "output_bfs")

LOGS_DIR_BFS = os.path.join(OUTPUT_DIR_BFS_ROOT, "logs")
TEMP_FILES_DIR_BFS = os.path.join(OUTPUT_DIR_BFS_ROOT, "temp_files")
VERIFIED_IMAGES_FINAL_DIR_BFS = os.path.join(OUTPUT_DIR_BFS_ROOT, "verified_images")
REFERENCE_FACES_DIR_BFS = image_verifier_bfs.BFS_REFERENCE_IMAGES_BASE_DIR


SYSTEM_PROMPT_BFS_PATH = os.path.join(PROMPTS_DIR_BFS, "system_prompt_bfs.txt")
INTERMEDIARY_REQUEST_TEMPLATE_PATH = os.path.join(PROMPTS_DIR_BFS, "intermediary_request_template.xml")

# Default operational parameters
DEFAULT_MAX_DEPTH_BIDIRECTIONAL = 3
DEFAULT_MAX_SUGGESTIONS_PER_GEMINI_CALL = 10
DEFAULT_IMAGES_TO_DOWNLOAD_PER_LINK_BFS = 5
DEFAULT_DOWNLOAD_MAX_RETRIES = 3
DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS = 10
# Default verifier parameters (mirroring image_verifier_bfs.py defaults for clarity)
DEFAULT_NUM_TARGET_REF_IMAGES = image_verifier_bfs.DEFAULT_NUM_TARGET_REF_IMAGES
DEFAULT_NUM_REF_IMAGES_DOWNLOAD_POOL = image_verifier_bfs.DEFAULT_NUM_DOWNLOAD_FOR_REF_POOL
DEFAULT_VERIFICATION_MODELS = image_verifier_bfs.DEFAULT_VERIF_MODELS
DEFAULT_DETECTOR_BACKEND = image_verifier_bfs.DEFAULT_DETECTOR_BACKEND

# Fast Mode specific parameters
FAST_MODE_MAX_DEPTH = 2
FAST_MODE_IMAGES_TO_DOWNLOAD_PER_LINK = 3 # Changed from 1 to 3
FAST_MODE_NUM_TARGET_REF_IMAGES = 1
FAST_MODE_NUM_REF_IMAGES_DOWNLOAD_POOL = 3
FAST_MODE_VERIFICATION_MODELS = ['VGG-Face']
FAST_MODE_DETECTOR_BACKEND = 'opencv'

logger = logging.getLogger("VisualChainBiBFSApp")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    setup_logger = logging.getLogger("SetupLogger")
    setup_logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    logger = logging.getLogger("VisualChainBiBFSApp")

    if not os.path.exists(OUTPUT_DIR_BFS_ROOT):
        try:
            os.makedirs(OUTPUT_DIR_BFS_ROOT, exist_ok=True)
            setup_logger.info(f"Created root output directory {OUTPUT_DIR_BFS_ROOT}")
        except OSError as e:
            print(f"CRITICAL ERROR: Could not create root output directory {OUTPUT_DIR_BFS_ROOT}: {e}")
            logger.critical(f"Could not create root output directory {OUTPUT_DIR_BFS_ROOT}: {e}")

    utils_bfs.ensure_dir_bfs(LOGS_DIR_BFS, logger)
    log_file_path = os.path.join(LOGS_DIR_BFS, "visual_chain_bibfs_run.log")

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False


def run_single_bfs_expansion_step(
    queue_to_process,
    visited_people_as_source_for_this_bfs,
    all_nodes_in_this_bfs_paths,
    other_bfs_all_nodes,
    all_verified_links_global_list,
    max_depth_for_this_bfs,
    max_gemini_suggestions,
    # Parameters controlling download and verification behavior
    imgs_to_dl_for_link,
    download_max_retries,
    download_retry_delay,
    num_target_ref_images_arg,
    num_ref_images_download_pool_arg,
    verification_models_arg,
    detector_backend_arg,
    system_prompt,
    intermediary_template,
    bfs_direction_name
    ):
    if not queue_to_process:
        return False

    current_person, path_so_far = queue_to_process.popleft()

    current_path_str_display = current_person
    if path_so_far:
        display_nodes = [path_so_far[0][0]] + [link[1] for link in path_so_far]
        current_path_str_display = " → ".join(display_nodes)

    logger.info(f"\n--- BiBFS Step ({bfs_direction_name}) ---")
    logger.info(f"Dequeued: '{current_person}'. Current Path: [{current_path_str_display}]. Depth: {len(path_so_far)} / {max_depth_for_this_bfs}")

    if len(path_so_far) >= max_depth_for_this_bfs:
        logger.info(f"  Path for '{current_person}' ({bfs_direction_name}) reached max depth {max_depth_for_this_bfs}. Not expanding.")
        visited_people_as_source_for_this_bfs.add(current_person)
        return False

    nodes_to_avoid_for_gemini = set(visited_people_as_source_for_this_bfs)
    current_path_nodes_for_avoidance = set(link[1] for link in path_so_far)
    if path_so_far:
        current_path_nodes_for_avoidance.add(path_so_far[0][0])
    current_path_nodes_for_avoidance.add(current_person)
    nodes_to_avoid_for_gemini.update(current_path_nodes_for_avoidance)

    gemini_response_xml_str = gemini_bfs_query.get_intermediary_suggestions_bidirectional(
        current_person_name=current_person,
        max_suggestions_count=max_gemini_suggestions,
        nodes_to_avoid_list=list(nodes_to_avoid_for_gemini),
        system_prompt_content=system_prompt,
        user_input_template_str=intermediary_template,
        api_key=GOOGLE_API_KEY,
        logger=logger
    )

    if not gemini_response_xml_str or "<error>" in gemini_response_xml_str.lower()[:100]:
        logger.error(f"  Failed to get valid suggestions from Gemini for '{current_person}' ({bfs_direction_name}). Response: {gemini_response_xml_str[:300] if gemini_response_xml_str else 'None'}")
        visited_people_as_source_for_this_bfs.add(current_person)
        return False

    try:
        suggestions_root = ET.fromstring(gemini_response_xml_str)
        if suggestions_root.tag != "intermediary_suggestions":
            logger.error(f"  Unexpected root tag '{suggestions_root.tag}' ({bfs_direction_name}). Expected 'intermediary_suggestions'.")
            visited_people_as_source_for_this_bfs.add(current_person)
            return False

        if suggestions_root.find("no_further_suggestions_reason") is not None:
            reason = suggestions_root.findtext("no_further_suggestions_reason", "N/A")
            logger.info(f"  Gemini reported no further suggestions from '{current_person}' ({bfs_direction_name}): {reason}")
            visited_people_as_source_for_this_bfs.add(current_person)
            return False

        suggested_intermediaries = suggestions_root.findall("suggestion")
        if not suggested_intermediaries:
            logger.warning(f"  No <suggestion> tags found for '{current_person}' ({bfs_direction_name}). Checking for <intermediary> tags as a fallback.")
            suggested_intermediaries = suggestions_root.findall("intermediary")

        if not suggested_intermediaries:
            logger.info(f"  No <suggestion> or <intermediary> tags found in XML from Gemini for '{current_person}' ({bfs_direction_name}).")
            visited_people_as_source_for_this_bfs.add(current_person)
            return False

        def get_score(s_node):
            score_text = s_node.findtext("ranking_score", "0")
            try:
               return int(score_text)
            except ValueError:
               logger.warning(f"Invalid ranking_score '{score_text}', using 0.")
               return 0
        suggested_intermediaries.sort(key=get_score, reverse=True)
        logger.info(f"  Received {len(suggested_intermediaries)} suggestion items for '{current_person}' ({bfs_direction_name}). Processing...")


    except ET.ParseError as e:
        logger.error(f"  Failed to parse Gemini XML for '{current_person}' ({bfs_direction_name}): {e}. XML: {gemini_response_xml_str[:500]}")
        visited_people_as_source_for_this_bfs.add(current_person)
        return False

    met_other_bfs_node_this_step = False
    for sugg_idx, suggestion_node in enumerate(suggested_intermediaries):
        intermediary_name = suggestion_node.findtext("intermediary_person_name", "").strip()
        if not intermediary_name:
            logger.warning(f"    Suggestion {sugg_idx+1} has no intermediary_person_name. Skipping.")
            continue

        rank_score_text = suggestion_node.findtext("ranking_score", "N/A")
        logger.info(f"    Processing Suggestion {sugg_idx+1} (Rank: {rank_score_text}): '{current_person}' → '{intermediary_name}' ({bfs_direction_name})")

        connection_details_dict = utils_bfs.extract_connection_details_from_suggestion(suggestion_node, logger)
        link_google_query = f'"{current_person}" AND "{intermediary_name}"'
        logger.debug(f"      Using standardized Google query: {link_google_query}")
        connection_details_dict["google_query"] = link_google_query

        link_attempt_folder_name = f"{utils_bfs.sanitize_filename_bfs(current_person)}_to_{utils_bfs.sanitize_filename_bfs(intermediary_name)}_{bfs_direction_name}_{int(time.time())}"
        link_attempt_images_dl_folder = os.path.join(TEMP_FILES_DIR_BFS, link_attempt_folder_name)
        utils_bfs.ensure_dir_bfs(link_attempt_images_dl_folder, logger)

        verification_status, verification_data, verification_details_dict = image_verifier_bfs.verify_single_link_attempt(
            person1_name=current_person, person2_name=intermediary_name,
            google_query_for_link=link_google_query,
            link_attempt_images_dl_folder=link_attempt_images_dl_folder,
            logger_obj=logger,
            num_images_to_dl=imgs_to_dl_for_link,
            download_max_retries=download_max_retries,
            download_retry_delay=download_retry_delay,
            num_target_ref_images_to_acquire_per_person=num_target_ref_images_arg,
            ref_images_download_batch_size_for_person_pool=num_ref_images_download_pool_arg,
            list_of_verification_models_to_use=verification_models_arg,
            deepface_detector_backend_to_use_for_verification=detector_backend_arg
        )

        if verification_status == "VERIFIED_OK":
            verified_image_path_abs_in_temp = str(verification_data)
            logger.info(f"      SUCCESS: Verified '{current_person}' → '{intermediary_name}' ({bfs_direction_name}) with img: {os.path.basename(verified_image_path_abs_in_temp)}")

            if verification_details_dict:
                connection_details_dict.update(verification_details_dict)

            img_ext = os.path.splitext(verified_image_path_abs_in_temp)[1] or ".jpg"
            final_image_filename = f"{len(path_so_far)+1:02d}_{bfs_direction_name}_{utils_bfs.sanitize_filename_bfs(current_person)}_to_{utils_bfs.sanitize_filename_bfs(intermediary_name)}{img_ext}"
            final_image_path_dest_abs = os.path.join(VERIFIED_IMAGES_FINAL_DIR_BFS, final_image_filename)
            rel_image_path_for_xml = ""
            try:
                utils_bfs.ensure_dir_bfs(VERIFIED_IMAGES_FINAL_DIR_BFS, logger)
                shutil.copy(verified_image_path_abs_in_temp, final_image_path_dest_abs)
                rel_image_path_for_xml = os.path.join(os.path.basename(VERIFIED_IMAGES_FINAL_DIR_BFS), final_image_filename)
                logger.debug(f"      Copied verified image to: {final_image_path_dest_abs}")
            except Exception as e_copy:
                logger.error(f"      Error copying verified image '{verified_image_path_abs_in_temp}' to '{final_image_path_dest_abs}': {e_copy}. Using temp path in XML.")
                rel_image_path_for_xml = os.path.relpath(verified_image_path_abs_in_temp, OUTPUT_DIR_BFS_ROOT) + " (TEMP_UNCOPIED)"
                final_image_path_dest_abs = verified_image_path_abs_in_temp

            all_verified_links_global_list.append(
                (current_person, intermediary_name, connection_details_dict, final_image_path_dest_abs, rel_image_path_for_xml)
            )

            current_link_tuple_for_path = (current_person, intermediary_name, connection_details_dict, final_image_path_dest_abs, rel_image_path_for_xml)
            new_path = path_so_far + [current_link_tuple_for_path]

            if intermediary_name not in visited_people_as_source_for_this_bfs:
                queue_to_process.append((intermediary_name, new_path))
                all_nodes_in_this_bfs_paths.add(intermediary_name)
                logger.info(f"      Added '{intermediary_name}' to {bfs_direction_name} queue. New path depth: {len(new_path)}")

                if other_bfs_all_nodes and intermediary_name in other_bfs_all_nodes:
                    logger.info(f"!!! MEET-IN-THE-MIDDLE CANDIDATE: '{intermediary_name}' ({bfs_direction_name}) found in other BFS's node set!")
                    met_other_bfs_node_this_step = True
            else:
                 logger.info(f"      '{intermediary_name}' already processed as a source by {bfs_direction_name} BFS or in its own path. Not re-adding to queue.")
        else:
            logger.warning(f"      Verification FAILED for '{current_person}' → '{intermediary_name}' ({bfs_direction_name}). Status: {verification_status}, Details: {verification_data}")

    visited_people_as_source_for_this_bfs.add(current_person)
    return met_other_bfs_node_this_step


def find_visual_connection_bibfs(
    person1_start, person2_target,
    is_fast_mode=False,
    no_cleanup_arg=False,
    max_depth_override=None, max_gemini_suggestions_override=None,
    imgs_to_dl_override=None,
    download_max_retries_override=None,
    download_retry_delay_override=None
    ):

    # Step 1: Set base parameters depending on Fast Mode or Default Mode
    if is_fast_mode:
        logger.info("--- FAST MODE SELECTED (applying base fast settings) ---")
        base_max_depth = FAST_MODE_MAX_DEPTH
        base_images_to_dl_per_link = FAST_MODE_IMAGES_TO_DOWNLOAD_PER_LINK
        base_num_target_ref_images = FAST_MODE_NUM_TARGET_REF_IMAGES
        base_num_ref_images_download_pool = FAST_MODE_NUM_REF_IMAGES_DOWNLOAD_POOL
        base_verification_models = FAST_MODE_VERIFICATION_MODELS
        base_detector_backend = FAST_MODE_DETECTOR_BACKEND
        base_max_suggestions_per_gemini = DEFAULT_MAX_SUGGESTIONS_PER_GEMINI_CALL
        base_download_max_retries = DEFAULT_DOWNLOAD_MAX_RETRIES
        base_download_retry_delay = DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS
    else: # Default mode
        logger.info("--- DEFAULT MODE SELECTED (applying base default settings) ---")
        base_max_depth = DEFAULT_MAX_DEPTH_BIDIRECTIONAL
        base_images_to_dl_per_link = DEFAULT_IMAGES_TO_DOWNLOAD_PER_LINK_BFS
        base_num_target_ref_images = DEFAULT_NUM_TARGET_REF_IMAGES
        base_num_ref_images_download_pool = DEFAULT_NUM_REF_IMAGES_DOWNLOAD_POOL
        base_verification_models = DEFAULT_VERIFICATION_MODELS
        base_detector_backend = DEFAULT_DETECTOR_BACKEND
        base_max_suggestions_per_gemini = DEFAULT_MAX_SUGGESTIONS_PER_GEMINI_CALL
        base_download_max_retries = DEFAULT_DOWNLOAD_MAX_RETRIES
        base_download_retry_delay = DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS

    # Step 2: Apply command-line overrides to the base parameters
    actual_max_depth_bidirectional = max_depth_override if max_depth_override is not None else base_max_depth
    actual_images_to_dl_per_link = imgs_to_dl_override if imgs_to_dl_override is not None else base_images_to_dl_per_link
    actual_max_suggestions_per_gemini = max_gemini_suggestions_override if max_gemini_suggestions_override is not None else base_max_suggestions_per_gemini
    actual_download_max_retries = download_max_retries_override if download_max_retries_override is not None else base_download_max_retries
    actual_download_retry_delay = download_retry_delay_override if download_retry_delay_override is not None else base_download_retry_delay

    actual_num_target_ref_images = base_num_target_ref_images
    actual_num_ref_images_download_pool = base_num_ref_images_download_pool
    actual_verification_models = base_verification_models
    actual_detector_backend = base_detector_backend

    logger.info(f"--- Starting Bi-Directional BFS Visual Chain Generation ---")
    logger.info(f"Person 1 (Start): {person1_start}")
    logger.info(f"Person 2 (Target): {person2_target}")
    logger.info(f"--- Effective Configuration ---")
    logger.info(
        f"Search: MaxDepthPerDirection={actual_max_depth_bidirectional}, "
        f"MaxGeminiSugg={actual_max_suggestions_per_gemini}"
    )
    logger.info(
        f"Downloads: ImgsToDLPerLink={actual_images_to_dl_per_link}, MaxRetries={actual_download_max_retries}, RetryDelay={actual_download_retry_delay}s"
    )
    logger.info(
        f"Verifier: TargetRefImgs={actual_num_target_ref_images}, RefImgPool={actual_num_ref_images_download_pool}, "
        f"VerifModels={actual_verification_models}, Detector={actual_detector_backend}"
    )

    if not GOOGLE_API_KEY:
        logger.fatal("GOOGLE_API_KEY not found in .env file. Exiting.")
        return
    if not nx:
        logger.fatal("NetworkX library not found. Please install it (`pip install networkx`). Exiting.")
        return

    utils_bfs.ensure_dir_bfs(OUTPUT_DIR_BFS_ROOT, logger)
    utils_bfs.ensure_dir_bfs(LOGS_DIR_BFS, logger)

    if not no_cleanup_arg:
        logger.info("--- Cleaning up previous BiBFS output sub-directories (temp_files, verified_images) ---")
        utils_bfs.clean_output_sub_dir_bfs(TEMP_FILES_DIR_BFS, logger)
        utils_bfs.clean_output_sub_dir_bfs(VERIFIED_IMAGES_FINAL_DIR_BFS, logger)

    utils_bfs.ensure_dir_bfs(TEMP_FILES_DIR_BFS, logger)
    utils_bfs.ensure_dir_bfs(VERIFIED_IMAGES_FINAL_DIR_BFS, logger)
    utils_bfs.ensure_dir_bfs(REFERENCE_FACES_DIR_BFS, logger)

    try:
        system_prompt_bibfs = utils_bfs.load_prompt_template_bfs(SYSTEM_PROMPT_BFS_PATH, logger)
        intermediary_request_template_bibfs = utils_bfs.load_prompt_template_bfs(INTERMEDIARY_REQUEST_TEMPLATE_PATH, logger)
    except FileNotFoundError:
        logger.fatal("Essential BiBFS prompt templates missing. Exiting.")
        return
    if not system_prompt_bibfs or not intermediary_request_template_bibfs:
        logger.fatal("Failed to load essential BiBFS prompt templates. Exiting.")
        return

    queue_fwd = collections.deque([(person1_start, [])])
    visited_fwd_sources = set()
    all_nodes_fwd_paths = {person1_start}

    queue_bwd = collections.deque([(person2_target, [])])
    visited_bwd_sources = set()
    all_nodes_bwd_paths = {person2_target}

    all_verified_links_global = []
    path_found_via_intersection = False

    for current_processing_depth in range(actual_max_depth_bidirectional + 1):
        logger.info(f"\n=== BiBFS Processing Depth Level {current_processing_depth} (Max: {actual_max_depth_bidirectional}) ===")

        intersection = all_nodes_fwd_paths.intersection(all_nodes_bwd_paths)
        if intersection:
            logger.info(f"!!! INTERSECTION DETECTED at start of depth {current_processing_depth}! Nodes: {intersection}")
            path_found_via_intersection = True
            break

        nodes_to_process_this_level_fwd = sum(1 for item in queue_fwd if len(item[1]) == current_processing_depth)
        if nodes_to_process_this_level_fwd > 0:
            logger.info(f"  Expanding {nodes_to_process_this_level_fwd} FWD nodes at depth {current_processing_depth}")
        for _ in range(nodes_to_process_this_level_fwd):
            if not queue_fwd or len(queue_fwd[0][1]) != current_processing_depth:
                break
            if run_single_bfs_expansion_step(
                queue_fwd, visited_fwd_sources, all_nodes_fwd_paths, all_nodes_bwd_paths,
                all_verified_links_global, actual_max_depth_bidirectional,
                actual_max_suggestions_per_gemini, actual_images_to_dl_per_link,
                actual_download_max_retries, actual_download_retry_delay,
                actual_num_target_ref_images, actual_num_ref_images_download_pool,
                actual_verification_models, actual_detector_backend,
                system_prompt_bibfs, intermediary_request_template_bibfs, "Forward"
            ):
                path_found_via_intersection = True; break
        if path_found_via_intersection: logger.info(f"!!! INTERSECTION DETECTED during FWD expansion!"); break

        intersection = all_nodes_fwd_paths.intersection(all_nodes_bwd_paths)
        if intersection:
            logger.info(f"!!! INTERSECTION DETECTED after FWD expansion at depth {current_processing_depth}! Nodes: {intersection}")
            path_found_via_intersection = True; break

        nodes_to_process_this_level_bwd = sum(1 for item in queue_bwd if len(item[1]) == current_processing_depth)
        if nodes_to_process_this_level_bwd > 0:
            logger.info(f"  Expanding {nodes_to_process_this_level_bwd} BWD nodes at depth {current_processing_depth}")
        for _ in range(nodes_to_process_this_level_bwd):
            if not queue_bwd or len(queue_bwd[0][1]) != current_processing_depth:
                break
            if run_single_bfs_expansion_step(
                queue_bwd, visited_bwd_sources, all_nodes_bwd_paths, all_nodes_fwd_paths,
                all_verified_links_global, actual_max_depth_bidirectional,
                actual_max_suggestions_per_gemini, actual_images_to_dl_per_link,
                actual_download_max_retries, actual_download_retry_delay,
                actual_num_target_ref_images, actual_num_ref_images_download_pool,
                actual_verification_models, actual_detector_backend,
                system_prompt_bibfs, intermediary_request_template_bibfs, "Backward"
            ):
                path_found_via_intersection = True; break
        if path_found_via_intersection: logger.info(f"!!! INTERSECTION DETECTED during BWD expansion!"); break

        intersection = all_nodes_fwd_paths.intersection(all_nodes_bwd_paths)
        if intersection:
            logger.info(f"!!! INTERSECTION DETECTED after BWD expansion at depth {current_processing_depth}! Nodes: {intersection}")
            path_found_via_intersection = True; break

        if not queue_fwd and not queue_bwd:
            logger.info(f"Both BFS queues are empty after processing depth {current_processing_depth}. Ending expansion.")
            break
        if nodes_to_process_this_level_fwd == 0 and nodes_to_process_this_level_bwd == 0 and (queue_fwd or queue_bwd):
            logger.info(f"No nodes expanded at depth {current_processing_depth}. Remaining queue items are deeper or unexpandable. Continuing if within max depth.")


    logger.info(f"\n--- Bi-Directional BFS Expansion Phase Complete ---")
    if path_found_via_intersection:
        logger.info("An intersection between forward and backward searches was detected during expansion.")
    logger.info(f"Total verified links found globally: {len(all_verified_links_global)}")

    if not all_verified_links_global:
        logger.info(f"No verifiable links found. Cannot build graph or find path between '{person1_start}' and '{person2_target}'.")
        utils_bfs.save_chain_to_xml(person1_start, [], OUTPUT_DIR_BFS_ROOT, logger, no_path_person2=person2_target)
        logger.info("BiBFS Process Complete.")
        return

    logger.info("Constructing graph from all verified links...")
    G = nx.Graph()
    link_data_map = {}

    for p_from, p_to, details, img_abs, img_rel in all_verified_links_global:
        G.add_edge(p_from, p_to)
        link_data_map[(p_from, p_to)] = (details, img_abs, img_rel)
        if (p_to, p_from) not in link_data_map:
             link_data_map[(p_to, p_from)] = (details, img_abs, img_rel)


    if not G.has_node(person1_start) or not G.has_node(person2_target):
        logger.warning(f"Graph missing start ('{person1_start}') or target ('{person2_target}') node. Pathfinding not possible.")
        if not G.has_node(person1_start): logger.warning(f"Start node '{person1_start}' not in graph.")
        if not G.has_node(person2_target): logger.warning(f"Target node '{person2_target}' not in graph.")
        logger.debug(f"Nodes in graph: {list(G.nodes())}")
        utils_bfs.save_chain_to_xml(person1_start, [], OUTPUT_DIR_BFS_ROOT, logger, no_path_person2=person2_target)
        logger.info("BiBFS Process Complete.")
        return

    logger.info(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    logger.info(f"Attempting to find shortest path from '{person1_start}' to '{person2_target}'...")

    final_path_tuples_for_xml = []
    try:
        if nx.has_path(G, source=person1_start, target=person2_target):
            shortest_path_nodes = nx.shortest_path(G, source=person1_start, target=person2_target)
            logger.info(f"Shortest path found: {' → '.join(shortest_path_nodes)}")

            for i in range(len(shortest_path_nodes) - 1):
                u, v = shortest_path_nodes[i], shortest_path_nodes[i+1]
                data = link_data_map.get((u,v))
                if not data:
                    data = link_data_map.get((v,u))

                if data:
                    details, img_abs, img_rel = data
                    final_path_tuples_for_xml.append((u, v, details, img_abs, img_rel))
                else:
                    logger.error(f"CRITICAL: No link data found for segment {u}–{v} in reconstructed path!")
                    final_path_tuples_for_xml.append((u, v, {"error":f"Link data missing for {u}-{v}"}, "", ""))

            if final_path_tuples_for_xml or len(shortest_path_nodes) == 1:
                 if len(final_path_tuples_for_xml) == len(shortest_path_nodes) - 1 :
                    utils_bfs.save_chain_to_xml(person1_start, final_path_tuples_for_xml, OUTPUT_DIR_BFS_ROOT, logger)
                 elif len(shortest_path_nodes) == 1 and person1_start == person2_target :
                    logger.info(f"Start and target nodes are the same: {person1_start}. No path segments needed.")
                    utils_bfs.save_chain_to_xml(person1_start, [], OUTPUT_DIR_BFS_ROOT, logger, no_path_person2=person2_target)
                 else:
                    logger.error(f"Path reconstruction mismatch. Expected {len(shortest_path_nodes)-1} segments, got {len(final_path_tuples_for_xml)} with data.")
                    utils_bfs.save_chain_to_xml(person1_start, final_path_tuples_for_xml, OUTPUT_DIR_BFS_ROOT, logger, error_message="Path reconstruction mismatch")

        else:
            logger.info(f"No path could be found between '{person1_start}' and '{person2_target}' in the constructed graph.")
            utils_bfs.save_chain_to_xml(person1_start, [], OUTPUT_DIR_BFS_ROOT, logger, no_path_person2=person2_target)

    except nx.NodeNotFound as e:
        logger.error(f"Node not found during pathfinding: {e}")
        utils_bfs.save_chain_to_xml(person1_start, [], OUTPUT_DIR_BFS_ROOT, logger, no_path_person2=person2_target, error_message=str(e))
    except Exception as e_path:
        logger.error(f"Error during graph pathfinding or XML generation: {e_path}", exc_info=True)
        utils_bfs.save_chain_to_xml(person1_start, [], OUTPUT_DIR_BFS_ROOT, logger, no_path_person2=person2_target, error_message=str(e_path))

    logger.info("BiBFS Process Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bi-directional BFS for visual chain.")
    parser.add_argument("--person1", required=True, help="Name of Person 1 (start).")
    parser.add_argument("--person2", required=True, help="Name of Person 2 (target).")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip output dir cleanup.")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode with reduced image counts, simpler verification, and shallower depth. Specific settings like --depth can override fast mode defaults.")
    parser.add_argument("--depth", type=int, dest="max_depth",
                        help=f"Max depth for EACH BFS direction. Overrides default ({DEFAULT_MAX_DEPTH_BIDIRECTIONAL}) or fast mode's default ({FAST_MODE_MAX_DEPTH}) depth.")
    parser.add_argument("--max-sugg", type=int, dest="max_gemini_suggestions",
                        help=f"Max Gemini suggestions per call. Default {DEFAULT_MAX_SUGGESTIONS_PER_GEMINI_CALL}")
    parser.add_argument("--imgs-dl", type=int, dest="imgs_to_dl",
                        help=f"Images to download per link. Overrides default ({DEFAULT_IMAGES_TO_DOWNLOAD_PER_LINK_BFS}) or fast mode's default ({FAST_MODE_IMAGES_TO_DOWNLOAD_PER_LINK}) value.")
    parser.add_argument("--dl-retries", type=int, dest="download_max_retries",
                        help=f"Max download attempts per link (incl. initial). Default {DEFAULT_DOWNLOAD_MAX_RETRIES}")
    parser.add_argument("--dl-delay", type=int, dest="download_retry_delay",
                        help=f"Delay in seconds between download retries. Default {DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS}")
    args = parser.parse_args()

    find_visual_connection_bibfs(
        args.person1, args.person2,
        is_fast_mode=args.fast,
        no_cleanup_arg=args.no_cleanup,
        max_depth_override=args.max_depth,
        max_gemini_suggestions_override=args.max_gemini_suggestions,
        imgs_to_dl_override=args.imgs_to_dl,
        download_max_retries_override=args.download_max_retries,
        download_retry_delay_override=args.download_retry_delay
    )