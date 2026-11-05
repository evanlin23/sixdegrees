# bfs/gemini_bfs_query.py
import google.generativeai as genai
import os
import xml.etree.ElementTree as ET
import time
import html

from utils_bfs import load_prompt_template_bfs, sanitize_filename_bfs # Added sanitize_filename_bfs

TEXT_MODEL_NAME_BFS = "gemini-1.5-flash-latest"
GEMINI_TASK_HEADER_FILE_PATH_BFS = os.path.join(os.path.dirname(__file__), "prompts", "gemini_task_header.txt")
API_CALL_DELAY_SECONDS = 2

def clean_gemini_xml_response_bfs(ai_output, logger):
    logger.debug(f"BFS Raw AI output before cleaning (first 300 chars): {ai_output[:300]}")
    ai_output = ai_output.strip()
    if ai_output.startswith("```xml"):
        ai_output = ai_output[len("```xml"):]
        if ai_output.endswith("```"): ai_output = ai_output[:-3]
        ai_output = ai_output.strip()
    elif ai_output.startswith("```"):
        ai_output = ai_output[3:]
        if ai_output.endswith("```"): ai_output = ai_output[:-3]
        ai_output = ai_output.strip()

    first_angle_bracket = ai_output.find('<')
    if first_angle_bracket > 0:
        logger.debug(f"BFS Stripping leading non-XML: '{ai_output[:first_angle_bracket]}'")
        ai_output = ai_output[first_angle_bracket:]
    elif first_angle_bracket == -1 and ai_output.strip() and not ai_output.startswith("<"):
        logger.warning(f"BFS AI response no XML start after cleaning: '{ai_output[:100]}'")
        return f"<error><type>MalformedResponse</type><message>AI resp (BFS) no XML start after cleaning.</message><raw_response_snippet>{html.escape(ai_output[:200])}</raw_response_snippet></error>" # Escaped raw_response
    logger.debug(f"BFS AI output after cleaning (first 300 chars): {ai_output[:300]}")
    return ai_output

def get_intermediary_suggestions_bidirectional(
    current_person_name,
    max_suggestions_count,
    nodes_to_avoid_list,
    system_prompt_content,
    user_input_template_str,
    api_key,
    logger
):
    logger.info(f"BFS Gemini (BiDir): Requesting suggestions from '{current_person_name}'. Avoiding {len(nodes_to_avoid_list)} nodes.")

    if not api_key:
        logger.error("GOOGLE_API_KEY not configured for BFS.")
        return "<error><type>ConfigurationError</type><message>GOOGLE_API_KEY not configured.</message></error>"
    try:
        genai.configure(api_key=api_key)
    except Exception as e_conf:
        logger.error(f"Error configuring GenAI (BFS): {e_conf}")
        return f"<error><type>ConfigurationError</type><message>GenAI configuration failed (BFS): {html.escape(str(e_conf))}</message></error>" # Escaped error

    avoid_nodes_xml_elements_str = ""
    if nodes_to_avoid_list:
        for node_name in nodes_to_avoid_list:
            avoid_nodes_xml_elements_str += f"    <person>{html.escape(node_name)}</person>\n"

    try:
        user_input_xml_payload = user_input_template_str.format(
            current_person_name=html.escape(current_person_name),
            max_suggestions_count=max_suggestions_count,
            avoid_nodes_xml_list=avoid_nodes_xml_elements_str
        )
    except KeyError as e_format:
        logger.error(f"Failed to format BFS BiDir user input template. Missing key: {e_format}. Template snippet: {user_input_template_str[:200]}")
        return f"<error><type>TemplateFormatError</type><message>Failed to format BFS BiDir template: {html.escape(str(e_format))}</message></error>" # Escaped error
    except Exception as e_other_format:
        logger.error(f"Unexpected error formatting BFS BiDir user input template: {e_other_format}. Template snippet: {user_input_template_str[:200]}")
        return f"<error><type>TemplateFormatError</type><message>Unexpected error formatting BFS BiDir template: {html.escape(str(e_other_format))}</message></error>" # Escaped error

    task_header = load_prompt_template_bfs(GEMINI_TASK_HEADER_FILE_PATH_BFS, logger, is_critical=False)
    if task_header is None: task_header = "## Task Execution\nPlease process:\n"

    full_prompt = f"{system_prompt_content}\n{task_header}\n{user_input_xml_payload}"

    log_dir_path = "."
    try:
        # Attempt to get the log directory from the logger's file handler
        if logger.handlers:
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler) and hasattr(handler, 'baseFilename'):
                    log_dir_path = os.path.dirname(handler.baseFilename)
                    break
        # Fallback if still '.', try to use the script's directory relative path (assuming output_bfs/logs)
        if log_dir_path == ".":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            potential_log_dir = os.path.join(script_dir, "output_bfs", "logs")
            if os.path.isdir(potential_log_dir):
                 log_dir_path = potential_log_dir
            else: # Final fallback to script's dir if output_bfs/logs doesn't exist
                log_dir_path = script_dir
    except Exception as e_logdir:
        logger.warning(f"Could not reliably determine log directory, defaulting to current: {e_logdir}")
        log_dir_path = "."


    log_filepath = os.path.join(log_dir_path,
                                f"full_prompt_to_gemini_BiBFS_{sanitize_filename_bfs(current_person_name)}_{int(time.time())}.txt")
    try:
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True) # Ensure dir exists
        with open(log_filepath, "w", encoding="utf-8") as f_prompt:
            f_prompt.write(full_prompt)
        logger.info(f"Full prompt to Gemini (BiBFS) saved to: {log_filepath}")
    except Exception as e_write_prompt:
        logger.error(f"Could not write full prompt (BiBFS) to file {log_filepath}: {e_write_prompt}")

    logger.debug(f"BFS User input XML for Gemini (BiDir, part of full prompt, snippet):\n{user_input_xml_payload[:500]}")
    logger.info(f"--- Sending to Gemini (BFS BiDir intermediary suggestions for '{current_person_name}') ---")

    try:
        time.sleep(API_CALL_DELAY_SECONDS)
        model = genai.GenerativeModel(TEXT_MODEL_NAME_BFS)
        response = model.generate_content(full_prompt)

        ai_output = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        cleaned_output = clean_gemini_xml_response_bfs(ai_output, logger)

        try:
            parsed_root = ET.fromstring(cleaned_output)
            if parsed_root.tag not in ["intermediary_suggestions", "error"]:
                 logger.warning(f"BFS Gemini response unexpected root: {parsed_root.tag}. Expected 'intermediary_suggestions'.")
            logger.info(f"BFS Gemini response parsed (BiDir), root: {parsed_root.tag}")
        except ET.ParseError as e_parse:
            logger.error(f"BFS Gemini response not well-formed XML (BiDir): {e_parse}")
            logger.debug(f"BFS Malformed XML (BiDir): {cleaned_output}")
            return f"<error><type>MalformedXMLFromAI</type><message>AI response (BFS BiDir) not XML: {html.escape(str(e_parse))}</message><raw_response>{html.escape(cleaned_output)}</raw_response></error>" # Escaped messages
        return cleaned_output
    except Exception as e:
        logger.error(f"Error querying Gemini (BFS BiDir): {e}", exc_info=True)
        prompt_feedback_info = ""
        response_obj = locals().get('response')
        if response_obj and hasattr(response_obj, 'prompt_feedback') and response_obj.prompt_feedback:
            logger.warning(f"Prompt Feedback from Gemini (BFS BiDir): {response_obj.prompt_feedback}")
            prompt_feedback_info = f"<prompt_feedback>{html.escape(str(response_obj.prompt_feedback))}</prompt_feedback>" # Escaped feedback
        return f"<error><type>APIError</type><message>Gemini API call (BFS BiDir) failed: {html.escape(str(e))}</message>{prompt_feedback_info}</error>" # Escaped error message

if __name__ == "__main__":
    print("This script (gemini_bfs_query.py) is intended to be called by main_bfs.py")