# initial_gemini_query.py
import google.generativeai as genai
import os
import xml.etree.ElementTree as ET

TEXT_MODEL_NAME = "gemini-1.5-flash-latest"
GEMINI_TASK_HEADER_FILE_PATH = os.path.join("prompts", "gemini_task_header.txt") # New

def _load_gemini_task_header(logger_obj):
    try:
        with open(GEMINI_TASK_HEADER_FILE_PATH, "r", encoding="utf-8") as f:
            header = f.read()
        logger_obj.info(f"Gemini task header loaded from {GEMINI_TASK_HEADER_FILE_PATH}")
        return header
    except FileNotFoundError:
        logger_obj.error(f"FATAL: Gemini task header file '{GEMINI_TASK_HEADER_FILE_PATH}' not found. Using default.")
        return ("## Task Execution\nNow, using the rules, input format, and output formats defined above, "
                "please process the following connection request to find a full chain:\n\n") # Fallback
    except Exception as e:
        logger_obj.error(f"Error loading Gemini task header file '{GEMINI_TASK_HEADER_FILE_PATH}': {e}. Using default.")
        return ("## Task Execution\nNow, using the rules, input format, and output formats defined above, "
                "please process the following connection request to find a full chain:\n\n") # Fallback


def clean_gemini_xml_response(ai_output, logger):
    logger.debug(f"Raw AI output before cleaning (first 300 chars): {ai_output[:300]}")
    ai_output = ai_output.strip()
    if ai_output.startswith("```xml"):
        ai_output = ai_output[len("```xml"):]
        if ai_output.endswith("```"):
            ai_output = ai_output[:-3]
        ai_output = ai_output.strip()
    elif ai_output.startswith("```"): 
        ai_output = ai_output[3:]
        if ai_output.endswith("```"):
            ai_output = ai_output[:-3]
        ai_output = ai_output.strip()
    
    first_angle_bracket = ai_output.find('<')
    if first_angle_bracket > 0:
        logger.debug(f"Stripping leading non-XML characters: '{ai_output[:first_angle_bracket]}'")
        ai_output = ai_output[first_angle_bracket:]
    elif first_angle_bracket == -1 and ai_output.strip() and not ai_output.startswith("<"):
        logger.warning(f"AI response has content but does not appear to start with XML after cleaning: '{ai_output[:100]}'")
        return f"<error><type>MalformedResponse</type><message>AI response has content but does not appear to start with XML after cleaning.</message><raw_response_snippet>{ai_output[:200]}</raw_response_snippet></error>"
    logger.debug(f"AI output after cleaning (first 300 chars): {ai_output[:300]}")
    return ai_output

def get_initial_chain_from_gemini_direct(system_prompt_content, user_input_xml_for_gemini, api_key, logger):
    logger.info(f"Querying Gemini with pre-formatted user input for a chain.")
    if not api_key:
        logger.error("GOOGLE_API_KEY not configured.")
        return "<error><type>ConfigurationError</type><message>GOOGLE_API_KEY not configured.</message></error>"
    if not system_prompt_content:
        logger.error("System prompt is missing.")
        return "<error><type>ConfigurationError</type><message>System prompt is missing.</message></error>"
    if not user_input_xml_for_gemini:
        logger.error("Formatted user input XML for Gemini is missing.")
        return "<error><type>ConfigurationError</type><message>Formatted user input XML for Gemini is missing.</message></error>"
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e_conf:
        logger.error(f"Error configuring GenAI: {e_conf}")
        return f"<error><type>ConfigurationError</type><message>GenAI configuration failed: {e_conf}</message></error>"

    full_prompt = f"{system_prompt_content}\n{user_input_xml_for_gemini}"
    
    log_user_input_snippet = user_input_xml_for_gemini
    if len(user_input_xml_for_gemini) > 600 : 
        log_user_input_snippet = user_input_xml_for_gemini[:300] + "\n... (user input snippet) ...\n" + user_input_xml_for_gemini[-300:]
    
    logger.debug(f"User input XML for Gemini (part of full prompt):\n{log_user_input_snippet}")
    logger.info(f"--- Sending to Gemini (initial_gemini_query.get_initial_chain_from_gemini_direct) ---")

    try:
        model = genai.GenerativeModel(TEXT_MODEL_NAME)
        response = model.generate_content(full_prompt)
        
        ai_output = ""
        if response.parts:
            ai_output = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text'):
            ai_output = response.text
        else: 
            ai_output = str(response) if response else ""

        cleaned_output = clean_gemini_xml_response(ai_output, logger)
        
        try:
            parsed_root_tag = ET.fromstring(cleaned_output).tag
            logger.info(f"Gemini response parsed as XML (root tag: {parsed_root_tag}).")
        except ET.ParseError as e_parse:
            logger.error(f"Gemini response is not well-formed XML after cleaning: {e_parse}")
            logger.debug(f"Malformed XML content: {cleaned_output}")
            return f"<error><type>MalformedXMLFromAI</type><message>Response from AI not well-formed XML: {e_parse}</message><raw_response>{cleaned_output}</raw_response></error>"

        logger.info(f"Received response from Gemini.")
        return cleaned_output

    except Exception as e:
        logger.error(f"An error occurred while querying Gemini: {e}", exc_info=True)
        prompt_feedback_info = ""
        if 'response' in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            logger.warning(f"Prompt Feedback from Gemini: {response.prompt_feedback}")
            prompt_feedback_info = f"<prompt_feedback>{response.prompt_feedback}</prompt_feedback>"
        return f"<error><type>APIError</type><message>Gemini API call failed: {str(e)}</message>{prompt_feedback_info}</error>"

def get_initial_chain(person_a, person_b, system_prompt_content, user_input_template_str, api_key, logger, exclusion_instruction=""):
    logger.info(f"Requesting chain: {person_a} -> ... -> {person_b}.")
    if exclusion_instruction:
        logger.info(f"With exclusion instruction: {exclusion_instruction}")

    try:
        user_input_xml_content = user_input_template_str.format(
            person1_name=person_a, 
            person2_name=person_b,
            additional_instructions=exclusion_instruction 
        )
    except KeyError as e_format:
        logger.error(f"Failed to format user input template. Missing key: {e_format}. Template: {user_input_template_str}")
        return f"<error><type>TemplateFormatError</type><message>Failed to format user input template: {e_format}</message></error>"
    
    task_header = _load_gemini_task_header(logger) # Load from file
    user_input_xml_for_gemini = task_header + user_input_xml_content

    return get_initial_chain_from_gemini_direct(system_prompt_content, user_input_xml_for_gemini, api_key, logger)


if __name__ == "__main__":
    print("This script is intended to be called by main_orchestrator.py")