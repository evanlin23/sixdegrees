import google.generativeai as genai
import os
from dotenv import load_dotenv
import xml.etree.ElementTree as ET # For validating output (optional but good)

# Load environment variables (for API key)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or environment.")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Configuration ---
PERSON_A = "Charlie Kirk"
PERSON_B = "James Charles"
MODEL_NAME = "gemini-2.5-flash-preview-05-20" # Or 'gemini-pro' or other compatible models

def load_system_prompt(filepath="prompt.txt"):
    """Loads the system prompt from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: System prompt file '{filepath}' not found.")
        return None

def construct_user_input_xml(person1_name, person2_name):
    """Constructs the XML input for the AI."""
    return f"""
<connection_request>
  <person1>{person1_name}</person1>
  <person2>{person2_name}</person2>
</connection_request>
"""

def query_gemini(system_prompt, user_input):
    """Sends the combined prompt to Gemini and gets the response."""
    if not system_prompt:
        return "Error: System prompt is missing."

    full_prompt = f"""{system_prompt}

## Task Execution
Now, using the rules, input format, and output formats defined above, please process the following connection request:

{user_input}
"""
    print("--- Sending to Gemini ---")
    # print(full_prompt)
    print("-------------------------\n")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(full_prompt)
        
        ai_output = ""
        if response.parts:
            ai_output = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text'):
            ai_output = response.text
        else:
            ai_output = str(response) # Fallback

        # --- More Robust Cleaning ---
        ai_output = ai_output.strip() # Remove leading/trailing whitespace

        # 1. Remove common markdown code block fences
        if ai_output.startswith("```xml"):
            ai_output = ai_output[5:] # len("```xml")
            if ai_output.endswith("```"):
                ai_output = ai_output[:-3]
            ai_output = ai_output.strip()
        elif ai_output.startswith("```"): # More general markdown block
            ai_output = ai_output[3:]
            if ai_output.endswith("```"):
                ai_output = ai_output[:-3]
            ai_output = ai_output.strip()
        
        # 2. Find the first '<' which should be the start of the XML
        #    This will effectively remove any leading garbage like the 'l'
        first_angle_bracket = ai_output.find('<')
        if first_angle_bracket > 0:
            # print(f"NOTE: Stripping leading non-XML characters: '{ai_output[:first_angle_bracket]}'") # For debugging
            ai_output = ai_output[first_angle_bracket:]
        elif first_angle_bracket == -1 and ai_output: # No XML tag found, but there's content
            print("WARNING: AI response does not appear to start with XML after cleaning.")
            # You might want to return the problematic string or an error here
            # For now, we'll let it pass to the validator to report the error

        return ai_output

    except Exception as e:
        print(f"An error occurred while querying Gemini: {e}")
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            print(f"Prompt Feedback: {response.prompt_feedback}")
        # Ensure some form of XML-like error is returned if possible for consistency
        return f"<error><type>APIError</type><message>Gemini API call failed: {str(e)}</message></error>"


def validate_xml(xml_string):
    """Tries to parse the XML string to check if it's well-formed."""
    try:
        ET.fromstring(xml_string)
        return True, "XML is well-formed."
    except ET.ParseError as e:
        return False, f"XML ParseError: {e}"

if __name__ == "__main__":
    system_prompt_content = load_system_prompt()

    if system_prompt_content:
        user_xml_input = construct_user_input_xml(PERSON_A, PERSON_B)
        print(f"--- Constructed User XML Input for {PERSON_A} and {PERSON_B} ---")
        print(user_xml_input)
        print("--------------------------------------------------\n")

        ai_response = query_gemini(system_prompt_content, user_xml_input)

        print("--- AI Response ---")
        print(ai_response)
        print("-------------------\n")

        is_valid, message = validate_xml(ai_response)
        print(f"DEBUG: {message}")
        if is_valid:
            print(f"SUCCESS: {message}")
        else:
            print(f"WARNING: {message}. The AI response might not be valid XML as per instructions.")