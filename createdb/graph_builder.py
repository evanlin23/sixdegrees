import pandas as pd
import networkx as nx
import os
import time
import json # For caching
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process # For fuzzy name matching
import re # For parsing retry_delay
import random # For jitter in backoff
import signal # For handling Ctrl+C
import argparse # For command-line arguments

# --- LLM Client Imports (conditional) ---
try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    import ollama
except ImportError:
    ollama = None

# --- Configuration ---
CSV_FILE_PATH = "people.csv"
OUTPUT_GRAPH_FILE = "met_people_graph.gexf"
LLM_CACHE_FILE = "llm_responses_cache.json"
PROGRESS_FILE = "processing_progress.txt"

MAX_PEOPLE_TO_PROCESS_IN_SESSION = None
API_CALL_DELAY_SECONDS = 4
MAX_RETRIES_API_CALL = 5 # Max retries for API call itself (network, rate limit)
MAX_FORMAT_CORRECTION_ATTEMPTS = 2 # How many times to re-prompt if format is bad (within one API call attempt)
FUZZY_MATCH_THRESHOLD = 90
BIRTHYEAR_LOWER_BOUND = -9999 # Allow for ancient figures

DEFAULT_OLLAMA_MODEL = "phi3:mini"

# --- Global graph object and state for signal handler ---
current_graph_object = nx.Graph()
graph_state_initialized = False # NEW: Flag to track graph initialization status

# --- Helper for ETA and time formatting ---
def format_time_delta(seconds):
    if seconds is None or seconds < 0:
        return "N/A"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes:02d}m {secs:02d}s"
    else:
        return f"{secs:02d}s"

# --- Caching Functions ---
def load_cache():
    if os.path.exists(LLM_CACHE_FILE):
        try:
            with open(LLM_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Cache file {LLM_CACHE_FILE} is corrupted. Starting with an empty cache.")
            return {}
        except Exception as e:
            print(f"Warning: Could not load cache file {LLM_CACHE_FILE}: {e}. Starting with an empty cache.")
            return {}
    return {}

def save_cache(cache_data):
    try:
        with open(LLM_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"Error saving cache to {LLM_CACHE_FILE}: {e}")

# --- Progress Functions ---
def load_start_index():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return int(content)
        except ValueError:
            print(f"Warning: Progress file {PROGRESS_FILE} content is not a valid integer. Starting from scratch.")
        except FileNotFoundError: pass
        except Exception as e: print(f"Warning: Could not load progress file {PROGRESS_FILE}: {e}. Starting from scratch.")
    return 0

def save_progress(index):
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f: f.write(str(index))
    except Exception as e: print(f"Error saving progress to {PROGRESS_FILE}: {e}")

# --- Signal Handler for Ctrl+C ---
def save_graph_on_interrupt(signum, frame):
    global current_graph_object, graph_state_initialized # MODIFIED: include graph_state_initialized
    print("\nCtrl+C detected.")

    if not graph_state_initialized:
        print("Interrupt occurred during initial script setup (before graph was fully loaded/initialized).")
        print(f"To prevent accidental overwrite of '{OUTPUT_GRAPH_FILE}' with an incomplete graph, no save operation will be performed at this stage.")
        print("If the script was previously interrupted, the existing graph file (if any) should be intact.")
        print("Exiting script prematurely."); exit(1) # Exit with a non-zero code to indicate abnormal termination

    print("Attempting to save current graph state...")
    if current_graph_object is not None and current_graph_object.number_of_nodes() > 0:
        try:
            nx.write_gexf(current_graph_object, OUTPUT_GRAPH_FILE)
            print(f"Partial graph saved to {OUTPUT_GRAPH_FILE} (Nodes: {current_graph_object.number_of_nodes()}, Edges: {current_graph_object.number_of_edges()})")
        except Exception as e: print(f"Error saving graph on interrupt: {e}")
    else:
        # This case means graph_state_initialized is True, but the graph is empty.
        # This can happen if:
        # 1. It's the first run and no data was processed yet before interrupt.
        # 2. An existing graph file was empty or unreadable, and no new data was processed yet.
        print("Graph is currently empty. Not saving an empty graph on interrupt.")
        print(f"If '{OUTPUT_GRAPH_FILE}' existed and was meant to be loaded, check for prior errors regarding its loading.")
    print("Exiting script."); exit(0) # Normal exit after attempting/skipping save

# --- LLM Query Functions ---
llm_cache = {}
gemini_model = None

def initialize_gemini():
    global gemini_model
    if not genai:
        print("ERROR: google-generativeai library not installed. Cannot use Gemini.")
        return False
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in .env file or environment. Cannot use Gemini.")
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Gemini API initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        return False

def is_valid_name_list_format(text_response: str) -> bool:
    if not text_response or not isinstance(text_response, str):
        return False
    
    text_response = text_response.strip()
    if not text_response:
        return True

    bad_intros = [
        "here is a list of people", "based on my knowledge", "individuals that",
        "i can list the following", "sure, here are some", "the people that",
        "some of the notable individuals", "considering the figure"
    ]
    if any(text_response.lower().startswith(intro) for intro in bad_intros) and ',' not in text_response:
        if len(text_response.split()) > 15:
            return False

    if re.fullmatch(r"^[a-zA-Z0-9 .,'-]+$", text_response):
        if text_response.lower() in ["none", "n/a", "unknown", "no one", "no known meetings"]:
            return True
        return True
    
    if ',' not in text_response and len(text_response.split()) > 7:
        print(f"      Format Check: Suspicious response - many words, no commas: '{text_response[:100]}...'")
        return False
    return True


def get_people_met_from_gemini(person_name, birth_year, death_year, occupation, alive_status):
    global llm_cache, gemini_model
    if not gemini_model:
        print("   ERROR: Gemini model not initialized. Skipping API call.")
        return []

    prompt_version = "gemini_v2.1_fmt_strict"
    cache_key = f"{person_name}_{birth_year}_{occupation}_{alive_status}_{prompt_version}"

    if cache_key in llm_cache:
        print(f"   Cache hit for: {person_name} (Provider: Gemini, Prompt: {prompt_version})")
        return llm_cache[cache_key]

    life_span = f"Born: {birth_year if pd.notna(birth_year) else 'Unknown'}"
    current_alive_status_text = ""
    if alive_status is pd.NA: current_alive_status = False
    elif isinstance(alive_status, bool): current_alive_status = alive_status
    else: current_alive_status = str(alive_status).upper() == 'TRUE'

    if current_alive_status: life_span += " (Alive)"; current_alive_status_text = " (currently alive)"
    elif pd.notna(death_year): life_span += f", Died: {death_year}"
    else: life_span += ", Death year unknown"

    base_prompt_text = (
        f"Consider the historical or public figure: {person_name} "
        f"(Occupation: {occupation if pd.notna(occupation) else 'Unknown'}, {life_span}{current_alive_status_text}). "
        f"List individuals that {person_name} definitively met during their lifetime. "
        f"Include well-known contemporaries, collaborators, mentors, protégés, or even notable adversaries they directly encountered. "
        f"Prioritize interactions with a higher degree of certainty or historical record. "
        f"Provide as many names as you can confidently list. "
        f"Return ONLY the full names, separated by commas. For example: Isaac Newton, Robert Boyle, John Locke, Gottfried Wilhelm Leibniz. "
        f"Do NOT include any introductory phrases, explanations, or any text other than the comma-separated list of names."
    )
    
    for attempt in range(MAX_RETRIES_API_CALL):
        current_prompt = base_prompt_text
        if attempt > 0 :
             current_prompt += "\nIMPORTANT: Your response MUST be ONLY a comma-separated list of names. No other text."

        print(f"   Querying Gemini for: {person_name} (Attempt {attempt + 1}/{MAX_RETRIES_API_CALL}, Prompt: {prompt_version})...")
        try:
            response = gemini_model.generate_content(current_prompt)
            
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"      LLM Query blocked: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}")
                llm_cache[cache_key] = []; save_cache(llm_cache); return []
                
            if not response.candidates or not response.candidates[0].content.parts:
                print(f"      LLM returned no candidates/empty content on attempt {attempt + 1}.")
                if attempt + 1 < MAX_RETRIES_API_CALL: time.sleep(API_CALL_DELAY_SECONDS); continue
                else: break 

            text_response = response.text.strip()
            print(f"      LLM Raw Response (attempt {attempt+1}, first 200 chars): {text_response[:200]}...")

            if not is_valid_name_list_format(text_response):
                print(f"      Response format invalid on attempt {attempt + 1}. Response: '{text_response[:100]}...'")
                if attempt + 1 < MAX_RETRIES_API_CALL:
                    print(f"      Retrying with stricter format prompt...")
                    time.sleep(API_CALL_DELAY_SECONDS / 2) 
                    continue
                else:
                    print(f"      Max retries for format correction reached. Giving up.")
                    llm_cache[cache_key] = []; save_cache(llm_cache); return [] 

            met_people_names = [name.strip() for name in text_response.split(',') if name.strip() and name.strip().lower() not in ["none", "n/a", "unknown", "various", "multiple", "no one", "no known meetings"]]
            llm_cache[cache_key] = met_people_names
            save_cache(llm_cache)
            return met_people_names

        except Exception as e:
            error_message = str(e); print(f"      Error during Gemini API call attempt {attempt + 1}/{MAX_RETRIES_API_CALL}: {error_message}")
            retry_after = API_CALL_DELAY_SECONDS
            if "429" in error_message or "ResourceExhausted" in error_message or "rate limit" in error_message.lower():
                match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_message, re.IGNORECASE | re.DOTALL)
                extracted_delay_server = int(match.group(1)) if match else None
                if extracted_delay_server: retry_after = max(extracted_delay_server, 5)
                else: retry_after = (API_CALL_DELAY_SECONDS * (2**attempt)) + random.uniform(0,1)
                print(f"      Rate limit or resource exhausted. Retrying in {retry_after:.2f}s...")
            elif attempt + 1 == MAX_RETRIES_API_CALL:
                print(f"      Max API retries reached after error. Skipping."); break
            else:
                retry_after = (API_CALL_DELAY_SECONDS / 2 * (2**attempt)) + random.uniform(0,0.5)
                print(f"      Retrying in {retry_after:.2f}s...")
            time.sleep(retry_after)

    llm_cache[cache_key] = []; save_cache(llm_cache); return []


def get_people_met_from_ollama(person_name, birth_year, death_year, occupation, alive_status, ollama_model_name):
    global llm_cache
    if not ollama:
        print("   ERROR: ollama library not installed. Skipping local LLM call.")
        return []

    prompt_version = "ollama_v3.1_fmt_strict"
    cache_key = f"{person_name}_{birth_year}_{occupation}_{alive_status}_{ollama_model_name}_{prompt_version}"

    if cache_key in llm_cache:
        print(f"   Cache hit for: {person_name} (Ollama: {ollama_model_name}, Prompt: {prompt_version})")
        return llm_cache[cache_key]

    life_span = f"Born: {birth_year if pd.notna(birth_year) else 'Unknown'}"
    current_alive_status_text = ""
    if alive_status is pd.NA: current_alive_status = False
    elif isinstance(alive_status, bool): current_alive_status = alive_status
    else: current_alive_status = str(alive_status).upper() == 'TRUE'

    if current_alive_status: life_span += " (Alive)"; current_alive_status_text = " (currently alive)"
    elif pd.notna(death_year): life_span += f", Died: {death_year}"
    else: life_span += ", Death year unknown"

    base_prompt_text = (
        f"Consider the historical or public figure: {person_name} "
        f"(Occupation: {occupation if pd.notna(occupation) else 'Unknown'}, {life_span}{current_alive_status_text}). "
        f"List individuals that {person_name} definitively met during their lifetime. "
        f"Include well-known contemporaries, collaborators, mentors, protégés, or even notable adversaries they directly encountered. "
        f"Only give interactions with a high degree of certainty or with records. "
        f"Provide as many names as you can confidently list. "
        f"Return ONLY the full names, separated by commas. For example: Isaac Newton, Robert Boyle, John Locke, Gottfried Wilhelm Leibniz. "
        f"Do NOT include any introductory phrases, explanations, or any text other than the comma-separated list of names. Your entire response should just be the list."
    )

    for attempt in range(MAX_FORMAT_CORRECTION_ATTEMPTS):
        current_prompt = base_prompt_text
        if attempt > 0:
            current_prompt += "\n\nIMPORTANT REMINDER: Your entire response must be ONLY a comma-separated list of names. No other text. Example: Name One, Name Two, Name Three"
            print(f"   Re-querying local Ollama with stricter format prompt (Attempt {attempt + 1}/{MAX_FORMAT_CORRECTION_ATTEMPTS})...")
        else:
            print(f"   Querying local Ollama ({ollama_model_name}) for: {person_name} (Prompt: {prompt_version})...")

        try:
            response = ollama.generate(model=ollama_model_name, prompt=current_prompt, stream=False,
                                       options={"temperature": 0.05, "num_predict": 350})
            text_response_raw = response['response'].strip()
            print(f"      Ollama Raw Response (attempt {attempt+1}, first 200): {text_response_raw[:200]}...")

            if not is_valid_name_list_format(text_response_raw):
                print(f"      Ollama response format invalid on attempt {attempt + 1}. Response: '{text_response_raw[:100]}...'")
                if attempt + 1 < MAX_FORMAT_CORRECTION_ATTEMPTS:
                    time.sleep(0.5)
                    continue
                else:
                    print(f"      Max format correction attempts reached for Ollama. Giving up on this person.")
                    llm_cache[cache_key] = []; save_cache(llm_cache); return []

            match = re.match(r"^(?:here(?: is|'s) a list of people .*? met:|based on .*?, .*? met:|individuals .*? met:|people .*? met:|.*?:\s*)?(.*)", text_response_raw, re.IGNORECASE | re.DOTALL)
            actual_list_part = match.group(1).strip() if match and match.group(1) else text_response_raw

            met_people_names_raw_split = actual_list_part.split(',')
            met_people_names = []
            for name_candidate in met_people_names_raw_split:
                name = name_candidate.strip()
                name = re.sub(r'\s*\(.*?\)\s*$', '', name).strip()
                name = name.strip('\'"')

                if name and name.lower() not in ["none", "n/a", "unknown", "various", "multiple", "several", "", "no one", "no known meetings"] and len(name) > 2 :
                    met_people_names.append(name)

            met_people_names = [name for name in met_people_names if name.lower() != person_name.lower()]
            print(f"      Processed list (first 5): {met_people_names[:5]}")
            llm_cache[cache_key] = met_people_names; save_cache(llm_cache); return met_people_names

        except Exception as e:
            print(f"      Error querying local Ollama ({ollama_model_name}) for {person_name} on attempt {attempt + 1}: {e}")
            llm_cache[cache_key] = []; save_cache(llm_cache); return []

    llm_cache[cache_key] = []; save_cache(llm_cache); return []


# --- Helper for describing the processing set ---
def get_processing_set_description(master_list_size: int, processing_list_size: int) -> str:
    if processing_list_size < master_list_size:
        return f"top {processing_list_size} (from {master_list_size} total filtered & sorted)"
    else:
        return f"all {master_list_size} filtered & sorted"

# --- Main Script ---
def create_meeting_graph(use_local_llm: bool, local_llm_model: str, limit_top_n: int = None):
    global llm_cache, current_graph_object, MAX_PEOPLE_TO_PROCESS_IN_SESSION, graph_state_initialized # MODIFIED

    signal.signal(signal.SIGINT, save_graph_on_interrupt)
    llm_cache = load_cache()

    if not use_local_llm:
        if not initialize_gemini():
            print("Exiting due to Gemini initialization failure.")
            # graph_state_initialized remains False, interrupt will not save
            return
    elif not ollama:
        print("ERROR: --local flag used, but 'ollama' library not installed. Please install it (`pip install ollama`).")
        # graph_state_initialized remains False, interrupt will not save
        return
    else:
        print(f"Using local Ollama model: {local_llm_model}. Make sure Ollama server is running and model '{local_llm_model}' is pulled.")

    print(f"Loading data from {CSV_FILE_PATH}...")
    try:
        dtypes_map = {
            'name': str, 'occupation': str,
            'birthyear': 'Int64', 'deathyear': 'Int64',
            'alive': 'boolean', 'non_en_page_views': 'Int64'
        }
        df_full_raw = pd.read_csv(CSV_FILE_PATH, low_memory=False)

        for col, dtype_val in dtypes_map.items():
            if col in df_full_raw.columns:
                if dtype_val == 'Int64':
                    df_full_raw[col] = pd.to_numeric(df_full_raw[col], errors='coerce').astype(pd.Int64Dtype())
                elif dtype_val == 'boolean':
                    df_full_raw[col] = df_full_raw[col].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False}).astype(pd.BooleanDtype())
                else:
                    df_full_raw[col] = df_full_raw[col].astype(str)
            elif dtype_val in ['Int64', 'boolean', str]:
                if dtype_val == 'Int64': df_full_raw[col] = pd.Series(pd.NA, index=df_full_raw.index, dtype=pd.Int64Dtype())
                elif dtype_val == 'boolean': df_full_raw[col] = pd.Series(pd.NA, index=df_full_raw.index, dtype=pd.BooleanDtype())
                else: df_full_raw[col] = pd.Series(pd.NA, index=df_full_raw.index, dtype=str)

        df_full_raw['occupation'] = df_full_raw['occupation'].fillna('Unknown').astype(str)
        df_full_raw['name'] = df_full_raw['name'].fillna('').astype(str).str.strip()
        df_full_raw['non_en_page_views'] = pd.to_numeric(df_full_raw.get('non_en_page_views'), errors='coerce').fillna(0).astype(int)

        print(f"Original dataset size: {len(df_full_raw)} people.")
        df_full_raw['birthyear'] = pd.to_numeric(df_full_raw['birthyear'], errors='coerce')
        df_filtered = df_full_raw[df_full_raw['birthyear'].notna() & (df_full_raw['birthyear'] >= BIRTHYEAR_LOWER_BOUND)].copy()
        df_filtered['birthyear'] = df_filtered['birthyear'].astype(pd.Int64Dtype())

        print(f"Filtered by birthyear (>= {BIRTHYEAR_LOWER_BOUND} and not NA): {len(df_filtered)} people.")
        df_sorted = df_filtered.sort_values(by='non_en_page_views', ascending=False).copy()
        print(f"Sorted by page views (descending).")

        df_master_list = df_sorted.reset_index(drop=True)

        if df_master_list.empty:
            print(f"No people found after filtering and sorting. Exiting.")
            # graph_state_initialized remains False, interrupt will not save
            return

        all_known_names_full_dataset = list(df_master_list['name'][df_master_list['name'] != ''].unique())
        name_to_data_map_full_dataset = {row['name']: row.to_dict() for _, row in df_master_list.iterrows() if row['name'] != ''}
        print(f"LLM results will be matched against {len(all_known_names_full_dataset)} unique names from the full filtered & sorted dataset.")

        df_for_processing = df_master_list

        if limit_top_n is not None and limit_top_n > 0:
            if limit_top_n < len(df_master_list):
                print(f"Applying --limit: will process only the top {limit_top_n} people from the {len(df_master_list)} in the master list.")
                df_for_processing = df_master_list.head(limit_top_n).copy()
            else:
                print(f"--limit {limit_top_n} is >= master list size ({len(df_master_list)}). Will process all {len(df_master_list)} available.")
        elif limit_top_n is not None and limit_top_n <= 0:
            print("Warning: --limit value must be positive. Ignoring --limit, will process full master list.")

        df_current_scope = df_for_processing.reset_index(drop=True)
        processing_set_desc = get_processing_set_description(len(df_master_list), len(df_current_scope))

        if df_current_scope.empty:
            print(f"No people to process in the current scope ({processing_set_desc}). Exiting.")
            # graph_state_initialized remains False, interrupt will not save
            return

    except FileNotFoundError:
        print(f"Error: {CSV_FILE_PATH} not found.")
        # graph_state_initialized remains False, interrupt will not save
        return
    except Exception as e:
        print(f"Error loading/filtering CSV: {e}")
        # graph_state_initialized remains False, interrupt will not save
        return

    # --- Graph Loading/Initialization ---
    # current_graph_object is already global and initialized to nx.Graph()
    if os.path.exists(OUTPUT_GRAPH_FILE):
        try:
            print(f"Loading existing graph from {OUTPUT_GRAPH_FILE}...")
            current_graph_object = nx.read_gexf(OUTPUT_GRAPH_FILE) # This re-assigns the global
            print(f"Loaded graph: {current_graph_object.number_of_nodes()} nodes, {current_graph_object.number_of_edges()} edges.")
        except Exception as e:
            print(f"Could not load existing graph: {e}. Starting new graph.")
            current_graph_object = nx.Graph() # Ensure it's a new graph if loading fails
    else:
        print(f"No existing graph file found at {OUTPUT_GRAPH_FILE}. Starting new graph.")
        current_graph_object = nx.Graph() # Ensure it's a new graph if no file exists

    graph_state_initialized = True # NEW: Set flag only AFTER graph object is definitively set

    start_index = load_start_index()
    if start_index > 0 and start_index < len(df_current_scope):
        print(f"Resuming from index {start_index} of the {processing_set_desc} processing set.")
    elif start_index >= len(df_current_scope) and len(df_current_scope) > 0:
        print(f"All {len(df_current_scope)} people from the {processing_set_desc} processing set already processed according to progress file. Exiting.")
        return # graph_state_initialized is True, but main processing loop won't run.
    elif start_index > 0 and len(df_current_scope) == 0 :
        print(f"Progress file indicates a start index, but the {processing_set_desc} processing set is empty. Resetting progress.")
        start_index = 0; save_progress(0)

    if MAX_PEOPLE_TO_PROCESS_IN_SESSION is not None:
        end_slice_index = min(start_index + MAX_PEOPLE_TO_PROCESS_IN_SESSION, len(df_current_scope))
        people_to_process_df_slice = df_current_scope.iloc[start_index : end_slice_index]
    else:
        people_to_process_df_slice = df_current_scope.iloc[start_index:]

    total_to_process_this_session = len(people_to_process_df_slice)
    if total_to_process_this_session == 0:
        if start_index >= len(df_current_scope) and len(df_current_scope) > 0 :
            print(f"All people from the {processing_set_desc} processing set processed based on start_index.")
        else:
            print(f"No people to process this session (check {processing_set_desc} set, and MAX_PEOPLE_TO_PROCESS_IN_SESSION).")
        # No processing will occur, so we might just want to save if the graph was loaded.
        # However, the main save logic is at the end of the function, after the loop.
        # If we exit here, the graph (if loaded) won't be re-saved unless Ctrl+C was hit.
        # This is probably fine, as no changes were made.
        return

    print(f"\nProcessing up to {total_to_process_this_session} people this session (from index {start_index} to {start_index + total_to_process_this_session -1} of {processing_set_desc} set)...")

    processed_count_in_session = 0
    session_start_time = time.time()

    for current_scope_idx, row_series in people_to_process_df_slice.iterrows():
        eta_str = "Calculating..."
        avg_time_str = "N/A"

        if processed_count_in_session > 0:
            elapsed_session_time = time.time() - session_start_time
            avg_time_per_item = elapsed_session_time / processed_count_in_session
            avg_time_str = f"{avg_time_per_item:.2f}s/item"
            items_remaining_in_session = total_to_process_this_session - processed_count_in_session
            if items_remaining_in_session > 0:
                eta_seconds = avg_time_per_item * items_remaining_in_session
                eta_str = format_time_delta(eta_seconds)
            else:
                eta_str = "Finalizing..."
        elif total_to_process_this_session > 0 :
             eta_str = f"Estimating for {total_to_process_this_session} items this session..."

        row = row_series.to_dict()
        person_a_name = str(row.get('name', '')).strip()

        print(f"\nProcessing item {processed_count_in_session + 1}/{total_to_process_this_session} (Index in current processing set: {current_scope_idx}, Name: {person_a_name})")
        print(f"   Session Stats: Avg: {avg_time_str}, ETA (session): {eta_str} (PageViews: {row.get('non_en_page_views', 'N/A')})")

        item_work_start_time = time.time()

        if not person_a_name:
            print(f"Skipping row with empty name at index {current_scope_idx} in current processing set.")
            item_work_duration = time.time() - item_work_start_time
            print(f"   Item processing (skip) took: {item_work_duration:.2f}s.")
            processed_count_in_session += 1
            save_progress(current_scope_idx + 1)
            continue

        birth_year = row.get('birthyear', pd.NA)
        death_year = row.get('deathyear', pd.NA)
        occupation = row.get('occupation', 'Unknown')
        alive_status = row.get('alive', pd.NA)

        if not current_graph_object.has_node(person_a_name):
            node_attrs = {}
            for k, v in row.items():
                if pd.isna(v): node_attrs[k] = "NA"
                elif isinstance(v, (pd.Timestamp, pd.Timedelta)): node_attrs[k] = str(v)
                elif isinstance(v, (bool, pd.BooleanDtype)): node_attrs[k] = str(v)
                else: node_attrs[k] = v
            current_graph_object.add_node(person_a_name, **node_attrs)

        if use_local_llm:
            potential_met_names_llm = get_people_met_from_ollama(
                person_a_name, birth_year, death_year, occupation, alive_status, local_llm_model
            )
        else:
            potential_met_names_llm = get_people_met_from_gemini(
                person_a_name, birth_year, death_year, occupation, alive_status
            )

        for llm_name in potential_met_names_llm:
            matched_name_from_dataset = None
            if llm_name in name_to_data_map_full_dataset:
                matched_name_from_dataset = llm_name
            elif all_known_names_full_dataset:
                fuzzy_match_result = process.extractOne(llm_name, all_known_names_full_dataset, scorer=fuzz.WRatio, score_cutoff=FUZZY_MATCH_THRESHOLD)
                if fuzzy_match_result:
                    matched_name_from_dataset = fuzzy_match_result[0]
                    print(f"      Fuzzy match (against full dataset): LLM '{llm_name}' -> Dataset '{matched_name_from_dataset}' (Score: {fuzzy_match_result[1]})")

            if matched_name_from_dataset:
                if person_a_name != matched_name_from_dataset:
                    if not current_graph_object.has_node(matched_name_from_dataset):
                        person_b_data = name_to_data_map_full_dataset.get(matched_name_from_dataset)
                        if person_b_data:
                            node_attrs_b = {}
                            for k, v_b in person_b_data.items():
                                if pd.isna(v_b): node_attrs_b[k] = "NA"
                                elif isinstance(v_b, (pd.Timestamp, pd.Timedelta)): node_attrs_b[k] = str(v_b)
                                elif isinstance(v_b, (bool, pd.BooleanDtype)): node_attrs_b[k] = str(v_b)
                                else: node_attrs_b[k] = v_b
                            current_graph_object.add_node(matched_name_from_dataset, **node_attrs_b)
                        else:
                            current_graph_object.add_node(matched_name_from_dataset, label=matched_name_from_dataset)

                    if current_graph_object.has_node(matched_name_from_dataset): # Check again, could have been added above
                        if not current_graph_object.has_edge(person_a_name, matched_name_from_dataset):
                            current_graph_object.add_edge(person_a_name, matched_name_from_dataset)
                            print(f"      Added edge: {person_a_name} <-> {matched_name_from_dataset}")
            elif llm_name: # Only print if llm_name is not empty/None
                print(f"      Note: LLM suggested '{llm_name}', but no close match found/valid in full dataset.")

        item_work_duration = time.time() - item_work_start_time
        processed_count_in_session += 1
        save_progress(current_scope_idx + 1)

        if not use_local_llm and processed_count_in_session < total_to_process_this_session :
            print(f"   Item core work took: {item_work_duration:.2f}s. Waiting {API_CALL_DELAY_SECONDS}s for next API call...")
            time.sleep(API_CALL_DELAY_SECONDS)
        else:
             print(f"   Item core work took: {item_work_duration:.2f}s.")

    total_session_duration = time.time() - session_start_time
    print("\n--- Graph Construction Session Complete ---")
    print(f"Processed {processed_count_in_session} items in this session.")
    print(f"Total session duration: {format_time_delta(total_session_duration)}")
    if processed_count_in_session > 0:
        avg_total_time_per_item = total_session_duration / processed_count_in_session
        print(f"Average time per item (incl. delays): {avg_total_time_per_item:.2f}s")

    print(f"Final graph state: {current_graph_object.number_of_nodes()} nodes, {current_graph_object.number_of_edges()} edges.")
    
    # Ensure graph_state_initialized is checked before saving at the end of normal execution
    if graph_state_initialized:
        if current_graph_object.number_of_nodes() > 0:
            try:
                nx.write_gexf(current_graph_object, OUTPUT_GRAPH_FILE)
                print(f"Graph saved to {OUTPUT_GRAPH_FILE} (Nodes: {current_graph_object.number_of_nodes()}, Edges: {current_graph_object.number_of_edges()})")
            except Exception as e: print(f"Error writing GEXF file: {e}")
        else:
            print("Graph is empty at the end of the session. Not saving an empty graph.")
            # Consider if an empty OUTPUT_GRAPH_FILE should be deleted if it existed
            # For now, it just won't overwrite a potentially non-empty file with an empty one.
    else:
        # This case should ideally not be reached if the script ran through,
        # as graph_state_initialized would have been set.
        # It's a safeguard.
        print("Graph state was not properly initialized during the session; not saving at script completion.")


    final_processed_index_in_current_scope = start_index + processed_count_in_session
    if final_processed_index_in_current_scope >= len(df_current_scope):
        print(f"\nAll people in the {processing_set_desc} processing set have been processed.")
        if os.path.exists(PROGRESS_FILE):
            try: os.remove(PROGRESS_FILE); print(f"Progress file {PROGRESS_FILE} removed as processing for this scope ({processing_set_desc}) is complete.")
            except OSError as e: print(f"Could not remove progress file {PROGRESS_FILE}: {e}")
    else:
        print(f"\nProcessing paused. Run script again to continue from index {final_processed_index_in_current_scope} of the {processing_set_desc} processing set.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a graph of people who may have met, using Gemini or a local LLM.")
    parser.add_argument("--local", action="store_true", help="Use a local LLM via Ollama instead of Gemini API.")
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help=f"Specify the local Ollama model name (default: {DEFAULT_OLLAMA_MODEL}). Used only with --local.")
    parser.add_argument("--max_process", type=int, default=None, help="Maximum number of people to process in this session (overrides script's MAX_PEOPLE_TO_PROCESS_IN_SESSION).")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the top N most popular people from the filtered and sorted list. This defines the set of people to query LLM for.")

    args = parser.parse_args()

    if args.max_process is not None:
        if args.max_process <= 0:
            print("Warning: --max_process must be a positive integer. Using script default or no limit for session batching.")
        else:
            MAX_PEOPLE_TO_PROCESS_IN_SESSION = args.max_process
            print(f"Command-line override: MAX_PEOPLE_TO_PROCESS_IN_SESSION (session batch size) set to {MAX_PEOPLE_TO_PROCESS_IN_SESSION}")

    limit_top_n_val = None
    if args.limit is not None:
        if args.limit <= 0:
            print("Warning: --limit value must be a positive integer. Ignoring --limit.")
        else:
            limit_top_n_val = args.limit
            print(f"Command-line setting: --limit (people to query LLM for) to top {limit_top_n_val} people.")

    create_meeting_graph(use_local_llm=args.local, local_llm_model=args.model, limit_top_n=limit_top_n_val)