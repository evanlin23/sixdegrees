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

MAX_PEOPLE_TO_PROCESS_IN_SESSION = None # Set to a number to limit, e.g., 10, or None for all. Can be overridden by CLI.
API_CALL_DELAY_SECONDS = 4 
MAX_RETRIES_API_CALL = 5 # For Gemini API
FUZZY_MATCH_THRESHOLD = 95
BIRTHYEAR_LOWER_BOUND = 1800

DEFAULT_OLLAMA_MODEL = "phi3:mini" # Model to use with --local if not specified

# --- Global graph object for signal handler ---
current_graph_object = nx.Graph()

# --- Helper for ETA and time formatting ---
def format_time_delta(seconds):
    if seconds is None or seconds < 0:
        return "N/A"
    seconds = int(seconds) # Ensure integer seconds for calculations
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
        except FileNotFoundError: pass # Should not happen if os.path.exists is true, but defensive
        except Exception as e: print(f"Warning: Could not load progress file {PROGRESS_FILE}: {e}. Starting from scratch.")
    return 0

def save_progress(index):
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f: f.write(str(index))
    except Exception as e: print(f"Error saving progress to {PROGRESS_FILE}: {e}")

# --- Signal Handler for Ctrl+C ---
def save_graph_on_interrupt(signum, frame):
    global current_graph_object
    print("\nCtrl+C detected. Attempting to save current graph state...")
    if current_graph_object is not None and current_graph_object.number_of_nodes() > 0:
        try:
            nx.write_gexf(current_graph_object, OUTPUT_GRAPH_FILE)
            print(f"Partial graph saved to {OUTPUT_GRAPH_FILE}")
        except Exception as e: print(f"Error saving graph on interrupt: {e}")
    else: print("Graph is empty or not initialized, not saving on interrupt.")
    print("Exiting script."); exit(0)

# --- LLM Query Functions ---
llm_cache = {} # Loaded in main function
gemini_model = None # Initialized if Gemini is used

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

def get_people_met_from_gemini(person_name, birth_year, death_year, occupation, alive_status):
    global llm_cache, gemini_model
    if not gemini_model:
        print("   ERROR: Gemini model not initialized. Skipping API call.")
        return []

    prompt_version = "gemini_v2_more_people_confident" 
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

    prompt = (
        f"Consider the historical or public figure: {person_name} "
        f"(Occupation: {occupation if pd.notna(occupation) else 'Unknown'}, {life_span}{current_alive_status_text}). "
        f"List individuals that {person_name} definitively met during their lifetime. "
        f"Include well-known contemporaries, collaborators, mentors, protégés, or even notable adversaries they directly encountered. "
        f"Prioritize interactions with a higher degree of certainty or historical record. "
        f"Provide as many names as you can confidently list. "
        f"Return ONLY the full names, separated by commas. For example: Isaac Newton, Robert Boyle, John Locke, Gottfried Wilhelm Leibniz"
    )
    print(f"   Querying Gemini for: {person_name} (Prompt: {prompt_version})...")
    
    for attempt in range(MAX_RETRIES_API_CALL):
        try:
            response = gemini_model.generate_content(prompt)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"      LLM Query blocked: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"); break
            if not response.candidates or not response.candidates[0].content.parts:
                print(f"      LLM returned no candidates/empty content."); break
            text_response = response.text.strip()
            print(f"      LLM Response (first 200 chars): {text_response[:200]}...")
            met_people_names = [name.strip() for name in text_response.split(',') if name.strip() and name.strip().lower() not in ["none", "n/a", "unknown", "various", "multiple"]]
            llm_cache[cache_key] = met_people_names; save_cache(llm_cache); return met_people_names
        except Exception as e:
            error_message = str(e); print(f"      Error attempt {attempt + 1}/{MAX_RETRIES_API_CALL}: {error_message}")
            retry_after = API_CALL_DELAY_SECONDS 
            if "429" in error_message or "ResourceExhausted" in error_message or "rate limit" in error_message.lower():
                match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_message, re.IGNORECASE | re.DOTALL)
                # Use server-suggested delay if available, else exponential backoff
                extracted_delay_server = int(match.group(1)) if match else None
                if extracted_delay_server:
                    retry_after = max(extracted_delay_server, 5) # Ensure minimum 5s
                else:
                    retry_after = (API_CALL_DELAY_SECONDS * (2**attempt)) + random.uniform(0,1) # Exponential backoff
                print(f"      Rate limit or resource exhausted. Retrying in {retry_after:.2f}s...")
            elif attempt + 1 == MAX_RETRIES_API_CALL: print(f"      Max retries. Skipping."); break
            else: # General error, apply milder exponential backoff
                retry_after = (API_CALL_DELAY_SECONDS / 2 * (2**attempt)) + random.uniform(0,0.5) # Shorter base for general errors
                print(f"      Retrying in {retry_after:.2f}s...")
            time.sleep(retry_after)
    llm_cache[cache_key] = []; save_cache(llm_cache); return []


def get_people_met_from_ollama(person_name, birth_year, death_year, occupation, alive_status, ollama_model_name):
    global llm_cache
    if not ollama:
        print("   ERROR: ollama library not installed. Skipping local LLM call.")
        return []

    prompt_version = "ollama_v3_max_confident_cleaned" 
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

    prompt = (
        f"Consider the historical or public figure: {person_name} "
        f"(Occupation: {occupation if pd.notna(occupation) else 'Unknown'}, {life_span}{current_alive_status_text}). "
        f"List individuals that {person_name} definitively met during their lifetime. "
        f"Include well-known contemporaries, collaborators, mentors, protégés, or even notable adversaries they directly encountered. "
        f"Only give interactions with a high degree of certainty or with records. "
        f"Provide as many names as you can confidently list. "
        f"Return ONLY the full names, separated by commas. For example: Isaac Newton, Robert Boyle, John Locke, Gottfried Wilhelm Leibniz. Do not include any introductory phrases or explanations."
    )
    print(f"   Querying local Ollama ({ollama_model_name}) for: {person_name} (Prompt: {prompt_version})...")
    
    try:
        response = ollama.generate(model=ollama_model_name, prompt=prompt, stream=False,
                                   options={"temperature": 0.1, "num_predict": 350}) # num_predict might need adjustment per model
        text_response = response['response'].strip()
        print(f"      Ollama Raw Response (first 200): {text_response[:200]}...")
        
        # Attempt to remove common introductory phrases if model still adds them
        # This regex tries to find the start of the list after potential intro
        match = re.match(r"^(?:here(?: is|'s) a list of people .*? met:|based on .*?, .*? met:|individuals .*? met:|people .*? met:|.*?:\s*)?(.*)", text_response, re.IGNORECASE | re.DOTALL)
        if match and match.group(1):
            actual_list_part = match.group(1).strip()
        else:
            actual_list_part = text_response # Fallback to full response if regex doesn't find a clear list start

        met_people_names_raw = actual_list_part.split(',')
        met_people_names = []
        for name_candidate in met_people_names_raw:
            name = name_candidate.strip()
            # Remove trailing explanations like "(collaborator)" or "(met in Paris)"
            name = re.sub(r'\s*\(.*?\)\s*$', '', name).strip()
            # Remove leading/trailing quotes if any
            name = name.strip('\'"')

            if name and name.lower() not in ["none", "n/a", "unknown", "various", "multiple", "several", ""] and len(name) > 2:
                met_people_names.append(name)
        
        met_people_names = [name for name in met_people_names if name.lower() != person_name.lower()] # Remove self-mentions
        print(f"      Processed list (first 5): {met_people_names[:5]}")
        llm_cache[cache_key] = met_people_names; save_cache(llm_cache); return met_people_names
    except Exception as e:
        print(f"      Error querying local Ollama ({ollama_model_name}) for {person_name}: {e}")
        llm_cache[cache_key] = []; save_cache(llm_cache); return []


# --- Main Script ---
def create_meeting_graph(use_local_llm: bool, local_llm_model: str):
    global llm_cache, current_graph_object, MAX_PEOPLE_TO_PROCESS_IN_SESSION # Allow modification by CLI args
    
    signal.signal(signal.SIGINT, save_graph_on_interrupt)
    llm_cache = load_cache()

    if not use_local_llm:
        if not initialize_gemini():
            print("Exiting due to Gemini initialization failure.")
            return
    elif not ollama:
        print("ERROR: --local flag used, but 'ollama' library not installed. Please install it (`pip install ollama`).")
        return
    else:
        print(f"Using local Ollama model: {local_llm_model}. Make sure Ollama server is running and model '{local_llm_model}' is pulled.")


    print(f"Loading data from {CSV_FILE_PATH}...")
    try:
        # Define dtypes for critical columns to ensure correct parsing
        dtypes_map = {
            'name': str, 'occupation': str, 
            'birthyear': 'Int64', 'deathyear': 'Int64', 
            'alive': 'boolean', 'non_en_page_views': 'Int64'
        }
        df_full = pd.read_csv(CSV_FILE_PATH, low_memory=False) # Read all first
        
        for col, dtype_val in dtypes_map.items():
            if col in df_full.columns:
                if dtype_val == 'Int64':
                    df_full[col] = pd.to_numeric(df_full[col], errors='coerce').astype(pd.Int64Dtype())
                elif dtype_val == 'boolean':
                    # Handle various string representations of boolean
                    df_full[col] = df_full[col].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False}).astype(pd.BooleanDtype())
                else:
                    df_full[col] = df_full[col].astype(str)
            elif dtype_val in ['Int64', 'boolean', str]: # If column missing, add it with appropriate NA type
                if dtype_val == 'Int64': df_full[col] = pd.Series(pd.NA, index=df_full.index, dtype=pd.Int64Dtype())
                elif dtype_val == 'boolean': df_full[col] = pd.Series(pd.NA, index=df_full.index, dtype=pd.BooleanDtype())
                else: df_full[col] = pd.Series(pd.NA, index=df_full.index, dtype=str)


        df_full['occupation'] = df_full['occupation'].fillna('Unknown').astype(str)
        df_full['name'] = df_full['name'].fillna('').astype(str).str.strip()
        df_full['non_en_page_views'] = pd.to_numeric(df_full.get('non_en_page_views'), errors='coerce').fillna(0).astype(int) # Ensure it's int after fillna

        print(f"Original dataset size: {len(df_full)} people.")
        df_filtered = df_full[df_full['birthyear'].notna() & (df_full['birthyear'] >= BIRTHYEAR_LOWER_BOUND)].copy()
        print(f"Filtered by birthyear (>= {BIRTHYEAR_LOWER_BOUND} and not NA): {len(df_filtered)} people.")
        df_sorted = df_filtered.sort_values(by='non_en_page_views', ascending=False).copy()
        print(f"Sorted by page views (descending).")
        df = df_sorted
        
        if df.empty: print(f"No people found matching filtering criteria. Exiting."); return
        df = df.reset_index(drop=True)

    except FileNotFoundError: print(f"Error: {CSV_FILE_PATH} not found."); return
    except Exception as e: print(f"Error loading/filtering CSV: {e}"); return

    all_known_names_in_dataset = list(df['name'][df['name'] != ''].unique())
    name_to_data_map = {row['name']: row.to_dict() for index, row in df.iterrows() if row['name'] != ''}
    print(f"Using {len(all_known_names_in_dataset)} unique names from filtered & sorted dataset for matching.")

    if os.path.exists(OUTPUT_GRAPH_FILE):
        try:
            print(f"Loading existing graph from {OUTPUT_GRAPH_FILE}...")
            current_graph_object = nx.read_gexf(OUTPUT_GRAPH_FILE)
            print(f"Loaded graph: {current_graph_object.number_of_nodes()} nodes, {current_graph_object.number_of_edges()} edges.")
        except Exception as e:
            print(f"Could not load existing graph: {e}. Starting new graph."); current_graph_object = nx.Graph()
    else: current_graph_object = nx.Graph()

    start_index = load_start_index()
    if start_index > 0 and start_index < len(df):
        print(f"Resuming from CSV index {start_index} of filtered & sorted dataset.")
    elif start_index >= len(df) and len(df) > 0:
        print(f"All {len(df)} people from filtered & sorted dataset already processed according to progress file. Exiting.")
        return
    elif start_index > 0 and len(df) == 0: 
        print("Progress file indicates a start index, but dataset is empty after filtering. Resetting progress.")
        start_index = 0; save_progress(0)


    if MAX_PEOPLE_TO_PROCESS_IN_SESSION is not None:
        end_slice_index = min(start_index + MAX_PEOPLE_TO_PROCESS_IN_SESSION, len(df))
        people_to_process_df = df.iloc[start_index : end_slice_index]
    else:
        people_to_process_df = df.iloc[start_index:]
    
    total_to_process_this_session = len(people_to_process_df)
    if total_to_process_this_session == 0:
        if start_index >= len(df) and len(df) > 0 : print("All people from filtered & sorted CSV processed based on start_index.")
        else: print("No people to process this session (check filters, CSV, and MAX_PEOPLE_TO_PROCESS_IN_SESSION).")
        return
        
    print(f"\nProcessing up to {total_to_process_this_session} people this session (from CSV index {start_index} to {start_index + total_to_process_this_session -1})...")

    processed_count_in_session = 0 
    session_start_time = time.time() 

    for original_df_idx, row_series in people_to_process_df.iterrows():
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
        
        print(f"\nProcessing item {processed_count_in_session + 1}/{total_to_process_this_session} (CSV Index: {original_df_idx}, Name: {person_a_name})")
        print(f"   Session Stats: Avg: {avg_time_str}, ETA (session): {eta_str} (PageViews: {row.get('non_en_page_views', 'N/A')})")
        
        item_work_start_time = time.time()

        if not person_a_name:
            print(f"Skipping row with empty name at CSV index {original_df_idx}.")
            item_work_duration = time.time() - item_work_start_time
            print(f"   Item processing (skip) took: {item_work_duration:.2f}s.")
            processed_count_in_session += 1 
            save_progress(original_df_idx + 1) 
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
                elif isinstance(v, (bool, pd.BooleanDtype)): node_attrs[k] = str(v) # GEXF prefers strings for bools often
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
            if llm_name in name_to_data_map: 
                matched_name_from_dataset = llm_name
            elif all_known_names_in_dataset: 
                fuzzy_match_result = process.extractOne(llm_name, all_known_names_in_dataset, scorer=fuzz.WRatio, score_cutoff=FUZZY_MATCH_THRESHOLD)
                if fuzzy_match_result: 
                    matched_name_from_dataset = fuzzy_match_result[0]
                    print(f"      Fuzzy match: LLM '{llm_name}' -> Dataset '{matched_name_from_dataset}' (Score: {fuzzy_match_result[1]})")
            
            if matched_name_from_dataset:
                if person_a_name != matched_name_from_dataset: 
                    if not current_graph_object.has_node(matched_name_from_dataset):
                        person_b_data = name_to_data_map.get(matched_name_from_dataset)
                        if person_b_data:
                            node_attrs_b = {}
                            for k, v_b in person_b_data.items():
                                if pd.isna(v_b): node_attrs_b[k] = "NA"
                                elif isinstance(v_b, (pd.Timestamp, pd.Timedelta)): node_attrs_b[k] = str(v_b)
                                elif isinstance(v_b, (bool, pd.BooleanDtype)): node_attrs_b[k] = str(v_b)
                                else: node_attrs_b[k] = v_b
                            current_graph_object.add_node(matched_name_from_dataset, **node_attrs_b)
                        else: 
                            current_graph_object.add_node(matched_name_from_dataset, label=matched_name_from_dataset) # Basic node
                    
                    if current_graph_object.has_node(matched_name_from_dataset): 
                        if not current_graph_object.has_edge(person_a_name, matched_name_from_dataset):
                            current_graph_object.add_edge(person_a_name, matched_name_from_dataset)
                            print(f"      Added edge: {person_a_name} <-> {matched_name_from_dataset}")
            elif llm_name: 
                print(f"      Note: LLM suggested '{llm_name}', but no close match found/valid in dataset.")
        
        item_work_duration = time.time() - item_work_start_time
        
        processed_count_in_session += 1
        save_progress(original_df_idx + 1) 

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

    print(f"Current graph: {current_graph_object.number_of_nodes()} nodes, {current_graph_object.number_of_edges()} edges.")
    if current_graph_object.number_of_nodes() > 0:
        try: 
            nx.write_gexf(current_graph_object, OUTPUT_GRAPH_FILE)
            print(f"Graph saved to {OUTPUT_GRAPH_FILE}")
        except Exception as e: print(f"Error writing GEXF file: {e}")
    else: print("Graph is empty, not saving.")
    
    final_processed_csv_index = start_index + processed_count_in_session
    if final_processed_csv_index >= len(df): 
        print("\nAll people in (filtered & sorted) CSV processed.")
        if os.path.exists(PROGRESS_FILE): 
            try: os.remove(PROGRESS_FILE); print(f"Progress file {PROGRESS_FILE} removed as processing is complete.")
            except OSError as e: print(f"Could not remove progress file {PROGRESS_FILE}: {e}")
    else: 
        print(f"\nProcessing paused. Run script again to continue from CSV index {final_processed_csv_index} of filtered & sorted dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a graph of people who may have met, using Gemini or a local LLM.")
    parser.add_argument("--local", action="store_true", help="Use a local LLM via Ollama instead of Gemini API.")
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help=f"Specify the local Ollama model name (default: {DEFAULT_OLLAMA_MODEL}). Used only with --local.")
    parser.add_argument("--max_process", type=int, default=None, help="Maximum number of people to process in this session (overrides script's MAX_PEOPLE_TO_PROCESS_IN_SESSION).")

    args = parser.parse_args()

    if args.max_process is not None:
        if args.max_process <= 0:
            print("Warning: --max_process must be a positive integer. Using script default or no limit.")
        else:
            MAX_PEOPLE_TO_PROCESS_IN_SESSION = args.max_process
            print(f"Command-line override: MAX_PEOPLE_TO_PROCESS_IN_SESSION set to {MAX_PEOPLE_TO_PROCESS_IN_SESSION}")


    create_meeting_graph(use_local_llm=args.local, local_llm_model=args.model)