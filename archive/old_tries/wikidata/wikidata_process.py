import networkx as nx
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import time
import itertools # For creating pairs from a list

# --- Configuration ---
# Path to your downloaded IMDb dataset files
IMDB_NAME_BASICS_PATH = "./name.basics.tsv.gz" # CHANGE THIS
IMDB_TITLE_PRINCIPALS_PATH = "./title.principals.tsv.gz" # CHANGE THIS

# --- Step 1: Setup ---
G = nx.Graph() # Undirected graph, as "meeting" is usually reciprocal

def add_connection(person1_name, person2_name, source, details=""):
    """Adds a connection to the graph if names are valid."""
    if person1_name and person2_name and person1_name != person2_name:
        p1 = person1_name.strip().lower()
        p2 = person2_name.strip().lower()
        if p1 != p2:  # double-check after lowercasing
            G.add_node(p1, source_added_by=source)
            G.add_node(p2, source_added_by=source)
            G.add_edge(p1, p2, source=source, details=details)

# --- Step 2: Wikidata Integration ---
def fetch_wikidata_connections():
    print("Fetching data from Wikidata...")
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    # Be polite to Wikidata's endpoint
    sparql.agent = "NetworkXFamousPeopleGraph/1.0 (https://example.com/myproject; myemail@example.com) SPARQLWrapper/1.8.5"


    # Query 1: People participating in the same event (e.g., award ceremony, summit)
    # We'll look for instances of 'event' (wd:Q1656682) or specific types like 'award ceremony' (wd:Q47253)
    # This query can be broad; refine event types (P31) for more specific meetings.
    query_event_participants = """
    SELECT ?person1Label ?person2Label ?eventLabel WHERE {
      # ?event wdt:P31/wdt:P279* wd:Q1656682. # instance of event (or subclass of) - can be too broad
      ?event wdt:P31 wd:Q47253 . # Example: Award ceremony
      ?event wdt:P710 ?person1; # P710 = participant
             wdt:P710 ?person2.
      FILTER(?person1 != ?person2).
      ?person1 rdfs:label ?person1Label FILTER(LANG(?person1Label) = "en").
      ?person2 rdfs:label ?person2Label FILTER(LANG(?person2Label) = "en").
      OPTIONAL { ?event rdfs:label ?eventLabel FILTER(LANG(?eventLabel) = "en"). }
    } LIMIT 200 # Adjust limit as needed, can be very large
    """

    # Query 2: People who were cast members (P161) in the same film (Q11424)
    query_film_cast = """
    SELECT ?actor1Label ?actor2Label ?filmLabel WHERE {
      ?film wdt:P31 wd:Q11424. # instance of film
      ?film wdt:P161 ?actor1;  # P161 = cast member
            wdt:P161 ?actor2.
      FILTER(?actor1 != ?actor2).
      ?actor1 rdfs:label ?actor1Label FILTER(LANG(?actor1Label) = "en").
      ?actor2 rdfs:label ?actor2Label FILTER(LANG(?actor2Label) = "en").
      OPTIONAL { ?film rdfs:label ?filmLabel FILTER(LANG(?filmLabel) = "en"). }
    } LIMIT 200 # Adjust limit
    """

    # Query 3: Spouses (P26) - very likely to have met
    query_spouses = """
    SELECT ?person1Label ?person2Label WHERE {
      ?person1 wdt:P26 ?person2. # P26 = spouse
      # Ensure both are humans (Q5) to avoid linking to organizations if data is messy
      ?person1 wdt:P31 wd:Q5.
      ?person2 wdt:P31 wd:Q5.
      ?person1 rdfs:label ?person1Label FILTER(LANG(?person1Label) = "en").
      ?person2 rdfs:label ?person2Label FILTER(LANG(?person2Label) = "en").
    } LIMIT 200 # Adjust limit
    """

    queries = {
        "wikidata_event": query_event_participants,
        "wikidata_film_cast": query_film_cast,
        "wikidata_spouses": query_spouses,
    }

    for source_detail, query_string in queries.items():
        print(f"  Executing Wikidata query for: {source_detail}")
        try:
            sparql.setQuery(query_string)
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                p1_label = result.get("person1Label", {}).get("value") or result.get("actor1Label", {}).get("value")
                p2_label = result.get("person2Label", {}).get("value") or result.get("actor2Label", {}).get("value")
                
                event_label = result.get("eventLabel", {}).get("value")
                film_label = result.get("filmLabel", {}).get("value")
                details = event_label or film_label or "Spousal relationship"

                if p1_label and p2_label:
                    add_connection(p1_label, p2_label, source_detail, details)
            time.sleep(1) # Be polite
        except Exception as e:
            print(f"  Error querying Wikidata for {source_detail}: {e}")
    print("Finished fetching data from Wikidata.")


# --- Step 3: IMDb Integration ---
def process_imdb_data():
    print("Processing IMDb data...")
    try:
        # Load name basics: nconst -> primaryName
        print(f"  Loading IMDb names from {IMDB_NAME_BASICS_PATH}...")
        name_basics_df = pd.read_csv(IMDB_NAME_BASICS_PATH, sep='\t', usecols=['nconst', 'primaryName'], low_memory=False)
        nconst_to_name = pd.Series(name_basics_df.primaryName.values, index=name_basics_df.nconst).to_dict()
        print(f"    Loaded {len(nconst_to_name)} names.")

        # Load title principals: tconst, nconst (people involved in a title)
        # We only care about actors, actresses, directors, writers for "meeting" likelihood.
        # You can expand `category` if needed.
        print(f"  Loading IMDb title principals from {IMDB_TITLE_PRINCIPALS_PATH}...")
        # Reading in chunks can help with memory for very large files, though title.principals is usually manageable
        chunk_size = 1000000  # Process 1 million rows at a time
        principals_by_title = defaultdict(list)

        # Usecols to load only necessary columns
        # 'category' can be used to filter e.g. 'actor', 'actress', 'director'
        for chunk_df in pd.read_csv(IMDB_TITLE_PRINCIPALS_PATH, sep='\t', usecols=['tconst', 'nconst', 'category'], chunksize=chunk_size, low_memory=False):
            # Filter for relevant categories (e.g., actors, directors - who are likely to meet on set)
            # Add more categories if they imply a meeting
            relevant_categories = ['actor', 'actress', 'director', 'writer', 'producer', 'composer', 'cinematographer']
            filtered_chunk = chunk_df[chunk_df['category'].isin(relevant_categories)]
            for _, row in filtered_chunk.iterrows():
                principals_by_title[row['tconst']].append(row['nconst'])
        print(f"    Processed principals for {len(principals_by_title)} titles.")

        # Create connections
        print("  Creating connections from IMDb collaborations...")
        connection_count = 0
        for tconst, nconst_list in principals_by_title.items():
            if len(nconst_list) > 1:
                # Create pairs of people who worked on the same title
                for nconst1, nconst2 in itertools.combinations(nconst_list, 2):
                    name1 = nconst_to_name.get(nconst1)
                    name2 = nconst_to_name.get(nconst2)
                    if name1 and name2:
                        add_connection(name1, name2, "imdb_collaboration", f"Worked on title {tconst}")
                        connection_count += 1
                        if connection_count % 10000 == 0:
                             print(f"    Added {connection_count} IMDb connections...")
        print(f"  Finished processing IMDb data. Added {connection_count} potential connections.")

    except FileNotFoundError as e:
        print(f"  Error: IMDb file not found: {e}. Please check IMDB_XXX_PATH variables.")
    except Exception as e:
        print(f"  An error occurred during IMDb processing: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # Fetch data from Wikidata
    fetch_wikidata_connections()

    # Process data from IMDb
    # process_imdb_data()

    # --- Step 4: Combine Data (Implicitly done by add_connection) ---
    # NetworkX automatically handles merging nodes if their names are the same.

    # --- Step 5: Basic Analysis/Output ---
    print("\n--- Graph Construction Complete ---")
    print(f"Number of nodes (people): {G.number_of_nodes()}")
    print(f"Number of edges (connections): {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        # Example: Find a path between two people if they exist
        person_a = "Leonardo DiCaprio" # Change to test
        person_b = "Tom Hanks"         # Change to test

        if G.has_node(person_a) and G.has_node(person_b):
            try:
                path = nx.shortest_path(G, source=person_a, target=person_b)
                print(f"\nShortest path between {person_a} and {person_b}:")
                for i in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[i], path[i+1])
                    print(f"  {path[i]} --({edge_data.get('source', '')}: {edge_data.get('details', '')})--> {path[i+1]}")
            except nx.NetworkXNoPath:
                print(f"\nNo path found between {person_a} and {person_b} in the constructed graph.")
            except nx.NodeNotFound:
                 print(f"\nOne or both nodes ({person_a}, {person_b}) not found for path search, though graph is not empty.")
        else:
            print(f"\nNodes {person_a} or {person_b} not found in the graph for path search.")

        # Example: List some high-degree nodes (well-connected people)
        degrees = sorted(G.degree(), key=lambda item: item[1], reverse=True)
        print("\nTop 10 most connected people in this dataset:")
        for node, degree in degrees[:10]:
            print(f"  {node}: {degree} connections")

    # Optional: Save the graph
    nx.write_gexf(G, "famous_people_connections.gexf")
    # print("\nGraph saved to famous_people_connections.gexf (can be opened with Gephi)")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")