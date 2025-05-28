import matplotlib # Import the top-level package first
matplotlib.use('Agg') # Set the backend to Agg (non-interactive)
import matplotlib.pyplot as plt # Now import pyplot

import networkx as nx
# import pandas as pd # Not strictly needed if LABEL_ATTRIBUTE_TITLE is sufficient
import numpy as np

# --- Monkey-patching for lenient GEXF float parsing ---
# This patch allows networkx to read GEXF files where float attributes
# might contain non-standard values like 'NA', empty strings, or other unparseable text.
# It modifies the GEXFReader to use a custom function for converting
# attribute values declared as 'float' or 'double' in the GEXF,
# ensuring such problematic values become float('nan').

try:
    from networkx.readwrite.gexf import GEXFReader

    # Define the lenient caster function
    def _custom_lenient_gexf_float_caster(value_from_gexf):
        """
        Custom caster for GEXF attributes declared as float or double.
        Converts 'NA' (case-insensitive), empty/whitespace-only strings to float('nan').
        Also converts other strings that cause ValueError during float() to float('nan').
        If the input is already a number (int/float), it's cast to float.
        Other non-string, non-numeric types (e.g., None) also become float('nan').
        """
        if isinstance(value_from_gexf, (float, int)): # If already a number
            return float(value_from_gexf)
        
        if not isinstance(value_from_gexf, str):
            # For other non-string types (like None passed from parser, or unexpected objects)
            # print(f"GEXF Read Warning: Attribute value '{value_from_gexf}' (type: {type(value_from_gexf)})"
            #       f" is not a string or number. Using NaN for GEXF float/double attribute.")
            return float('nan')

        # At this point, value_from_gexf is a string.
        val_stripped = value_from_gexf.strip()
        if val_stripped.upper() == 'NA' or val_stripped == '':
            return float('nan')
        
        try:
            # Attempt to convert the stripped string to a float.
            return float(val_stripped)
        except ValueError:
            # Catches other strings that cannot be converted to float (e.g., "Missing", "N/A", or just text).
            # print(f"GEXF Read Warning: Could not convert attribute string '{value_from_gexf}' to float. Using NaN.")
            return float('nan')

    # Apply the patch by modifying GEXFReader.__init__
    # This flag ensures the patch is applied only once, even if the module is reloaded.
    _GEXFREADER_INIT_PATCH_FLAG_FOR_FLOAT_NA = '_gexfreader_init_patched_for_custom_float_na_parsing'

    if not hasattr(GEXFReader, _GEXFREADER_INIT_PATCH_FLAG_FOR_FLOAT_NA):
        _original_gexfreader_init = GEXFReader.__init__

        def _patched_gexfreader_init(self, *args, **kwargs):
            # Call the original __init__ method of GEXFReader
            _original_gexfreader_init(self, *args, **kwargs)
            
            # After the original __init__ has run, self.python_type dictionary should exist.
            # We modify this instance dictionary to use our custom caster.
            if hasattr(self, 'python_type') and isinstance(self.python_type, dict):
                # Only replace if the current caster is the built-in `float`.
                # This avoids issues if it's already been patched or is some other callable.
                if self.python_type.get('float') is float:
                    self.python_type['float'] = _custom_lenient_gexf_float_caster
                
                # GEXF 'double' type also typically maps to Python's `float`.
                if self.python_type.get('double') is float:
                    self.python_type['double'] = _custom_lenient_gexf_float_caster
            # else:
                # This case would be unexpected if GEXFReader.__init__ behaves as known.
                # print("Warning: GEXFReader patch: self.python_type not found or not a dict after __init__.")

        GEXFReader.__init__ = _patched_gexfreader_init
        setattr(GEXFReader, _GEXFREADER_INIT_PATCH_FLAG_FOR_FLOAT_NA, True)
        # print("Applied GEXFReader.__init__ patch for lenient float parsing of 'NA' values.") # For debugging
    # else:
        # print("GEXFReader.__init__ patch for lenient float parsing already applied.") # For debugging
        # pass

except ImportError:
    # Handle cases where GEXFReader might not be found (e.g., very old NetworkX or unusual install)
    print("Warning: Could not import GEXFReader from networkx.readwrite.gexf. Patch for 'NA' float values not applied.")
    pass
# --- End of monkey-patching ---


# --- Configuration ---
GRAPH_FILE_PATH = "met_people_graph.gexf"
LABEL_ATTRIBUTE_TITLE = "name" 
OUTPUT_IMAGE_FILE = "graph_visualization_final.png" # Define output filename here

# --- Main Display Function ---
def display_graph(graph_file_path):
    print(f"Attempting to load graph from: {graph_file_path}")
    try:
        # With the patch, nx.read_gexf should now handle 'NA' in float fields.
        graph = nx.read_gexf(graph_file_path)
    except FileNotFoundError:
        print(f"Error: Graph file not found at '{graph_file_path}'.")
        return
    except Exception as e:
        print(f"Error reading GEXF file: {e}")
        return

    if graph.number_of_nodes() == 0:
        print("The graph is empty. Nothing to display.")
        return

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print(f"\nGraph loaded successfully:")
    print(f"  Nodes: {num_nodes}, Edges: {num_edges}")

    fig, ax = plt.subplots(figsize=(20, 20)) # Even larger for potentially many labels

    graph_to_draw = graph
    
    print("\nCalculating layout...")
    pos = {}
    x_coords_list, y_coords_list = [], [] # Initialize to prevent NameError if layout fails early
    try:
        pos = nx.spring_layout(graph_to_draw, seed=42, k=0.5, iterations=50) 
        if not pos and graph_to_draw.number_of_nodes() > 0:
            print("Warning: Position dictionary 'pos' is empty after layout calculation! Using random layout.")
            pos = nx.random_layout(graph_to_draw, seed=42)
        
        if pos: # Ensure pos is not empty
            x_coords_list = [p[0] for p in pos.values()] 
            y_coords_list = [p[1] for p in pos.values()]
        
        if x_coords_list and y_coords_list:
            print("\n--- Node Position Statistics ---")
            print(f"  X-coords: Min={min(x_coords_list):.2f}, Max={max(x_coords_list):.2f}")
            print(f"  Y-coords: Min={min(y_coords_list):.2f}, Max={max(y_coords_list):.2f}")
        elif graph_to_draw.number_of_nodes() > 0:
            print("  Could not extract coordinates for layout statistics.")
            if not pos and graph_to_draw.number_of_nodes() > 0: # If pos became empty again
                 print("Re-attempting with random_layout due to coordinate extraction issues.")
                 pos = nx.random_layout(graph_to_draw, seed=42)
                 if pos:
                     x_coords_list = [p[0] for p in pos.values()]
                     y_coords_list = [p[1] for p in pos.values()]
                 if not (x_coords_list and y_coords_list):
                     print("Critical error: Failed to get coordinates even with random_layout.")
                     plt.close(fig)
                     return
    
    except Exception as e:
        print(f"Error during layout calculation: {e}. Using random layout as fallback.")
        if graph_to_draw.number_of_nodes() > 0:
            pos = nx.random_layout(graph_to_draw, seed=42)
            if pos:
                x_coords_list = [p[0] for p in pos.values()]
                y_coords_list = [p[1] for p in pos.values()]
        else:
            print("Graph has no nodes, cannot calculate fallback layout.")
            plt.close(fig)
            return

    if not pos: # If pos is still empty after all attempts
        print("Error: Layout positions are not available. Cannot draw graph.")
        plt.close(fig)
        return

    print("\nDrawing graph using Matplotlib's plot and scatter...")
    
    for u, v in graph_to_draw.edges():
        if u in pos and v in pos:
            x_edge = [pos[u][0], pos[v][0]]
            y_edge = [pos[u][1], pos[v][1]]
            ax.plot(x_edge, y_edge, color='gray', alpha=0.3, lw=0.5, zorder=1)
        else:
            print(f"Warning: Missing position for edge ({u}, {v})")

    if x_coords_list and y_coords_list: # Ensure coordinates are available
        ax.scatter(x_coords_list, y_coords_list, s=100, c='skyblue', alpha=0.8, edgecolors='darkblue', linewidths=0.3, zorder=2)
    else:
        print("Warning: Node coordinates for scatter plot are missing. Nodes will not be drawn if lists are empty.")

    print("\nAdding labels...")
    label_count = 0
    for node_id, node_position in pos.items():
        x_node, y_node = node_position
        node_data = graph.nodes[node_id]
        label_text = ""
        if LABEL_ATTRIBUTE_TITLE and LABEL_ATTRIBUTE_TITLE in node_data:
            label_text = str(node_data[LABEL_ATTRIBUTE_TITLE])
        elif 'label' in node_data: 
            label_text = str(node_data['label'])
        else: 
            label_text = str(node_id) 
        
        if len(label_text) > 20:
            label_text = label_text[:17] + "..."

        ax.text(x_node, y_node + 0.01, 
                label_text, 
                fontsize=6, 
                ha='center', 
                va='bottom', 
                color='#333333',
                zorder=3)
        label_count +=1
    print(f"Attempted to draw {label_count} labels.")

    ax.set_title(f"People Who May Have Met (Nodes: {num_nodes}, Edges: {num_edges})", size=16, pad=15)
    
    if x_coords_list and y_coords_list:
        min_x, max_x = min(x_coords_list), max(x_coords_list)
        min_y, max_y = min(y_coords_list), max(y_coords_list)
        
        padding_x = (max_x - min_x) * 0.05 if (max_x - min_x) > 0 else 0.2 
        padding_y = (max_y - min_y) * 0.05 if (max_y - min_y) > 0 else 0.2
        
        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)
    else:
        ax.autoscale_view()

    ax.axis('off') 

    try:
        fig.savefig(OUTPUT_IMAGE_FILE, dpi=300, bbox_inches='tight') 
        print(f"\nGraph visualization saved to {OUTPUT_IMAGE_FILE}")
    except Exception as e:
        print(f"Error saving graph image: {e}")
        
    plt.close(fig)


if __name__ == "__main__":
    display_graph(GRAPH_FILE_PATH)