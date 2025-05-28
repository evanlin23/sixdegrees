import matplotlib # Import the top-level package first
matplotlib.use('Agg') # Set the backend to Agg (non-interactive)
import matplotlib.pyplot as plt # Now import pyplot

import networkx as nx
# import pandas as pd # Not strictly needed if LABEL_ATTRIBUTE_TITLE is sufficient
import numpy as np

# --- Monkey-patching for lenient GEXF attribute parsing ---
# This patch allows networkx to read GEXF files where attributes
# (float, double, integer, long, boolean) might contain non-standard
# values like 'NA', empty strings, or other unparseable text.
# It modifies GEXFReader to use custom functions for converting these
# attribute values, ensuring problematic values become float('nan') for
# float/double types, and None for integer/long/boolean types.

try:
    from networkx.readwrite.gexf import GEXFReader

    # --- Custom Caster Factory for Numeric Types ---
    def _create_lenient_numeric_caster(target_type, na_value_for_type):
        """
        Factory to create a lenient caster for numeric GEXF attributes.
        Handles 'NA', empty strings, and conversion errors.
        """
        def caster(value_from_gexf):
            # If already a number, try to convert to target_type
            if isinstance(value_from_gexf, (int, float)):
                try:
                    return target_type(value_from_gexf)
                except (ValueError, TypeError): # e.g. int(1.0) is fine, int(float('nan')) is error
                    return na_value_for_type 
            
            if not isinstance(value_from_gexf, str):
                # For non-string, non-numeric types (e.g., None)
                return na_value_for_type

            # At this point, value_from_gexf is a string.
            val_stripped = value_from_gexf.strip()
            if val_stripped.upper() == 'NA' or val_stripped == '':
                return na_value_for_type
            
            try:
                return target_type(val_stripped)
            except ValueError:
                return na_value_for_type
        return caster

    # Instantiate casters for float and int using the factory
    _custom_lenient_gexf_float_caster = _create_lenient_numeric_caster(float, float('nan'))
    _custom_lenient_gexf_int_caster = _create_lenient_numeric_caster(int, None) # 'NA' for int becomes None

    # --- Custom Caster for Boolean Type ---
    def _custom_lenient_gexf_boolean_caster(value_from_gexf):
        """
        Custom caster for GEXF attributes declared as boolean.
        Converts 'NA' (case-insensitive), empty/whitespace-only strings to None.
        Converts "true"/"false" (case-insensitive) to True/False.
        Other unparseable strings also become None.
        """
        if isinstance(value_from_gexf, bool):
            return value_from_gexf
        
        if not isinstance(value_from_gexf, str):
            return None # Non-string, non-bool becomes None

        val_stripped = value_from_gexf.strip()
        val_lower = val_stripped.lower()

        if val_lower == 'na' or val_stripped == '':
            return None
        
        if val_lower == "true":
            return True
        if val_lower == "false":
            return False
        
        # For other strings that are not "true", "false", "NA", or empty
        return None


    # Apply the patch by modifying GEXFReader.__init__
    _GEXFREADER_INIT_PATCH_FLAG_FOR_UNIVERSAL_NA_HANDLING = '_gexfreader_init_patched_for_universal_na_handling'

    if not hasattr(GEXFReader, _GEXFREADER_INIT_PATCH_FLAG_FOR_UNIVERSAL_NA_HANDLING):
        _original_gexfreader_init = GEXFReader.__init__

        def _patched_gexfreader_init(self, *args, **kwargs):
            _original_gexfreader_init(self, *args, **kwargs) # Call original __init__
            
            # Modify self.python_type for lenient parsing
            if hasattr(self, 'python_type') and isinstance(self.python_type, dict):
                # Patch float and double types
                if self.python_type.get('float') is float:
                    self.python_type['float'] = _custom_lenient_gexf_float_caster
                if self.python_type.get('double') is float: # GEXF double also maps to Python float
                    self.python_type['double'] = _custom_lenient_gexf_float_caster
                
                # Patch integer and long types
                if self.python_type.get('integer') is int:
                    self.python_type['integer'] = _custom_lenient_gexf_int_caster
                if self.python_type.get('long') is int: # GEXF long also maps to Python int
                    self.python_type['long'] = _custom_lenient_gexf_int_caster

                # Patch boolean type
                # Original GEXFReader.__init__ sets self.python_type['boolean'] = self._parse_boolean (instance method)
                # We replace this with our custom function if it's still the original.
                original_nx_bool_parser_method_name = '_parse_boolean' # Name of the GEXFReader method
                current_bool_caster = self.python_type.get('boolean')
                if current_bool_caster is not None and \
                   hasattr(current_bool_caster, '__func__') and \
                   current_bool_caster.__func__.__name__ == original_nx_bool_parser_method_name:
                    self.python_type['boolean'] = _custom_lenient_gexf_boolean_caster
                elif current_bool_caster is not None and not (hasattr(current_bool_caster, '__func__')) and \
                     current_bool_caster.__name__ == original_nx_bool_parser_method_name : # Python 2 case for instancemethod
                     self.python_type['boolean'] = _custom_lenient_gexf_boolean_caster


        GEXFReader.__init__ = _patched_gexfreader_init
        setattr(GEXFReader, _GEXFREADER_INIT_PATCH_FLAG_FOR_UNIVERSAL_NA_HANDLING, True)
        # print("Applied GEXFReader.__init__ patch for lenient NA parsing (all types).")
    # else:
        # print("GEXFReader.__init__ patch for lenient NA parsing (all types) already applied.")
        # pass

except ImportError:
    print("Warning: Could not import GEXFReader from networkx.readwrite.gexf. Patches for 'NA' handling not applied.")
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
        # With the patch, nx.read_gexf should now handle 'NA' in float, int, bool fields.
        graph = nx.read_gexf(graph_file_path)
    except FileNotFoundError:
        print(f"Error: Graph file not found at '{graph_file_path}'.")
        return
    except Exception as e:
        print(f"Error reading GEXF file: {e}")
        # If you want more details during debugging the patch:
        # import traceback
        # traceback.print_exc()
        return

    if graph.number_of_nodes() == 0:
        print("The graph is empty. Nothing to display.")
        return

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print(f"\nGraph loaded successfully:")
    print(f"  Nodes: {num_nodes}, Edges: {num_edges}")

    # Example: Check an attribute that might have been 'NA'
    # for node_id, data in graph.nodes(data=True):
    #     if 'some_integer_attribute_key' in data: # Replace with an actual key
    #         print(f"Node {node_id}, int_attr: {data['some_integer_attribute_key']} (type: {type(data['some_integer_attribute_key'])})")
    #     if 'some_float_attribute_key' in data: # Replace with an actual key
    #         print(f"Node {node_id}, float_attr: {data['some_float_attribute_key']} (type: {type(data['some_float_attribute_key'])})")
    #     if 'some_boolean_attribute_key' in data: # Replace with an actual key
    #         print(f"Node {node_id}, bool_attr: {data['some_boolean_attribute_key']} (type: {type(data['some_boolean_attribute_key'])})")
    #     break # Just check one node for example

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