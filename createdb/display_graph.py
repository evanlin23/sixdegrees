import matplotlib # Import the top-level package first
matplotlib.use('Agg') # Set the backend to Agg (non-interactive)
import matplotlib.pyplot as plt # Now import pyplot

import networkx as nx
# import pandas as pd # Not strictly needed if LABEL_ATTRIBUTE_TITLE is sufficient
import numpy as np

# --- Configuration ---
GRAPH_FILE_PATH = "met_people_graph.gexf"
LABEL_ATTRIBUTE_TITLE = "name" 
OUTPUT_IMAGE_FILE = "graph_visualization_final.png" # Define output filename here

# --- Main Display Function ---
def display_graph(graph_file_path):
    print(f"Attempting to load graph from: {graph_file_path}")
    try:
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
    try:
        # For very large graphs, consider reducing iterations or k, or using a simpler layout
        pos = nx.spring_layout(graph_to_draw, seed=42, k=0.5, iterations=50) 
        if not pos:
            print("Error: Position dictionary 'pos' is empty!")
            return
        
        print("\n--- Node Position Statistics ---")
        x_coords_list = [p[0] for p in pos.values()] 
        y_coords_list = [p[1] for p in pos.values()]
        if x_coords_list and y_coords_list:
            print(f"  X-coords: Min={min(x_coords_list):.2f}, Max={max(x_coords_list):.2f}")
            print(f"  Y-coords: Min={min(y_coords_list):.2f}, Max={max(y_coords_list):.2f}")
        else:
            print("  Could not extract coordinates.")
            return
    
    except Exception as e:
        print(f"Error during layout calculation: {e}. Using random layout as fallback.")
        pos = nx.random_layout(graph_to_draw, seed=42)
        x_coords_list = [p[0] for p in pos.values()]
        y_coords_list = [p[1] for p in pos.values()]


    print("\nDrawing graph using Matplotlib's plot and scatter...")
    
    for u, v in graph_to_draw.edges():
        if u in pos and v in pos:
            x_edge = [pos[u][0], pos[v][0]]
            y_edge = [pos[u][1], pos[v][1]]
            ax.plot(x_edge, y_edge, color='gray', alpha=0.3, lw=0.5, zorder=1) # Lighter edges
        else:
            print(f"Warning: Missing position for edge ({u}, {v})")

    ax.scatter(x_coords_list, y_coords_list, s=100, c='skyblue', alpha=0.8, edgecolors='darkblue', linewidths=0.3, zorder=2) # Smaller nodes for dense graphs

    print("\nAdding labels...")
    label_count = 0
    for node_id, (x_node, y_node) in pos.items():
        node_data = graph.nodes[node_id]
        label_text = ""
        if LABEL_ATTRIBUTE_TITLE and LABEL_ATTRIBUTE_TITLE in node_data:
            label_text = str(node_data[LABEL_ATTRIBUTE_TITLE]) # Ensure string
        elif 'label' in node_data: 
            label_text = str(node_data['label']) # Ensure string
        else: 
            label_text = str(node_id) 
        
        if len(label_text) > 20: # Shorter label limit
            label_text = label_text[:17] + "..."

        ax.text(x_node, y_node + 0.01, 
                label_text, 
                fontsize=6, # Smaller font for dense graphs
                ha='center', 
                va='bottom', 
                color='#333333', # Dark gray
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
        # fig.savefig will work correctly with the 'Agg' backend
        fig.savefig(OUTPUT_IMAGE_FILE, dpi=300, bbox_inches='tight') 
        print(f"\nGraph visualization saved to {OUTPUT_IMAGE_FILE}")
    except Exception as e:
        print(f"Error saving graph image: {e}")
        
    # With 'Agg' backend, plt.show() does nothing, so it can be removed or left (it's a no-op)
    # plt.show() 

    # It's good practice to close the figure to free memory, especially in loops or long scripts
    plt.close(fig)


if __name__ == "__main__":
    display_graph(GRAPH_FILE_PATH)