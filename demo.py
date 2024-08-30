import langid
import streamlit as st
from streamlit_react_flow import react_flow
import re
import os

# Set up language identification
langid.set_languages(['en', 'zh'])
lang_dic = {'zh': 'en', 'en': 'zh'}

def set_page_config():
    """
    Configure the Streamlit page settings.
    """
    st.set_page_config(
        page_title='Knowledge Circuit',
        page_icon=':shark:',
        layout='wide',
        initial_sidebar_state='expanded',
        menu_items={
            'Report a bug': 'https://github.com/zjunlp/KnowledgeCircuits/issues',
            'About': '## This work aims to build the circuits in the pretrained language models that are responsible for the specific knowledge and analyze the behavior of these components.'
        }
    )

def get_gv_files(circuit_dir):
    """
    Retrieve all .gv files from the specified directory.
    """
    gv_files = []
    for subdir in os.listdir(circuit_dir):
        subdir_path = os.path.join(circuit_dir, subdir)
        if os.path.isdir(subdir_path):
            gv_file = os.path.join(subdir_path, 'graph.gv')
            if os.path.exists(gv_file):
                gv_files.append((subdir, gv_file))
    gv_files.sort(key=lambda x: x[0])
    return gv_files

def gv_to_edges(gv_file):
    """
    Parse a .gv file and extract edges.
    """
    edges = []
    edge_pattern = re.compile(r'\"<(.+?)>\" -> \"<(.+?)>\"')
    
    if isinstance(gv_file, str):
        # If it's a file path
        with open(gv_file, 'r') as file:
            lines = file.readlines()
    else:
        # If it's an UploadedFile object
        lines = gv_file.getvalue().decode().split('\n')
    
    for line in lines:
        match = edge_pattern.search(line)
        if match:
            edges.append((match.group(1), match.group(2)))
    
    return edges

def get_layer(node):
    """
    Determine the layer of a node based on its name.
    """
    if node.startswith('m'):
        return int(node[1:].split('-')[0])
    elif node.startswith('a'):
        return int(node[1:].split('.')[0])
    else:
        return 100 # For nodes like 'resid_post'

def get_attention_head(node):
    """
    Extract the attention head number from a node name.
    """
    if node.startswith('a'):
        return int(node.split('.')[-1])
    return -1

def create_elements(edges, graph_width, graph_height):
    """
    Create elements for the React Flow graph based on the edges.
    """
    nodes = set()
    for source, target in edges:
        nodes.add(source)
        nodes.add(target)

    sorted_nodes = sorted(nodes, key=lambda x: (get_layer(x), get_attention_head(x) if x.startswith('a') else -1), reverse=True)

    min_layer = min(get_layer(node) for node in nodes if get_layer(node) != float('-inf'))
    max_layer = max(get_layer(node) for node in nodes if get_layer(node) != float('-inf'))

    layer_counts = [0] * (max_layer - min_layer + 1)
    layer_heights = [0] * (max_layer - min_layer + 1)

    for node in sorted_nodes:
        layer = get_layer(node)
        if layer != 100:
            layer_counts[layer - min_layer] += 1
            layer_heights[layer - min_layer] = max(layer_heights[layer - min_layer], 1)

    total_layers = sum(layer_heights)
    layer_height = graph_height / total_layers

    elements = []
    node_positions = {}
    attention_counts = [0] * (max_layer - min_layer + 1)
    
    for node in sorted_nodes:
        layer = get_layer(node)
        if layer == 100:
            x = graph_width * 0.7 
            y = 20
        else:
            if node.startswith('m'):
                x = graph_width * 0.05 
            else:
                attention_counts[layer - min_layer] += 1
                x = graph_width * 0.85 - attention_counts[layer - min_layer] * (graph_width * 0.8) / (layer_counts[layer - min_layer] + 1)  # Moved left from 0.9 to 0.85
            y = (total_layers - sum(layer_heights[:layer - min_layer]) - 0.5) * layer_height

        node_positions[node] = {'x': x, 'y': y}

        if node.startswith('m'):
            node_type = 'MLP'
        elif node.startswith('a'):
            node_type = 'Attention'
        elif node == 'resid_post':
            node_type = 'Output'
        else:
            node_type = 'default'

        elements.append({
            'id': node,
            'data': {'label': node},
            'type': node_type,
            'position': {'x': x, 'y': y},
            'style': node_types.get(node_type, {}).get('style', {})
        })

        if 'H' in node:
            elements[-1]['position']['y'] -= layer_height / 4

    for source, target in edges:
        elements.append({
            'id': f'{source}-{target}',
            'source': source,
            'target': target,
            'type': 'smoothstep',
            'animated': True,
            'style': {
                'stroke': '#888',
                'strokeWidth': 2,
                'strokeDasharray': '5, 5'
            },
        })

    return elements

# Define node types and their styles
node_types = {
    'Input': {'style': {'background': '#f472b6', 'width': 50, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}},
    'Embedding': {'style': {'background': '#4ea8de', 'width': 50, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}},
    'Attention': {'style': {'background': '#ff9a3c', 'width': 50, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}},
    'MLP': {'style': {'background': '#6ede87', 'width': 50, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}},
    'Output': {'style': {'background': '#ffcc50', 'width': 80, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}},
}

def main():
    """
    Main function to run the our app.
    """
    set_page_config()

    st.sidebar.title("Knowledge Circuit Analysis")

    # Choose analysis method
    analysis_method = st.sidebar.radio(
        "Choose analysis method",
        ("Select from existing cases", "Upload your own file")
    )

    if analysis_method == "Select from existing cases":
        selected_formatted_name = st.sidebar.radio(
            'Select LLM Model',
            list(['gpt2-medium'])
        )

        circuit_dir = selected_formatted_name
        gv_files = get_gv_files(circuit_dir)
        case_names = [name for name, _ in gv_files]
        case = st.sidebar.selectbox('Select Case', case_names)
        selected_name = case
        gv_file_path = next((file for name, file in gv_files if name == case), "")

    else:
        uploaded_file = st.sidebar.file_uploader("Upload your .gv file", type="gv")
        if uploaded_file is not None:
            gv_file_path = uploaded_file
            selected_formatted_name = "Custom Analysis"
            selected_name = uploaded_file.name
        else:
            st.sidebar.warning("Please upload a .gv file to proceed with the analysis.")
            return

    # an introduction to the case
    st.sidebar.write(f'')

    if not gv_file_path:
        st.error(f"No graph file found for {selected_name}")
        return

    model_display_name = selected_formatted_name
    st.title(f'`{model_display_name}` Knowledge Circuit')

    print(gv_file_path)
    edges = gv_to_edges(gv_file_path)

    # Set graph dimensions
    graph_width = 1500 - 100
    graph_height = 1000 - 200

    elements = create_elements(edges, graph_width, graph_height)

    flowStyles = {'height': f'{graph_height}px', 'width': f'{graph_width}px'}

    # Render the React Flow graph
    react_flow('transformer', elements=elements, flow_styles=flowStyles)

if __name__ == "__main__":
    main()
