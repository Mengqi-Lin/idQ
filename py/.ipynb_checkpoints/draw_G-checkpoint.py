import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt

def draw_G_K4(Q):
    """
    Draw the extended Hasse diagram G(Q) for a given Q (with K=4).
    
    This function draws the full Hasse diagram on {0,1}^4 (using fixed node positions),
    then restricts it to the latent nodes in the representative set R(Q).
    It also adds item nodes for each row of Q, labeling them with the exact binary vector,
    and positions each item node directly below the latent node corresponding to that vector.
    
    Parameters:
        Q (np.ndarray): A binary matrix of shape (J,4).
    """
    # Fixed positions for the full Hasse diagram on {0,1}^4.
    positions = {
        '1000': (2, 2), '0100': (4, 2), '0010': (6, 2), '0001': (8, 2),
        '1100': (1, 4), '1010': (3, 4), '1001': (5, 4),
        '0110': (7, 4), '0101': (9, 4), '0011': (11, 4),
        '1110': (2, 6), '1101': (4, 6), '1011': (6, 6), '0111': (8, 6),
        '1111': (5, 8)
    }
    
    # Create the full Hasse diagram as a directed graph.
    G = nx.DiGraph()
    for node in positions.keys():
        G.add_node(node)
    
    edges = [
        ('1000', '1100'), ('1000', '1010'), ('1000', '1001'),
        ('0100', '1100'), ('0100', '0110'), ('0100', '0101'),
        ('0010', '1010'), ('0010', '0110'), ('0010', '0011'),
        ('0001', '1001'), ('0001', '0101'), ('0001', '0011'),
        ('1100', '1110'), ('1100', '1101'),
        ('1010', '1110'), ('1010', '1011'),
        ('1001', '1101'), ('1001', '1011'),
        ('0110', '1110'), ('0110', '0111'),
        ('0101', '1101'), ('0101', '0111'),
        ('0011', '1011'), ('0011', '0111'),
        ('1110', '1111'),
        ('1101', '1111'),
        ('1011', '1111'),
        ('0111', '1111')
    ]
    G.add_edges_from(edges)
    
    # Compute the representative node set R(Q).
    rep_set_tuples = representative_node_set(Q)
    # Convert each tuple to a string (e.g., (1,0,0,0) -> "1000").
    rep_set = {''.join(map(str, t)) for t in rep_set_tuples}
    
    # Remove from G any latent node not in rep_set.
    nodes_to_remove = [node for node in G.nodes if node not in rep_set]
    G.remove_nodes_from(nodes_to_remove)
    
    # Create a new graph H that will contain both latent and item nodes.
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(G.edges(data=True))
    
    # Create and add item nodes.
    # For each row j of Q, create an item node labeled with its binary vector,
    # and place it directly below the corresponding latent node.
    J = Q.shape[0]
    item_positions = {}
    for j in range(J):
        vec_str = ''.join(map(str, Q[j]))
        item_node = f'item_{j}'
        H.add_node(item_node, label=vec_str, type='item')
        # If the latent node exists, position the item node directly below it.
        if vec_str in positions:
            x, y = positions[vec_str]
            # Offset vertically (e.g., 1 unit below).
            item_positions[item_node] = (x, y - 1)
        else:
            # Otherwise, assign a default position.
            item_positions[item_node] = (j, 0)
        # Add a directed edge from the item node to the corresponding latent node.
        if vec_str in H.nodes:
            H.add_edge(item_node, vec_str)
        else:
            print(f"Warning: latent node for vector {vec_str} not found in H.")
    
    # Merge positions: latent nodes use positions from the fixed dictionary, and item nodes use item_positions.
    pos = {}
    for node in H.nodes():
        if node in positions:
            pos[node] = positions[node]
        elif node in item_positions:
            pos[node] = item_positions[node]
    
    # Draw the graph.
    plt.figure(figsize=(8, 6))
    # Draw latent nodes (circles).
    latent_nodes = [n for n in H.nodes() if n in positions]
    nx.draw_networkx_nodes(H, pos, nodelist=latent_nodes, node_color='skyblue', node_shape='o', node_size=800)
    # Draw item nodes (squares) using the label attribute.
    item_nodes = [n for n in H.nodes() if H.nodes[n].get('type') == 'item']
    nx.draw_networkx_nodes(H, pos, nodelist=item_nodes, node_color='lightgreen', node_shape='s', node_size=800)
    # Draw edges.
    nx.draw_networkx_edges(H, pos, arrowstyle='->', arrowsize=15)
    # Build labels: for latent nodes, label is the node name; for item nodes, label is the 'label' attribute.
    labels = {}
    for node in H.nodes():
        if H.nodes[node].get('type') == 'item':
            labels[node] = H.nodes[node].get('label')
        else:
            labels[node] = node
    nx.draw_networkx_labels(H, pos, labels, font_size=10)
    
    plt.axis('off')
    plt.title(r"$G(Q)$")
    plt.show()

    
def draw_G_K5(Q):
    """
    Draw the extended Hasse diagram G(Q) for a given Q (with K=5).
    
    1) Creates the full Hasse diagram on {0,1}^5.
    2) Removes any latent node not in R(Q).
    3) Adds item nodes for each row of Q, placing them just below their corresponding latent node.
    4) Draws the resulting graph with a symmetric (centered) layered layout.
    
    Parameters:
        Q (np.ndarray): A binary matrix of shape (J,5).
    """
    # ============== 1) Create full Hasse diagram over {0,1}^5 ==============
    # All 5-bit patterns as strings, e.g., "00000", "00001", ...
    all_5bit = [''.join(bits) for bits in itertools.product('01', repeat=5)]
    
    # Group nodes by the number of ones (this defines the layers).
    layer_dict = {}
    for s in all_5bit:
        r = s.count('1')
        layer_dict.setdefault(r, []).append(s)
    for r in layer_dict:
        layer_dict[r].sort()
    
    # Build a positions dictionary with symmetric layout.
    # For each layer r (with nodes having r ones), we center the nodes horizontally.
    positions = {}
    x_gap = 2    # Horizontal gap between nodes.
    y_gap = 2    # Vertical gap between layers.
    for r in range(6):  # Layers 0 through 5.
        nodes = layer_dict.get(r, [])
        n = len(nodes)
        for i, node in enumerate(nodes):
            # Center nodes: shift x so that the middle node is at x=0.
            x = (i - (n - 1) / 2) * x_gap
            y = r * y_gap
            positions[node] = (x, y)
    
    # Build edges: For each node x, add an edge to node y if y is obtained by flipping one '0' in x to '1'.
    G = nx.DiGraph()
    for node in all_5bit:
        G.add_node(node)
    for x in all_5bit:
        for i in range(5):
            if x[i] == '0':
                y = x[:i] + '1' + x[i+1:]
                # Optionally, only add edge if y has exactly one more 1.
                if y.count('1') == x.count('1') + 1:
                    G.add_edge(x, y)
    
    # ============== 2) Remove latent nodes not in R(Q) ==============
    rep_set_tuples = representative_node_set(Q)
    rep_set = {''.join(map(str, t)) for t in rep_set_tuples}
    
    nodes_to_remove = [node for node in G.nodes() if node not in rep_set]
    G.remove_nodes_from(nodes_to_remove)
    
    # ============== 3) Create a new graph H that includes item nodes ==============
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(G.edges(data=True))
    
    # ============== 4) Add item nodes for each row in Q ==============
    # For each row j, create an item node labeled with the row's bit pattern.
    J = Q.shape[0]
    item_positions = {}
    for j in range(J):
        row_bits = ''.join(map(str, Q[j]))  # e.g., "10010"
        item_node = f'item_{j}'
        H.add_node(item_node, label=row_bits, type='item')
        if row_bits in positions:
            x, y = positions[row_bits]
            item_positions[item_node] = (x, y - 1)  # Place it 1 unit below the latent node.
        else:
            item_positions[item_node] = (j * x_gap, -y_gap)
        if row_bits in H.nodes():
            H.add_edge(item_node, row_bits)
        else:
            print(f"Warning: latent node for vector {row_bits} not found in H.")
    
    # Merge positions: latent nodes use positions from 'positions', item nodes from 'item_positions'.
    final_positions = {}
    for node in H.nodes():
        if node in positions:
            final_positions[node] = positions[node]
        elif node in item_positions:
            final_positions[node] = item_positions[node]
    
    # ============== 5) Draw the final graph ==============
    plt.figure(figsize=(10, 8))
    
    # Separate latent and item nodes.
    latent_nodes = [n for n in H.nodes() if n in positions]
    item_nodes = [n for n in H.nodes() if H.nodes[n].get('type') == 'item']
    
    # Draw latent nodes (circles, skyblue).
    nx.draw_networkx_nodes(H, final_positions, nodelist=latent_nodes, node_color='skyblue', node_shape='o', node_size=700)
    # Draw item nodes (squares, lightgreen).
    nx.draw_networkx_nodes(H, final_positions, nodelist=item_nodes, node_color='lightgreen', node_shape='s', node_size=700)
    # Draw edges.
    nx.draw_networkx_edges(H, final_positions, arrowstyle='->', arrowsize=15)
    
    # Build labels.
    labels = {}
    for node in H.nodes():
        if H.nodes[node].get('type') == 'item':
            labels[node] = H.nodes[node].get('label')
        else:
            labels[node] = node
    nx.draw_networkx_labels(H, final_positions, labels, font_size=9)
    
    plt.axis('off')
    plt.title(r"$G(Q)$ for $K=5$")
    plt.show()
    
    
    
    