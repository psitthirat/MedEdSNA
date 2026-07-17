"""
Co-authorship Network Analysis Utility

This script provides functionality to construct and analyze co-authorship networks 
from bibliometric datasets (e.g., Scopus exports). It is designed to evaluate 
collaboration structures, centrality, modularity, and small-world properties of 
research communities.

Key functionalities include:
- Building weighted co-authorship graphs from publication datasets (DOI-based grouping).
- Adding node-level metadata (frequency, labels, attributes) for visualization in tools like Gephi.
- Computing centrality measures: degree, betweenness, closeness, eigenvector.
- Performing community detection using the Louvain algorithm and calculating modularity.
- Measuring participation coefficients (`participation_coefficient`) to assess cross-community integration.
- Calculating graph-level statistics: clustering, density, degree distribution, shortest paths, diameter, small-world coefficient.
- Assessing homophily based on chosen node attributes (e.g., country, institution).
- Exporting summary network statistics in tabular format for downstream analysis.

Dependencies:
- pandas
- numpy
- tqdm
- networkx
- python-louvain (community)
- collections

Usage:
1. Prepare a DataFrame with at least 'DOI' and node attributes (e.g., Author ID, Country).
2. Build the graph with `network_coauthorship(df, node_col, ...)`.
3. Analyze the graph with `network_params(df, G, homophily_attr=...)`.

Example:
    import pandas as pd
    from network_module import network_coauthorship, network_params

    df = pd.read_csv("coauthorship_data.csv")
    G = network_coauthorship(df, node_col="Author ID", node_label=["Country"])
    stats = network_params(df, G, homophily_attr="Country")
    print(stats)

Authors: P. Sitthirat et al
Version: 1.0
License: MIT License
"""


import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import community as community_louvain
from collections import defaultdict

def network_coauthorship(df, node_col, node_label=None, filter_col=None, filter_value=None):
    # Filter the dataset
    if filter_col and filter_value:
        df_nx = df[df[filter_col].isin(filter_value)]
    else:
        df_nx = df.copy()

    # Group by DOI to get co-authorship per publication
    grouped = df_nx.groupby('DOI')  

    # Create the graph
    G = nx.Graph()

    # Track frequency of each node (country or author) by DOI presence
    node_frequency = df_nx.groupby(node_col)['DOI'].nunique().to_dict()

    for doi, group in tqdm(grouped, total=len(grouped), desc="Building coauthorship graph"):
        authors = group[node_col].dropna().unique()
        for i, author1 in enumerate(authors):
            for author2 in authors[i + 1:]:
                if G.has_edge(author1, author2):
                    G[author1][author2]['weight'] += 1
                else:
                    G.add_edge(author1, author2, weight=1)

    # Add frequency as node attribute for Gephi
    for node in G.nodes():
        G.nodes[node]['size'] = node_frequency.get(node, 1)
        
        if node_label:
            node_data = df_nx[df_nx[node_col] == node].iloc[0]
            for label in node_label:
                G.nodes[node][label] = node_data.get(label, None)
        
    return G

def participation_coefficient(G, node_community):
    """
    Compute the participation coefficient of every node in a graph.

    The participation coefficient measures how evenly a node's (weighted) edges
    are distributed across communities: 0 means all edges stay within the
    node's own community, values closer to 1 mean edges are spread evenly
    across many communities. As a side effect, the coefficient is also stored
    on each node as the `participation_coefficient` attribute.

    Parameters:
    - G (networkx.Graph): The graph to analyze. Edge weights are read from the
      'weight' attribute (defaulting to 1 if absent).
    - node_community (dict): Mapping of node -> community id, e.g. as returned
      by `community.best_partition`.

    Returns:
    - dict: Mapping of node -> participation coefficient.

    Example:
        >>> partition = community_louvain.best_partition(G, random_state=42)
        >>> pc = participation_coefficient(G, partition)
    """

    participation_coefficients = {}
    for node in G.nodes():
        k_i = G.degree(node, weight='weight')
        if k_i == 0:
            participation = 0
        else:
            comm_weights = defaultdict(float)
            for neighbor in G.neighbors(node):
                edge_weight = G[node][neighbor].get('weight', 1)
                comm = node_community.get(neighbor)
                comm_weights[comm] += edge_weight
            participation = 1 - sum((w / k_i) ** 2 for w in comm_weights.values())
        G.nodes[node]['participation_coefficient'] = participation
        participation_coefficients[node] = participation

    return participation_coefficients

def network_params(df, G, filter_col=None, filter_value=None, homophily_attr=None):
    if filter_col and filter_value:
        df_nx = df[df[filter_col].isin(filter_value)]
    else:
        df_nx = df.copy()

    # Group by DOI to count single-author/single-country publications
    grouped = df_nx.groupby('DOI')
    single_author = sum(1 for _, group in grouped if len(group['Author ID'].dropna().unique()) == 1)
    single_country = sum(1 for _, group in grouped if len(group['Country'].dropna().unique()) == 1)    

    # if centralities:
        
    # Centralities
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06) if G.number_of_nodes() > 0 else {}
        
    print('Analysed centrality successfully.')
    
    # Community detection and modularity
    node_community = community_louvain.best_partition(G, random_state=42)
    nx.set_node_attributes(G, node_community, 'modularity_class')
    modularity_score = community_louvain.modularity(node_community, G, weight='weight')
    num_communities = len(set(node_community.values()))

    # Participation Coefficient
    participation_coefficients = participation_coefficient(G, node_community)

    avg_participation = np.mean(list(participation_coefficients.values())) if participation_coefficients else 0
    top_participation = max(participation_coefficients.values()) if participation_coefficients else 0

    print('Analysed community and participation coefficient successfully.')

    # Graph-level stats
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values())) if degrees else 0
    max_degree = max(degrees.values()) if degrees else 0

    # Shortest path and small-world
    if G.number_of_nodes() > 0:
        if nx.is_connected(G):
            G_largest = G
        else:
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc_nodes).copy()

        largest_nodes = G_largest.number_of_nodes()

        if G_largest.number_of_nodes() > 1:            
            avg_shortest_path = nx.average_shortest_path_length(G_largest)
            
            C_rand_list = []
            L_rand_list = []
            
            for _ in range(10):
                
                # Generate a random graph with the same number of nodes and edges
                G_rand = nx.gnm_random_graph(n=G.number_of_nodes(), m=G.number_of_edges())
                if nx.is_connected(G_rand):
                    C_rand = nx.average_clustering(G_rand)
                    L_rand = nx.average_shortest_path_length(G_rand)
                    C_rand_list.append(C_rand)
                    L_rand_list.append(L_rand)
                    
            C_rand_avg = sum(C_rand_list) / len(C_rand_list)
            L_rand_avg = sum(L_rand_list) / len(L_rand_list)
                    
            diameter = nx.diameter(G_largest)
            
            small_world = (avg_clustering / C_rand_avg) / (avg_shortest_path / L_rand_avg) if avg_shortest_path else np.nan
            
        else:
            avg_shortest_path = diameter = small_world = np.nan
    else:
        avg_shortest_path = diameter = small_world = np.nan
        largest_nodes = np.nan

    print('Analysed shortest path successfully.')

    # Homophily (fraction of same-attribute links)
    homophily_results = {}
    if homophily_attr:
        if not isinstance(homophily_attr, list):
            homophily_attr = [homophily_attr]
        for attr in homophily_attr:
            same = 0
            total = 0
            for u, v in G.edges():
                if G.nodes[u].get(attr) == G.nodes[v].get(attr):
                    same += 1
                total += 1
            homophily_results[f"Homophily on {attr}"] = same / total if total > 0 else np.nan

    print('Analysed homophily successfully.')

    # Collect into dictionary
    network_stats = {
        f"{filter_col}": filter_value,
        "No of Publications": len(df_nx['DOI'].unique()),
        "Single Author": single_author,
        "Single Country": single_country,
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Density": density,
        "Average Clustering": avg_clustering,
        "Average Degree": avg_degree,
        "Max Degree": max_degree,
        "Top Degree Centrality": max(degree_centrality.values()) if degree_centrality else 0,
        "Top Betweenness Centrality": max(betweenness_centrality.values()) if betweenness_centrality else 0,
        "Top Closeness Centrality": max(closeness_centrality.values()) if closeness_centrality else 0,
        "Top Eigenvector Centrality": max(eigenvector_centrality.values()) if eigenvector_centrality else 0,
        "Average Participation Coefficient": avg_participation,
        "Top Participation Coefficient": top_participation,
        "No. of Communities": num_communities,
        "Modularity Score": modularity_score,
        "Largest connected node": largest_nodes,
        "Average Shortest Path Length": avg_shortest_path,
        "Graph Diameter": diameter,
        "Small-World Coefficient": small_world
    }
    network_stats.update(homophily_results)

    return pd.DataFrame([network_stats])


# def draw_map(G, pos):
    
#     plt.figure(figsize=(15, 10))
#     m = Basemap(projection='merc', resolution='l',
#                 llcrnrlat=-60, urcrnrlat=80,
#                 llcrnrlon=-180, urcrnrlon=180)

#     m.drawcoastlines()
#     m.drawcountries()
#     m.drawmapboundary(fill_color='lightblue')
#     m.fillcontinents(color='lightgray', lake_color='lightblue')

#     pos = {}
#     for node in G.nodes(data=True):
#         lat = node[1].get('lat')
#         lon = node[1].get('lon')
#         if pd.notna(lat) and pd.notna(lon):
#             x, y = m(lon, lat)
#             pos[node[0]] = (x, y)

#     # Draw nodes and edges
#     nx.draw_networkx_nodes(G, pos, node_size=40, node_color='red', edgecolors='black')
#     nx.draw_networkx_edges(G, pos, alpha=0.3)
#     plt.show()