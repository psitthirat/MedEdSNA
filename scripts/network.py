import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import community as community_louvain
from collections import defaultdict

def network_coauthorship(df, node_col, filter_col=None, filter_value=None):
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

    for doi, group in grouped:
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
        
    return G

def network_params(df, G, filter_col=None, filter_value=None):
    
    # Filter the dataset
    if filter_col and filter_value:
        df_nx = df[df[filter_col].isin(filter_value)]
    else:
        df_nx = df.copy()

    # Group by DOI to get co-authorship per publication
    grouped = df_nx.groupby('DOI')
    single_author = sum(1 for _, group in grouped if len(group['Author ID'].dropna().unique()) == 1)
    single_country = sum(1 for _, group in grouped if len(group['Country'].dropna().unique()) == 1)    
    
    # Compute node-level centralities
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06) if G.number_of_nodes() > 0 else {}

    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
        G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
        G.nodes[node]['closeness_centrality'] = closeness_centrality.get(node, 0)
        G.nodes[node]['eigenvector_centrality'] = eigenvector_centrality.get(node, 0)

    # Detect communities using greedy modularity
    node_community = community_louvain.best_partition(G, random_state=42)
    for node, comm_id in node_community.items():
        G.nodes[node]['modularity_class'] = comm_id
            
    # Precompute neighbors' community memberships
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

    avg_participation = (sum(participation_coefficients.values()) / len(participation_coefficients) if participation_coefficients else 0)
    top_participation = (max(participation_coefficients.values()) if participation_coefficients else 0)

    # Compute graph-level stats
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / num_nodes if num_nodes else 0
    max_degree = max(degrees.values()) if degrees else 0

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
        "No. of Communities": len(communities),
    }

    stats_df = pd.DataFrame([network_stats])

    return stats_df

# Create a network map
def draw_network(G, title=None):
    
    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=60, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    if title:
        plt.title(title)

    plt.axis('off')
    plt.show()

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