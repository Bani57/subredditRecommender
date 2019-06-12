import networkx as nx
from matplotlib import pyplot as plt
from feature_utils import min_max_normalize
from node2vec import Node2Vec
from gensim.models import word2vec
from networkx.algorithms.community import greedy_modularity_communities


def reduce_graph_by_weight_threshold(graph, node_threshold=0, edge_threshold=0):
    node_weights = dict(nx.get_node_attributes(graph, 'weight'))
    nodes_to_remove = [node for node, weight in node_weights.items() if weight < node_threshold]

    edge_weights = dict(nx.get_edge_attributes(graph, 'weight'))
    edges_to_remove = [edge for edge, weight in edge_weights.items() if weight < edge_threshold]

    reduced_graph = graph.copy()
    reduced_graph.remove_edges_from(edges_to_remove)
    reduced_graph.remove_nodes_from(nodes_to_remove)

    return reduced_graph


def draw_network(graph, name, file, use_node_weights=True, use_edge_weights=True, draw_node_labels=True):
    node_cmap = plt.get_cmap("cool")
    edge_cmap = plt.get_cmap("Blues")

    node_positions = nx.drawing.layout.spring_layout(graph, iterations=50, seed=57)
    if use_node_weights:
        node_weights = list(nx.get_node_attributes(graph, 'weight').values())
        node_sizes = min_max_normalize(node_weights, 100, 5000)
        node_colors = node_weights
    else:
        node_sizes = 1000
        node_colors = "skyblue"
    if use_edge_weights:
        edge_weights = list(nx.get_edge_attributes(graph, 'weight').values())
        edge_widths = min_max_normalize(edge_weights, 1, 10)
        edge_colors = edge_weights
    else:
        edge_widths = 1
        edge_colors = "skyblue"
    plt.figure(figsize=(16, 9), dpi=int(1280 / 16))
    nx.draw_networkx_nodes(
        graph,
        pos=node_positions,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=node_cmap)
    if draw_node_labels:
        nx.draw_networkx_labels(graph, pos=node_positions, font_size=8)
    nx.draw_networkx_edges(
        graph,
        pos=node_positions,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=1,
        edge_cmap=edge_cmap)
    plt.suptitle(name, fontsize=24)
    plt.axis('off')
    plt.savefig(file, dpi="figure")
    plt.show()


def get_network_features_for_nodes(network):
    feature_dictionaries = []
    print("Calculating node degrees...")
    feature_dictionaries.append(dict(nx.degree(network)))
    feature_dictionaries.append(dict(nx.degree(network, weight='weight')))
    print("Calculating degree centrality...")
    feature_dictionaries.append(dict(nx.degree_centrality(network)))
    print("Calculating closeness centrality...")
    feature_dictionaries.append(dict(nx.closeness_centrality(network)))
    feature_dictionaries.append(dict(nx.closeness_centrality(network, distance='distance')))
    print("Calculating betweenness centrality...")
    feature_dictionaries.append(dict(nx.betweenness_centrality(network)))
    feature_dictionaries.append(dict(nx.betweenness_centrality(network, weight='distance')))
    print("Calculating clustering coefficients...")
    feature_dictionaries.append(dict(nx.clustering(network)))
    feature_dictionaries.append(dict(nx.clustering(network, weight='weight')))
    print("Performing HITS and calculating hub scores...")
    feature_dictionaries.append(dict(nx.hits(network)[0]))
    print("Performing PageRank and calculating scores...")
    feature_dictionaries.append(dict(nx.pagerank(network, weight=None)))
    feature_dictionaries.append(dict(nx.pagerank(network, weight='weight')))

    features = {}
    for node in network.nodes():
        features[node] = [feature_dictionary[node] for feature_dictionary in feature_dictionaries]

    return features


def get_distance_matrices(network):
    hop_counts = dict(nx.shortest_path_length(network))
    weighted_distances = dict(nx.shortest_path_length(network, weight='distance'))
    return hop_counts, weighted_distances


def get_average_distance_from_node_to_nodes(distance_matrix, node_from, nodes_to):
    num_nodes_to = len(nodes_to)
    distance_matrix_row = distance_matrix[node_from]
    max_distance = max(list(distance_matrix_row.values()))
    if num_nodes_to == 0:
        return max_distance
    distances = [distance_matrix_row[node_to] if node_to in distance_matrix_row else max_distance
                 for node_to in nodes_to]
    total_distance = sum(distances)
    return total_distance / num_nodes_to


def get_community_membership_for_nodes(network):
    communities = greedy_modularity_communities(network, weight='weight')
    print('Number of communities found: ' + str(len(communities)))
    # print('Community sizes: ' + str([len(community) for community in communities]))
    membership = {}
    for index, community in enumerate(communities):
        for node in community:
            membership[node] = index
    return membership


def get_node2vec_model_for_network(network):
    num_features = 500  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 6  # Number of threads to run in parallel
    context = 5  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    node2vec = Node2Vec(network, dimensions=num_features, walk_length=50, num_walks=30, workers=1, weight_key='weight')
    model = node2vec.fit(workers=num_workers,
                         size=num_features,
                         min_count=min_word_count,
                         window=context,
                         sample=downsampling)
    return model


def load_node2vec_embeddings_from_file(filename):
    return word2vec.Word2VecKeyedVectors.load_word2vec_format(filename, binary=True)
