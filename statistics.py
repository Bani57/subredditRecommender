from matplotlib import pyplot as plt
from file_utils import load_object
import numpy as np
import networkx as nx
from feature_utils import remove_extreme_values
from graph_utils import reduce_graph_by_weight_threshold, draw_network
from scipy.stats.kde import gaussian_kde
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


def generate_distribution_plot(vector, name, file):
    plt.figure(figsize=(16, 9), dpi=1280 // 16)
    plt.suptitle('Distribution plot for ' + name.lower(), fontsize=24)
    plt.subplot(1, 2, 1)
    plt.title('Histogram for ' + name.lower(), fontsize=18)
    plt.hist(vector, bins='sturges', color='darkorange', edgecolor='black', density=True)
    plt.xlabel(name, fontsize=18)
    x = np.arange(min(vector), max(vector), 0.01, dtype='float')
    kernel = gaussian_kde(vector)
    kernel.covariance_factor = lambda: 0.25
    kernel._compute_covariance()
    density = kernel.evaluate(x)
    plt.plot(x, density, color='red', linewidth=3)
    plt.subplot(1, 2, 2)
    plt.title('Boxplot for ' + name.lower() + '\nMedian: ' + str(np.percentile(vector, 50)), fontsize=14)
    plt.boxplot(vector, patch_artist=True, boxprops={'facecolor': 'darkorange'},
                medianprops={'color': 'black', 'linewidth': 3},
                flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 6})
    plt.xticks([])
    plt.xlabel(name, fontsize=18)
    plt.savefig(file, dpi="figure")


def get_confusion_matrix_and_plot_roc_curve(predicted_labels, predicted_label_probabilities, test_labels, name, file):
    accuracy = accuracy_score(test_labels, predicted_labels)
    roc_score = roc_auc_score(test_labels, predicted_label_probabilities[:, 1])
    confusion_mat = confusion_matrix(test_labels, predicted_labels)
    print('Confusion Matrix: \n' + str(confusion_mat))
    fpr, tpr, _ = roc_curve(test_labels, predicted_label_probabilities[:, 1], pos_label=1)
    plt.figure(figsize=(16, 9), dpi=1280 // 16)
    plt.plot(fpr, tpr, color='darkorange', lw=3)
    plt.plot([0, 1], [0, 1], color='red', lw=3, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC Curve - ' + str(name) +
              '\nAccuracy: ' + str(accuracy) +
              '\nROC AUC score: ' + str(roc_score),
              fontsize=24)
    plt.savefig(file, dpi="figure")
    return confusion_mat


if __name__ == "__main__":
    subreddits_dictionary = load_object('data/dictionaries/subreddits_dictionary_reduced')

    involvement_dictionary = load_object('data/dictionaries/involvement_dictionary_reduced')
    involvement = [value for value in involvement_dictionary.values()]
    involvement = remove_extreme_values(involvement)

    generate_distribution_plot(involvement, 'User-subreddit involvement',
                               "plots/user_subreddit_involvement_distribution.png")

    sus_network = nx.read_gpickle('data/networks/SuS')
    sus_degrees = dict(nx.degree(sus_network))
    sus_degrees = list(sus_degrees.values())

    sus_network_reduced = reduce_graph_by_weight_threshold(sus_network, 100, 10)
    draw_network(sus_network_reduced,
                 "Subreddit-user-Subreddit network", "plots/sus_network.png",
                 draw_node_labels=True)
    generate_distribution_plot(sus_degrees, 'Node degree in subreddit network',
                               "plots/sus_node_degree_distribution.png")

    usu_network = nx.read_gpickle('data/networks/UsU')
    usu_degrees = dict(nx.degree(usu_network))
    usu_degrees = [value for value in usu_degrees.values()]
    usu_degrees = remove_extreme_values(usu_degrees)

    usu_network_reduced = reduce_graph_by_weight_threshold(usu_network, node_threshold=20, edge_threshold=10)
    draw_network(usu_network_reduced,
                 "User-subreddit-user network", "plots/usu_network.png",
                 draw_node_labels=True)
    generate_distribution_plot(usu_degrees, 'Node degree in user network', "plots/usu_node_degree_distribution.png")

    usu_community_membership = load_object("data/features/UsU_community_membership")
    sus_community_membership = load_object("data/features/SuS_community_membership")

    num_sus_communities = max(list(sus_community_membership.values())) + 1
    num_usu_communities = max(list(usu_community_membership.values())) + 1

    print("Number of SuS communities:", num_sus_communities)
    print("Number of UsU communities:", num_usu_communities)

    subreddits_keywords = load_object("data/content/subreddits_keywords")
    users_keywords = load_object("data/content/users_keywords")

    print(subreddits_keywords['atheism'])
    print(subreddits_keywords['GifSound'])
    print(users_keywords["markycapone"])

    combined_model_predicted_labels = load_object("models/combined_model_predicted_labels")
    combined_model_predicted_label_probabilities = load_object("models/combined_model_predicted_label_probabilities")
    classification_testing_labels = load_object("data/labels/classification_testing_labels")

    combined_model_2 = load_object("models/combined_model_2")
    combined_model_2_predicted_labels = \
        load_object("models/combined_model_2_predicted_labels")
    combined_model_2_predicted_label_probabilities = \
        load_object("models/combined_model_2_predicted_label_probabilities")

    get_confusion_matrix_and_plot_roc_curve(combined_model_predicted_labels,
                                            combined_model_predicted_label_probabilities,
                                            classification_testing_labels,
                                            "Fusion model: Majority vote",
                                            "plots/combined_model_roc_curve.png")
    get_confusion_matrix_and_plot_roc_curve(combined_model_2_predicted_labels,
                                            combined_model_2_predicted_label_probabilities,
                                            classification_testing_labels,
                                            "Fusion model: All features",
                                            "plots/combined_model_2_roc_curve.png")
