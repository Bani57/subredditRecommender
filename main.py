from os.path import isfile, exists
from os import makedirs
from bs4 import BeautifulSoup
from collections import OrderedDict
from date_utils import *
from file_utils import *
from graph_utils import *
from nlp_utils import *
from feature_utils import *
from classification_utils import train_model, scores_to_table, score_classifier
import random

random.seed(57)

if __name__ == "__main__":

    folder_paths = ["data/content/", "data/datasets", "data/dictionaries", "data/distances", "data/features",
                    "data/labels", "data/networks", "models/", "plots/", "scores/"]
    for folder_path in folder_paths:
        if not exists(folder_path):
            makedirs(folder_path)

    dataset = pd.read_csv('../redditSubmissions/submissions.csv', sep=',', index_col=False, header=0,
                          encoding='utf-8')

    if not isfile('data/dictionaries/subreddits_dictionary') \
            or not isfile('data/dictionaries/users_dictionary') \
            or not isfile('data/dictionaries/involvement_dictionary') \
            or not isfile('data/content/subreddits') \
            or not isfile('data/content/users'):
        print('Calculating user-subreddit involvement...')
        subreddits_dictionary = {}
        users_dictionary = {}
        involvement_dictionary = {}
        subreddits_content = {}
        users_content = {}
        for index, row in dataset.iterrows():
            if index % 5000 == 0:
                print('Processing Reddit post #' + str(index) + '...')
            subreddit = row['subreddit']
            posting_user = row['username']
            post_id = row['reddit_id']

            if pd.isna(subreddit) or pd.isna(posting_user) or pd.isna(post_id):
                continue

            subreddit = str(subreddit)
            posting_user = str(posting_user)
            post_id = str(post_id)
            unix_time = int(row['unixtime'])
            post_title = str(row['title'])

            if posting_user in ('[deleted]', 'AutoModerator'):
                continue

            subreddits_dictionary.setdefault(subreddit, OrderedDict())
            subreddits_dictionary[subreddit].setdefault(posting_user, unix_time)

            users_dictionary.setdefault(posting_user, OrderedDict())
            users_dictionary[posting_user].setdefault(subreddit, unix_time)

            involvement_dictionary.setdefault((posting_user, subreddit), 0)
            involvement_dictionary[(posting_user, subreddit)] += 10

            subreddits_content.setdefault(subreddit, [])
            subreddits_content[subreddit].append(post_title)

            users_content.setdefault(posting_user, [])
            users_content[posting_user].append(post_title)

            html_file = open('../redditHtmlData/' + post_id + '.html', 'r', encoding="utf-8")
            html_data = html_file.read()
            html_file.close()
            html_data = BeautifulSoup(html_data, features="lxml")
            comment_tags = html_data.find_all(class_="comment")

            for comment_tag in comment_tags:
                comment_text = comment_tag.find(class_="md").text
                if "[deleted]" in comment_text:
                    continue
                commenting_user = comment_tag.find(class_="author").text
                if commenting_user in ('[deleted]', 'AutoModerator'):
                    continue
                comment_utc_string = comment_tag.find("time").attrs['title']
                comment_unix_time = reddit_utc_string_to_unix_timestamp(comment_utc_string)
                subreddits_dictionary[subreddit].setdefault(commenting_user, comment_unix_time)
                users_dictionary.setdefault(commenting_user, OrderedDict())
                users_dictionary[commenting_user].setdefault(subreddit, comment_unix_time)
                involvement_dictionary.setdefault((commenting_user, subreddit), 0)
                involvement_dictionary[(commenting_user, subreddit)] += 1
                subreddits_content.setdefault(subreddit, [])
                subreddits_content[subreddit].append(comment_text)

                users_content.setdefault(commenting_user, [])
                users_content[commenting_user].append(comment_text)

        subreddits_dictionary = {subreddit: list(dict(sorted(user_dict.items(), key=lambda x: x[1])).keys()) for
                                 subreddit, user_dict in subreddits_dictionary.items()}
        users_dictionary = {user: list(dict(sorted(subreddit_dict.items(), key=lambda x: x[1])).keys()) for
                            user, subreddit_dict in users_dictionary.items()}

        save_object(subreddits_dictionary, 'data/dictionaries/subreddits_dictionary')
        save_object(users_dictionary, 'data/dictionaries/users_dictionary')
        save_object(involvement_dictionary, 'data/dictionaries/involvement_dictionary')
        save_object(subreddits_content, 'data/content/subreddits')
        save_object(users_content, 'data/content/users')
    else:
        subreddits_dictionary = load_object('data/dictionaries/subreddits_dictionary')
        users_dictionary = load_object('data/dictionaries/users_dictionary')
        involvement_dictionary = load_object('data/dictionaries/involvement_dictionary')

    num_subreddits = len(subreddits_dictionary.keys())
    num_users = len(users_dictionary.keys())
    print('Total number of subreddits: ' + str(num_subreddits))
    print('Total number of users: ' + str(num_users))
    print('Total number of user-subreddit interactions: ' + str(len(involvement_dictionary)))

    if not isfile('data/dictionaries/users_dictionary_reduced'):
        users_dictionary_reduced = {user: subreddits for user, subreddits in users_dictionary.items() if
                                    len(subreddits) >= 10}
        save_object(users_dictionary_reduced, 'data/dictionaries/users_dictionary_reduced')
    else:
        users_dictionary_reduced = load_object('data/dictionaries/users_dictionary_reduced')

    num_users_reduced = len(users_dictionary_reduced)
    print('Reduced number of users: ' + str(num_users_reduced))

    if not isfile('data/networks/UsU'):
        print('Creating UsU network...')

        usu_network = nx.Graph()
        users_items = list(users_dictionary_reduced.items())

        for i in range(num_users_reduced):
            user, subreddits = users_items[i]
            num_user_subreddits = len(subreddits)
            usu_network.add_node(user, weight=num_user_subreddits)
            for j in range(i + 1, num_users_reduced):
                user_2, subreddits_2 = users_items[j]
                num_common_subreddits = len(set(subreddits).intersection(set(subreddits_2)))
                if num_common_subreddits >= 7:
                    usu_network.add_edge(user, user_2, weight=num_common_subreddits,
                                         distance=num_subreddits - num_common_subreddits)
        nx.write_gpickle(usu_network, 'data/networks/UsU')
    else:
        usu_network = nx.read_gpickle('data/networks/UsU')

    num_usu_edges = nx.number_of_edges(usu_network)
    num_possible_usu_edges = num_users_reduced * (num_users_reduced - 1) / 2
    print("Number of edges in UsU network: " + str(num_usu_edges))
    print("Edge density of UsU network: " + str(round(100 * num_usu_edges / num_possible_usu_edges, 2)) + "%")

    if not isfile('data/dictionaries/subreddits_dictionary_reduced'):
        subreddits_dictionary_reduced = {
            subreddit: set([user for user in users if user in users_dictionary_reduced.keys()])
            for subreddit, users in subreddits_dictionary.items()}
        save_object(subreddits_dictionary_reduced, 'data/dictionaries/subreddits_dictionary_reduced')
    else:
        subreddits_dictionary_reduced = load_object('data/dictionaries/subreddits_dictionary_reduced')

    if not isfile('data/networks/SuS'):
        print('Creating SuS network...')

        sus_network = nx.Graph()
        subreddits_items = list(subreddits_dictionary_reduced.items())

        for i in range(num_subreddits):
            subreddit, users = subreddits_items[i]
            sus_network.add_node(subreddit, weight=len(users))
            for j in range(i + 1, num_subreddits):
                subreddit_2, users_2 = subreddits_items[j]
                num_common_users = len(set(users).intersection(set(users_2)))
                if num_common_users >= 1:
                    sus_network.add_edge(subreddit, subreddit_2, weight=num_common_users,
                                         distance=num_users_reduced - num_common_users)

        nx.write_gpickle(sus_network, 'data/networks/SuS')
    else:
        sus_network = nx.read_gpickle('data/networks/SuS')

    num_sus_edges = nx.number_of_edges(sus_network)
    num_possible_sus_edges = num_subreddits * (num_subreddits - 1) / 2
    print("Number of edges in SuS network: " + str(num_sus_edges))
    print("Edge density of SuS network: " + str(round(100 * num_sus_edges / num_possible_sus_edges, 2)) + "%")

    if not isfile('data/dictionaries/involvement_dictionary_reduced'):
        involvement_dictionary_reduced = {(user, subreddit): involvement for (user, subreddit), involvement in
                                          involvement_dictionary.items() if user in users_dictionary_reduced.keys()}
        save_object(involvement_dictionary_reduced, 'data/dictionaries/involvement_dictionary_reduced')
    else:
        involvement_dictionary_reduced = load_object('data/dictionaries/involvement_dictionary_reduced')

    print('Reduced number of user-subreddit interactions: ' + str(len(involvement_dictionary_reduced)))

    del subreddits_dictionary, users_dictionary, involvement_dictionary

    if not isfile("data/features/SuS_node_features"):
        print("Calculating network features for subreddits...")
        sus_node_features = get_network_features_for_nodes(sus_network)
        save_object(sus_node_features, "data/features/SuS_node_features")
    else:
        sus_node_features = load_object("data/features/SuS_node_features")

    if not isfile("data/features/UsU_node_features"):
        print("Calculating network features for users...")
        usu_node_features = get_network_features_for_nodes(usu_network)
        save_object(usu_node_features, "data/features/UsU_node_features")
    else:
        usu_node_features = load_object("data/features/UsU_node_features")

    if not isfile("data/features/SuS_community_membership") or not isfile("data/features/UsU_community_membership"):
        print("Performing community detection for subreddits...")
        sus_community_membership = get_community_membership_for_nodes(sus_network)
        save_object(sus_community_membership, "data/features/SuS_community_membership")
        print("Performing community detection for users...")
        usu_community_membership = get_community_membership_for_nodes(usu_network)
        save_object(usu_community_membership, "data/features/UsU_community_membership")
    else:
        sus_community_membership = load_object("data/features/SuS_community_membership")
        usu_community_membership = load_object("data/features/UsU_community_membership")

    if not isfile("data/distances/SuS_hop_count") or not isfile("data/distances/SuS_weighted") \
            or not isfile("data/distances/UsU_hop_count") or not isfile("data/distances/UsU_weighted"):
        print("Calculating distance matrices for the SuS network...")
        sus_hop_count_matrix, sus_weighted_distance_matrix = get_distance_matrices(sus_network)
        save_object(sus_hop_count_matrix, "data/distances/SuS_hop_count")
        save_object(sus_weighted_distance_matrix, "data/distances/SuS_weighted")
        print("Calculating distance matrices for the UsU network...")
        usu_hop_count_matrix, usu_weighted_distance_matrix = get_distance_matrices(usu_network)
        save_object(usu_hop_count_matrix, "data/distances/UsU_hop_count")
        save_object(usu_weighted_distance_matrix, "data/distances/UsU_weighted")
    else:
        sus_hop_count_matrix = load_object("data/distances/SuS_hop_count")
        sus_weighted_distance_matrix = load_object("data/distances/SuS_weighted")
        usu_hop_count_matrix = load_object("data/distances/UsU_hop_count")
        usu_weighted_distance_matrix = load_object("data/distances/UsU_weighted")

    if not isfile("data/features/user_to_subreddit_users_average_hop_count") \
            or not isfile("data/features/user_to_subreddit_users_average_weighted_distance") \
            or not isfile("data/features/subreddit_to_user_subreddits_average_hop_count") \
            or not isfile("data/features/subreddit_to_user_subreddits_average_weighted_distance"):
        subreddit_to_user_subreddits_average_hop_count = {}
        subreddit_to_user_subreddits_average_weighted_distance = {}
        user_to_subreddit_users_average_hop_count = {}
        user_to_subreddit_users_average_weighted_distance = {}
        print("Calculating distance features...")
        for i, (user, subreddit) in enumerate(list(involvement_dictionary_reduced.keys())):
            if i % 5000 == 0:
                print("Processing user-subreddit pair #" + str(i) + "...")
            user_other_subreddits = list(users_dictionary_reduced[user])
            user_other_subreddits.remove(subreddit)
            subreddit_to_user_subreddits_average_hop_count[(user, subreddit)] = \
                get_average_distance_from_node_to_nodes(sus_hop_count_matrix, subreddit, user_other_subreddits)
            subreddit_to_user_subreddits_average_weighted_distance[(user, subreddit)] = \
                get_average_distance_from_node_to_nodes(sus_weighted_distance_matrix, subreddit, user_other_subreddits)

            subreddit_other_users = list(subreddits_dictionary_reduced[subreddit])
            subreddit_other_users.remove(user)
            user_to_subreddit_users_average_hop_count[(user, subreddit)] = \
                get_average_distance_from_node_to_nodes(usu_hop_count_matrix, user, subreddit_other_users)
            user_to_subreddit_users_average_weighted_distance[(user, subreddit)] = \
                get_average_distance_from_node_to_nodes(usu_weighted_distance_matrix, user, subreddit_other_users)

        save_object(subreddit_to_user_subreddits_average_hop_count,
                    "data/features/subreddit_to_user_subreddits_average_hop_count")
        save_object(subreddit_to_user_subreddits_average_weighted_distance,
                    "data/features/subreddit_to_user_subreddits_average_weighted_distance")
        save_object(user_to_subreddit_users_average_hop_count,
                    "data/features/user_to_subreddit_users_average_hop_count")
        save_object(user_to_subreddit_users_average_weighted_distance,
                    "data/features/user_to_subreddit_users_average_weighted_distance")

    else:
        subreddit_to_user_subreddits_average_hop_count = load_object(
            "data/features/subreddit_to_user_subreddits_average_hop_count")
        subreddit_to_user_subreddits_average_weighted_distance = load_object(
            "data/features/subreddit_to_user_subreddits_average_weighted_distance")
        user_to_subreddit_users_average_hop_count = load_object(
            "data/features/user_to_subreddit_users_average_hop_count")
        user_to_subreddit_users_average_weighted_distance = load_object(
            "data/features/user_to_subreddit_users_average_weighted_distance")

    if not isfile("data/features/SuS_node_embeddings"):
        print("Calculating subredddit node2vec embeddings...")
        sus_node2vec_model = get_node2vec_model_for_network(sus_network)
        sus_node2vec_model.save("models/SuS_node2vec")
        sus_node_embeddings = sus_node2vec_model.wv
        save_object(sus_node_embeddings, "data/features/SuS_node_embeddings")
    else:
        sus_node_embeddings = load_object("data/features/SuS_node_embeddings")

    if not isfile("data/features/UsU_node_embeddings"):
        print("Calculating user node2vec embeddings...")
        usu_node2vec_model = get_node2vec_model_for_network(usu_network)
        usu_node2vec_model.save("models/UsU_node2vec")
        usu_node_embeddings = usu_node2vec_model.wv
        save_object(usu_node_embeddings, "data/features/UsU_node_embeddings")
    else:
        usu_node_embeddings = load_object("data/features/UsU_node_embeddings")

    all_subreddits = list(subreddits_dictionary_reduced.keys())
    all_users = list(users_dictionary_reduced.keys())

    if not isfile("data/all_subreddits.txt") or not isfile("data/all_users.txt"):
        all_subreddits_file = open("data/all_subreddits.txt", "w", encoding="utf-8")
        for subreddit in all_subreddits:
            all_subreddits_file.write(subreddit + '\n')
        all_subreddits_file.close()
        all_users_file = open("data/all_users.txt", "w", encoding="utf-8")
        for user in all_users:
            all_users_file.write(user + '\n')
        all_users_file.close()

    subreddits_content = load_object('data/content/subreddits')
    users_content = load_object('data/content/users')

    if not isfile("data/content/users_vocabularies"):
        print("Calculating the vocabulary of each user...")
        users_vocabularies = {}
        for index, user in enumerate(all_users):
            if index % 100 == 0:
                print("Processing content from user #" + str(index) + "...")
            content = users_content[user]
            documents = [get_words_from_reddit_text(text) for text in content]
            processed_documents = [post_process_words(document) for document in documents]
            vocabulary = [word for document in processed_documents for word in document]
            users_vocabularies[user] = vocabulary
        save_object(users_vocabularies, "data/content/users_vocabularies")
    else:
        users_vocabularies = load_object("data/content/users_vocabularies")
    del users_content

    if not isfile("data/content/subreddits_vocabularies"):
        print("Calculating the vocabulary of each subreddit...")
        subreddits_vocabularies = {}
        for index, subreddit in enumerate(all_subreddits):
            if index % 50 == 0:
                print("Processing content from subreddit #" + str(index) + "...")
            content = subreddits_content[subreddit]
            documents = [get_words_from_reddit_text(text) for text in content]
            processed_documents = [post_process_words(document) for document in documents]
            vocabulary = [word for document in processed_documents for word in document]
            subreddits_vocabularies[subreddit] = vocabulary
        save_object(subreddits_vocabularies, "data/content/subreddits_vocabularies")
    else:
        subreddits_vocabularies = load_object("data/content/subreddits_vocabularies")
    del subreddits_content

    if not isfile("data/content/users_keywords"):
        print("Determining user keywords...")
        users_keywords = get_keywords_for_vocabularies(users_vocabularies)
        save_object(users_keywords, "data/content/users_keywords")
    else:
        users_keywords = load_object("data/content/users_keywords")
    del users_vocabularies

    if not isfile("data/content/subreddits_keywords"):
        print("Determining subreddit keywords...")
        subreddits_keywords = get_keywords_for_vocabularies(subreddits_vocabularies)
        save_object(subreddits_keywords, "data/content/subreddits_keywords")
    else:
        subreddits_keywords = load_object("data/content/subreddits_keywords")
    del subreddits_vocabularies

    if not isfile("data/features/subreddits_keyword_embeddings") \
            or not isfile("data/features/users_keyword_embeddings"):
        print("Calculating average word2vec embeddings from subreddit keywords...")
        word2vec_model = load_word2vec_google_model()
        subreddits_keyword_embeddings = {}
        for subreddit, keywords in subreddits_keywords.items():
            subreddits_keyword_embeddings[subreddit] = get_average_word2vec_keyword_embeddings(word2vec_model, keywords)
        save_object(subreddits_keyword_embeddings, "data/features/subreddits_keyword_embeddings")
        print("Calculating average word2vec embeddings from user keywords...")
        users_keyword_embeddings = {}
        for user, keywords in users_keywords.items():
            users_keyword_embeddings[user] = get_average_word2vec_keyword_embeddings(word2vec_model, keywords)
        save_object(users_keyword_embeddings, "data/features/users_keyword_embeddings")
        del word2vec_model
    else:
        subreddits_keyword_embeddings = load_object("data/features/subreddits_keyword_embeddings")
        users_keyword_embeddings = load_object("data/features/users_keyword_embeddings")

    if not isfile("data/labels/network_structure_feature_labels") \
            or not isfile("data/labels/node2vec_feature_labels") \
            or not isfile("data/labels/content_similarity_feature_labels"):
        network_structure_features = ["NODE DEGREE", "WEIGHTED NODE DEGREE", "DEGREE CENTRALITY",
                                      "CLOSENESS CENTRALITY", "WEIGHTED CLOSENESS CENTRALITY",
                                      "BETWEENNESS CENTRALITY", "WEIGHTED BETWEENNESS CENTRALITY",
                                      "CLUSTERING COEFFICIENT", "WEIGHTED CLUSTERING COEFFICIENT",
                                      "HUB SCORE", "PAGERANK", "WEIGHTED PAGERANK",
                                      "AVERAGE HOP COUNT", "AVERAGE WEIGHTED DISTANCE",
                                      "COMMUNITY"]
        network_structure_feature_labels = ["[SUBREDDIT] " + name for name in network_structure_features]
        network_structure_feature_labels.extend(["[USER] " + name for name in network_structure_features])

        node2vec_feature_labels = ["SUBREDDIT NODE EMBEDDING #" + str(i + 1) for i in range(500)]
        node2vec_feature_labels.extend(["USER NODE EMBEDDING #" + str(i + 1) for i in range(500)])

        content_similarity_feature_labels = ["SUBREDDIT KEYWORD EMBEDDING #" + str(i + 1) for i in range(300)]
        content_similarity_feature_labels.extend(["USER KEYWORD EMBEDDING #" + str(i + 1) for i in range(300)])

        save_object(network_structure_feature_labels, "data/labels/network_structure_feature_labels")
        save_object(node2vec_feature_labels, "data/labels/node2vec_feature_labels")
        save_object(content_similarity_feature_labels, "data/labels/content_similarity_feature_labels")
    else:
        network_structure_feature_labels = load_object("data/labels/network_structure_feature_labels")
        node2vec_feature_labels = load_object("data/labels/node2vec_feature_labels")
        content_similarity_feature_labels = load_object("data/labels/content_similarity_feature_labels")

    if not isfile("data/datasets/network_structure_training_set.csv") \
            or not isfile("data/datasets/network_structure_testing_set.csv") \
            or not isfile("data/datasets/node2vec_training_set.csv") \
            or not isfile("data/datasets/node2vec_testing_set.csv") \
            or not isfile("data/datasets/content_similarity_training_set.csv") \
            or not isfile("data/datasets/content_similarity_testing_set.csv") \
            or not isfile("data/labels/classification_training_labels") \
            or not isfile("data/labels/classification_testing_labels"):
        print("Creating datasets...")
        network_structure_training_set = []
        network_structure_testing_set = []
        node2vec_training_set = []
        node2vec_testing_set = []
        content_similarity_training_set = []
        content_similarity_testing_set = []
        classification_training_labels = []
        classification_testing_labels = []
        for i, (user, user_subreddits) in enumerate(users_dictionary_reduced.items()):
            if i % 100 == 0:
                print("Processing subreddits of user #" + str(i) + "...")
            num_user_subreddits = len(user_subreddits)
            num_training_samples = int(0.8 * num_user_subreddits)
            num_testing_samples = num_user_subreddits - num_training_samples
            user_non_subreddits = list(set(all_subreddits).difference(set(user_subreddits)))
            user_non_subreddits = random.sample(user_non_subreddits, num_user_subreddits)
            user_subreddit_samples = list(user_subreddits)
            user_subreddit_samples.extend(user_non_subreddits)
            network_structure_positive_samples = []
            network_structure_negative_samples = []
            node2vec_positive_samples = []
            node2vec_negative_samples = []
            content_similarity_positive_samples = []
            content_similarity_negative_samples = []
            for index, subreddit in enumerate(user_subreddit_samples):
                user_subreddit_tuple = (user, subreddit)
                network_structure_feature_vector = list(sus_node_features[subreddit])
                if index < num_user_subreddits:
                    # Positive sample
                    network_structure_feature_vector.append(
                        subreddit_to_user_subreddits_average_hop_count[user_subreddit_tuple])
                    network_structure_feature_vector.append(
                        subreddit_to_user_subreddits_average_weighted_distance[user_subreddit_tuple])
                else:
                    # Negative sample
                    network_structure_feature_vector.append(
                        get_average_distance_from_node_to_nodes(sus_hop_count_matrix, subreddit,
                                                                user_subreddits))
                    network_structure_feature_vector.append(
                        get_average_distance_from_node_to_nodes(sus_weighted_distance_matrix, subreddit,
                                                                user_subreddits))
                network_structure_feature_vector.append(sus_community_membership[subreddit])

                network_structure_feature_vector.extend(usu_node_features[user])
                if index < num_user_subreddits:
                    # Positive sample
                    network_structure_feature_vector.append(
                        user_to_subreddit_users_average_hop_count[user_subreddit_tuple])
                    network_structure_feature_vector.append(
                        user_to_subreddit_users_average_weighted_distance[user_subreddit_tuple])
                else:
                    # Negative sample
                    subreddit_users = subreddits_dictionary_reduced[subreddit]
                    network_structure_feature_vector.append(
                        get_average_distance_from_node_to_nodes(usu_hop_count_matrix, user, subreddit_users))
                    network_structure_feature_vector.append(
                        get_average_distance_from_node_to_nodes(usu_weighted_distance_matrix, user, subreddit_users))
                network_structure_feature_vector.append(usu_community_membership[user])

                node2vec_feature_vector = list(sus_node_embeddings.get_vector(subreddit))
                node2vec_feature_vector.extend(usu_node_embeddings.get_vector(user))

                content_similarity_feature_vector = list(subreddits_keyword_embeddings[subreddit])
                content_similarity_feature_vector.extend(users_keyword_embeddings[user])

                if index < num_user_subreddits:
                    # Positive sample
                    network_structure_positive_samples.append(network_structure_feature_vector)
                    node2vec_positive_samples.append(node2vec_feature_vector)
                    content_similarity_positive_samples.append(content_similarity_feature_vector)
                else:
                    # Negative sample
                    network_structure_negative_samples.append(network_structure_feature_vector)
                    node2vec_negative_samples.append(node2vec_feature_vector)
                    content_similarity_negative_samples.append(content_similarity_feature_vector)

            network_structure_training_set.extend(network_structure_positive_samples[:num_training_samples])
            network_structure_training_set.extend(network_structure_negative_samples[:num_training_samples])
            node2vec_training_set.extend(node2vec_positive_samples[:num_training_samples])
            node2vec_training_set.extend(node2vec_negative_samples[:num_training_samples])
            content_similarity_training_set.extend(content_similarity_positive_samples[:num_training_samples])
            content_similarity_training_set.extend(content_similarity_negative_samples[:num_training_samples])
            classification_training_labels.extend([1, ] * num_training_samples)
            classification_training_labels.extend([0, ] * num_training_samples)

            network_structure_testing_set.extend(network_structure_positive_samples[num_training_samples:])
            network_structure_testing_set.extend(network_structure_negative_samples[num_training_samples:])
            node2vec_testing_set.extend(node2vec_positive_samples[num_training_samples:])
            node2vec_testing_set.extend(node2vec_negative_samples[num_training_samples:])
            content_similarity_testing_set.extend(content_similarity_positive_samples[num_training_samples:])
            content_similarity_testing_set.extend(content_similarity_negative_samples[num_training_samples:])
            classification_testing_labels.extend([1, ] * num_testing_samples)
            classification_testing_labels.extend([0, ] * num_testing_samples)

        network_structure_training_set = pd.DataFrame(network_structure_training_set,
                                                      columns=network_structure_feature_labels)
        network_structure_training_set.to_csv('data/datasets/network_structure_training_set.csv', sep=',', header=True,
                                              index=False, encoding='utf-8')
        node2vec_training_set = pd.DataFrame(node2vec_training_set, columns=node2vec_feature_labels)
        node2vec_training_set.to_csv("data/datasets/node2vec_training_set.csv", sep=',', header=True,
                                     index=False, encoding='utf-8')
        content_similarity_training_set = pd.DataFrame(content_similarity_training_set,
                                                       columns=content_similarity_feature_labels)
        content_similarity_training_set.to_csv("data/datasets/content_similarity_training_set.csv", sep=',',
                                               header=True, index=False, encoding='utf-8')
        save_object(classification_training_labels, "data/labels/classification_training_labels")

        network_structure_testing_set = pd.DataFrame(network_structure_testing_set,
                                                     columns=network_structure_feature_labels)
        network_structure_testing_set.to_csv('data/datasets/network_structure_testing_set.csv', sep=',', header=True,
                                             index=False, encoding='utf-8')
        node2vec_testing_set = pd.DataFrame(node2vec_testing_set, columns=node2vec_feature_labels)
        node2vec_testing_set.to_csv("data/datasets/node2vec_testing_set.csv", sep=',', header=True,
                                    index=False, encoding='utf-8')
        content_similarity_testing_set = pd.DataFrame(content_similarity_testing_set,
                                                      columns=content_similarity_feature_labels)
        content_similarity_testing_set.to_csv("data/datasets/content_similarity_testing_set.csv", sep=',',
                                              header=True, index=False, encoding='utf-8')
        save_object(classification_testing_labels, "data/labels/classification_testing_labels")

    else:

        classification_training_labels = load_object("data/labels/classification_training_labels")
        classification_testing_labels = load_object("data/labels/classification_testing_labels")

    del sus_node_features, usu_node_features
    del sus_hop_count_matrix, sus_weighted_distance_matrix, usu_hop_count_matrix, usu_weighted_distance_matrix
    del subreddit_to_user_subreddits_average_hop_count, subreddit_to_user_subreddits_average_weighted_distance
    del user_to_subreddit_users_average_hop_count, user_to_subreddit_users_average_weighted_distance
    del sus_community_membership, usu_community_membership
    del sus_node_embeddings, usu_node_embeddings
    del subreddits_keyword_embeddings, users_keyword_embeddings

    if not isfile("models/network_structure_model") or not isfile("scores/network_structure_model.csv"):
        print("Loading network structure dataset...")
        network_structure_training_set = pd.read_csv('data/datasets/network_structure_training_set.csv', sep=',',
                                                     index_col=False, header=0, encoding='utf-8')
        network_structure_testing_set = pd.read_csv('data/datasets/network_structure_testing_set.csv', sep=',',
                                                    index_col=False, header=0, encoding='utf-8')
        # print("Preprocessing network structure dataset...")
        # feature_selector = FeatureSelector(network_structure_training_set, classification_training_labels)
        # feature_selector.preprocess_dataset()
        # network_structure_training_set = feature_selector.dataset
        # feature_selector = FeatureSelector(network_structure_testing_set, classification_testing_labels)
        # feature_selector.preprocess_dataset()
        # network_structure_testing_set = feature_selector.dataset
        print("Training network structure model...")
        dataset_tuple = (network_structure_training_set, classification_training_labels,
                         network_structure_testing_set, classification_testing_labels)
        network_structure_model, network_structure_model_scores = train_model(dataset_tuple)
        save_object(network_structure_model, "models/network_structure_model")
        network_structure_model_scores = scores_to_table(network_structure_model_scores)
        network_structure_model_scores.to_csv('scores/network_structure_model.csv', sep=',', index=True, header=True,
                                              encoding='utf-8')
    else:
        network_structure_model = load_object("models/network_structure_model")

    if not isfile("models/node2vec_model") or not isfile("scores/node2vec_model.csv"):
        print("Loading node2vec dataset...")
        node2vec_training_set = pd.read_csv('data/datasets/node2vec_training_set.csv', sep=',',
                                            index_col=False, header=0, encoding='utf-8')
        node2vec_testing_set = pd.read_csv('data/datasets/node2vec_testing_set.csv', sep=',',
                                           index_col=False, header=0, encoding='utf-8')
        print("Training node2vec model...")
        dataset_tuple = (node2vec_training_set, classification_training_labels,
                         node2vec_testing_set, classification_testing_labels)
        node2vec_model, node2vec_model_scores = train_model(dataset_tuple)
        save_object(node2vec_model, "models/node2vec_model")
        node2vec_model_scores = scores_to_table(node2vec_model_scores)
        node2vec_model_scores.to_csv('scores/node2vec_model.csv', sep=',', index=True, header=True,
                                     encoding='utf-8')
    else:
        node2vec_model = load_object("models/node2vec_model")

    if not isfile("models/content_similarity_model") or not isfile("scores/content_similarity_model.csv"):
        print("Loading content similarity dataset...")
        content_similarity_training_set = pd.read_csv('data/datasets/content_similarity_training_set.csv', sep=',',
                                                      index_col=False, header=0, encoding='utf-8')
        content_similarity_testing_set = pd.read_csv('data/datasets/content_similarity_testing_set.csv', sep=',',
                                                     index_col=False, header=0, encoding='utf-8')
        print("Training content similarity model...")
        dataset_tuple = (content_similarity_training_set, classification_training_labels,
                         content_similarity_testing_set, classification_testing_labels)
        content_similarity_model, content_similarity_model_scores = train_model(dataset_tuple)
        save_object(content_similarity_model, "models/content_similarity_model")
        content_similarity_model_scores = scores_to_table(content_similarity_model_scores)
        content_similarity_model_scores.to_csv('scores/content_similarity_model.csv', sep=',', index=True, header=True,
                                               encoding='utf-8')
    else:
        content_similarity_model = load_object("models/content_similarity_model")

    if not isfile("scores/combined_model.csv"):
        print("Scoring combined model...")
        network_structure_testing_set = pd.read_csv('data/datasets/network_structure_testing_set.csv', sep=',',
                                                    index_col=False, header=0, encoding='utf-8')
        node2vec_testing_set = pd.read_csv('data/datasets/node2vec_testing_set.csv', sep=',',
                                           index_col=False, header=0, encoding='utf-8')
        content_similarity_testing_set = pd.read_csv('data/datasets/content_similarity_testing_set.csv', sep=',',
                                                     index_col=False, header=0, encoding='utf-8')

        network_structure_predicted_labels = network_structure_model.predict(network_structure_testing_set)
        node2vec_predicted_labels = node2vec_model.predict(node2vec_testing_set)
        content_similarity_predicted_labels = content_similarity_model.predict(content_similarity_testing_set)

        predicted_labels_majority = np.array([1 if sum(labels) >= 2 else 0 for labels in
                                              zip(network_structure_predicted_labels, node2vec_predicted_labels,
                                                  content_similarity_predicted_labels)], dtype="int")
        save_object(predicted_labels_majority, "models/combined_model_predicted_labels")

        network_structure_predicted_probs = network_structure_model.predict_proba(network_structure_testing_set)
        node2vec_predicted_probs = node2vec_model.predict_proba(node2vec_testing_set)
        content_similarity_predicted_probs = content_similarity_model.predict_proba(content_similarity_testing_set)

        predicted_average_probs = np.array([np.mean(probabilities, axis=0) for probabilities in
                                            zip(network_structure_predicted_probs, node2vec_predicted_probs,
                                                content_similarity_predicted_probs)], dtype="float")
        save_object(predicted_average_probs, "models/combined_model_predicted_label_probabilities")

        combined_model_scores = {"Combined model": score_classifier(predicted_labels_majority, predicted_average_probs,
                                                                    classification_testing_labels)}
        combined_model_scores = scores_to_table(combined_model_scores)
        combined_model_scores.to_csv('scores/combined_model.csv', sep=',', index=True, header=True,
                                     encoding='utf-8')
