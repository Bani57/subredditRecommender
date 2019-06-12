import numpy as np
from file_utils import load_object
from graph_utils import get_average_distance_from_node_to_nodes
import pandas as pd

subreddits_dictionary = load_object('data/dictionaries/subreddits_dictionary_reduced')
all_subreddits = list(subreddits_dictionary.keys())
users_dictionary = load_object('data/dictionaries/users_dictionary_reduced')
all_users = list(users_dictionary.keys())
involvement_dictionary = load_object('data/dictionaries/involvement_dictionary_reduced')

sus_node_features = load_object("data/features/SuS_node_features")
usu_node_features = load_object("data/features/UsU_node_features")
sus_hop_count_matrix = load_object("data/distances/SuS_hop_count")
sus_weighted_distance_matrix = load_object("data/distances/SuS_weighted")
usu_hop_count_matrix = load_object("data/distances/UsU_hop_count")
usu_weighted_distance_matrix = load_object("data/distances/UsU_weighted")
sus_community_membership = load_object("data/features/SuS_community_membership")
usu_community_membership = load_object("data/features/UsU_community_membership")

sus_node_embeddings = load_object("data/features/SuS_node_embeddings")
usu_node_embeddings = load_object("data/features/UsU_node_embeddings")

subreddits_keyword_embeddings = load_object("data/features/subreddits_keyword_embeddings")
users_keyword_embeddings = load_object("data/features/users_keyword_embeddings")

network_structure_model = load_object("models/network_structure_model")
node2vec_model = load_object("models/node2vec_model")
content_similarity_model = load_object("models/content_similarity_model")


def get_network_structure_features(user, subreddit):
    user_subreddits = users_dictionary[user]
    subreddit_users = subreddits_dictionary[subreddit]

    features = list(sus_node_features[subreddit])
    features.append(get_average_distance_from_node_to_nodes(sus_hop_count_matrix, subreddit, user_subreddits))
    features.append(get_average_distance_from_node_to_nodes(sus_weighted_distance_matrix, subreddit, user_subreddits))
    features.append(sus_community_membership[subreddit])

    features.extend(usu_node_features[user])
    features.append(get_average_distance_from_node_to_nodes(usu_hop_count_matrix, user, subreddit_users))
    features.append(get_average_distance_from_node_to_nodes(usu_weighted_distance_matrix, user, subreddit_users))
    features.append(usu_community_membership[user])

    return features


def get_node2vec_features(user, subreddit):
    features = list(sus_node_embeddings.get_vector(subreddit))
    features.extend(usu_node_embeddings.get_vector(user))
    return features


def get_content_similarity_features(user, subreddit):
    features = list(subreddits_keyword_embeddings[subreddit])
    features.extend(users_keyword_embeddings[user])
    return features


def get_user_subreddit_recommendation_score(user, subreddit):
    predicted_probs = []

    network_structure_features = get_network_structure_features(user, subreddit)
    sample = np.array([network_structure_features, ], dtype="float")
    predicted_probs.append(network_structure_model.predict_proba(sample)[0])

    node2vec_features = get_node2vec_features(user, subreddit)
    sample = np.array([node2vec_features, ], dtype="float")
    predicted_probs.append(node2vec_model.predict_proba(sample)[0])

    content_similarity_features = get_content_similarity_features(user, subreddit)
    sample = np.array([content_similarity_features, ], dtype="float")
    predicted_probs.append(content_similarity_model.predict_proba(sample)[0])

    average_probs = np.mean(predicted_probs, axis=0)
    recommendation_score = average_probs[1] - average_probs[0]

    return recommendation_score


if __name__ == "__main__":

    user = None
    while True:
        if user not in all_users and user is not None:
            print("Sorry, that user is not in our database, try a different username.")
        else:
            user = input("Please enter username: ")
        if user == "END":
            break
        user_subreddits = users_dictionary[user]
        user_subreddits = sorted(user_subreddits, key=lambda x: involvement_dictionary[(user, x)], reverse=True)
        print("Top subreddits u/" + str(user) + " is active on:")
        top_user_subreddits = list(user_subreddits)
        if len(user_subreddits) > 5:
            top_user_subreddits = user_subreddits[:5]
        for subreddit in top_user_subreddits:
            print("r/" + str(subreddit))

        print("Finding subreddits to recommend...")
        user_non_subreddits = list(set(all_subreddits).difference(set(user_subreddits)))
        recommendation_scores = [(subreddit, get_user_subreddit_recommendation_score(user, subreddit))
                                 for subreddit in user_non_subreddits]
        recommendation_scores = [(subreddit, round(score, 4)) for (subreddit, score) in recommendation_scores]

        recommendation_scores = sorted(recommendation_scores, key=lambda x: x[1], reverse=True)
        if len(recommendation_scores) > 5:
            recommendation_scores = recommendation_scores[:5]

        recommendation_scores = pd.DataFrame(np.array(recommendation_scores), index=None,
                                             columns=["RECOMMENDED SUBREDDIT", "SCORE"])
        print(recommendation_scores.to_string(index=False, justify='left'))
