import json
import spacy
from spacy import displacy
from spacy.matcher import DependencyMatcher
import json
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
# from sentence_transformers import SentenceTransformer
# from models import InferSent
# import torch
import tensorflow as tf
import pickle


nlp = spacy.load("en_core_web_sm")
matcher = DependencyMatcher(nlp.vocab)

# Model = Universal Sentence Encoder:
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model_Universal = hub.load(module_url)
print("module %s loaded" % module_url)


def df_model_embed(model, sentences_index, sentences_input):
    """
    This function takes phrases/sentences and create Dataframe with their embeddings

    :param model: The sentence embedding model used (SentenceBERT/InferSent/Universal Sentence Encoder)
    :param sentences_index: The sentences which will be used as Indices for the Dataframe (since with the prefix is too long).
    :param sentences_input: The phrases/sentences the model generates embedding for.
    :return: Dataframe containing the sentences themselves and their embeddings.
    """
    if model is model_Universal:
        embed = model(sentences_input)
    else:
        embed = model.encode(sentences_input)

    df = pd.DataFrame(data=embed, index=sentences_index)
    return df


def dependency_matcher():
    pattern = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "PREDICATE",
            "RIGHT_ATTRS": {"DEP": "dobj"},
        },
        {
            "LEFT_ID": "PREDICATE",
            "REL_OP": ">",
            "RIGHT_ID": "PREDICATE_modifier",
            "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}},
        }
    ]

    pattern_sec = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "PREDICATE",
            "RIGHT_ATTRS": {"DEP": "dobj"},
        }
    ]

    pattern_third = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "PREDICATE",
            "RIGHT_ATTRS": {"DEP": "dobj"},
        },
        {
            "LEFT_ID": "PREDICATE",
            "REL_OP": ">",
            "RIGHT_ID": "PREDICATE_modifier",
            "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}},
        },
        {
            "LEFT_ID": "PREDICATE_modifier",
            "REL_OP": ".",
            "RIGHT_ID": "PREDICATE_modifier_modifier",
            "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}},
        }
    ]

    matcher.add("OBJECT", [pattern])
    matcher.add("OBJECT", [pattern_sec])
    matcher.add("OBJECT", [pattern_third])
    return matcher


def doc2phrase_restricted(data_json, num_of_phrases):
    phrase_list = []
    for dict in data_json:
        if len(phrase_list) < num_of_phrases:
            sentence = dict['words']
            # word tokenizeing and part-of-speech tagger
            document = ' '.join(sentence)

            doc = nlp(document)
            print(doc)
            # displacy.serve(doc, style="dep")  # http://localhost:5000/ copy to web page to see the visualization
            matcher = dependency_matcher()
            matches = matcher(doc)
            # print("matches: ", len(matches))  # [(4851363122962674176, [6, 0, 10, 9])]
            # Each token_id corresponds to one pattern dict
            match_list = []
            [match_list.append(matches[match][1]) for match in range(len(matches))]
            match_list = [list(x) for x in set(frozenset(i) for i in [set(i) for i in match_list])]
            # print(match_list)

            for match in match_list:
                # if len(matches) != 0:
                token_ids = match
                token_ids.sort()
                phrase = []
                [phrase.append(doc[token_id].lemma_) for token_id in token_ids]
                phrase = ' '.join(phrase)
                phrase_list.append(phrase)
                print(phrase)
            print(len(phrase_list))
    phrase_list = list(set(phrase_list))
    return phrase_list





def doc2phrase(data_json):
    phrase_list = []
    for dict in data_json:
        sentence = dict['words']
        # word tokenizeing and part-of-speech tagger
        document = ' '.join(sentence)

        doc = nlp(document)
        # print(doc)
        # displacy.serve(doc, style="dep")  # http://localhost:5000/ copy to web page to see the visualization
        matcher = dependency_matcher()
        matches = matcher(doc)
        # print("matches: ", len(matches))  # [(4851363122962674176, [6, 0, 10, 9])]
        # Each token_id corresponds to one pattern dict
        match_list = []
        [match_list.append(matches[match][1]) for match in range(len(matches))]
        match_list = [list(x) for x in set(frozenset(i) for i in [set(i) for i in match_list])]
        # print(match_list)

        for match in match_list:
            # if len(matches) != 0:
            token_ids = match
            token_ids.sort()
            phrase = []
            [phrase.append(doc[token_id].lemma_) for token_id in token_ids]
            phrase = ' '.join(phrase)
            phrase_list.append(phrase)
            # print(phrase)
        print(len(phrase_list))
    phrase_list = list(set(phrase_list))
    return phrase_list


def find_nearest(array, value):
    """
    find the index of the array's element who has the nearest value to the compared one ('value')
    :param array:
    :param value:
    :return: index of the array's element with nearest value
    """
    abs_val_array = np.abs(array - value)
    smallest_diff_indx = abs_val_array.argmin()
    # closest_dist = array[smallest_diff_indx]
    return smallest_diff_indx


def plot_dist_on_level(linkage_matrix):
    """
    plot the distance values as function of level/iteration of the linkage matrix
    :param linkage_matrix:
    :return: offered distance criterion for further clustering
    """
    # add circle on the point of highest slope (and add value in the title)
    distances = linkage_matrix[:, 2]
    first_derivative = np.diff(np.array(distances))
    sec_derivative = np.diff(np.array(distances), 2)
    idxs = np.arange(1, len(distances) + 1)
    indx_1der = idxs[:-1] + 1
    indx_2der = idxs[:-2] + 2

    x_steepest_1der = np.argmax(first_derivative)
    y_steepest_1der = distances[x_steepest_1der]

    x_steepest_2der = np.argmax(first_derivative)
    y_steepest_2der = distances[x_steepest_2der]

    plt.plot(idxs, distances, label='distances')
    plt.plot(indx_1der, first_derivative, label='1 der')
    plt.plot(indx_2der, sec_derivative, label='2 der')
    plt.plot(x_steepest_1der, y_steepest_1der, 'ro')
    plt.plot(x_steepest_2der, y_steepest_2der, 'go')
    plt.legend(loc='lower right')

    # plt.plot(x_steepest, y_steepest, 'ro')
    # plt.title("Cluster's distance as function of Linkage's iteration\n Highest slope at distance = "+str(round(y_steepest, 2)))

    plt.title("Cluster's distance as function of Linkage's iteration")
    plt.ylabel('Distance')
    plt.xlabel("Linkage's iteration")
    plt.show()

    # return round(y_steepest, 2)


def extract_clusters(Z, phrases_list, start_dist_cut_criterion, end_dist_cut_criterion):
    """

    :param Z: linkage matrix
    :param phrases_list: list of original phrases (str) which we create embedding for
    :param start_dist_cut_criterion: start printing clusters from clusters whose distance is close to this value
    :param end_dist_cut_criterion: stop creating clusters if the distance between clusters is above this value
    :return: dictionary of clusters
    """
    clusters = {}
    len_phrase_list = Z.shape[0] + 1

    # finding the closest distance value to the distance cut criterion:
    distances = np.array(Z[:, 2])
    end_dist_criterion = find_nearest(distances, end_dist_cut_criterion)
    start_dist_criterion = find_nearest(distances, start_dist_cut_criterion)
    # for row in range(len_phrase_list - 1):
    for row in range(start_dist_criterion, end_dist_criterion):
        distance = Z[row, 2]

        cluster_n = row + len_phrase_list
        # which clusters / labels are present in this row
        glob1, glob2 = Z[row, 0], Z[row, 1]

        # if this is a cluster, pull the cluster
        this_clust = []
        for glob in [int(glob1), int(glob2)]:
            if glob > (len_phrase_list-1):
                this_clust += clusters[glob]['Phrases']
            # if it isn't, add the label to this cluster
            else:
                this_clust.append(phrases_list[glob])

        phrases_and_distance = {'Phrases': this_clust, 'Distance': round(distance, 3)}
        clusters[cluster_n] = phrases_and_distance
    return clusters


def main():
    # Opening JSON file:
    f = open('wiki_data_out.json', )

    # returns JSON object as a dictionary:
    data = json.load(f)

    # extracts phrases list:
    phrases_list = doc2phrase_restricted(data, 200)
    # phrases_list = doc2phrase(data)

    # create sentence embeddings and arrange them in DataFrame:
    df_Universal = df_model_embed(model_Universal, phrases_list, phrases_list)

    # Agglomerative clustering - compute linkage matrix from sentences embeddings:
    Z = shc.linkage(df_Universal, method='average', metric='cosine')

    # plot distance (including first and sec derivatives) as function of level of linkage matrix:
    # distance_criterion = plot_dist_on_level(Z)

    # create clusters sorted according to distance from linkage matrix:
    clusters = extract_clusters(Z, phrases_list, 0, 0.4)  # I have a bug here, if I change the lower criterion
                                                            # from '0' to somthing else it gives me an error

    # # save and retrieve --phrases_list--:
    # save_phrases_list = open("data.pkl.phrases_list", "wb")
    # pickle.dump(phrases_list, save_phrases_list)
    # save_phrases_list.close()
    #
    # save_phrases_list = open("data.pkl.phrases_list", "rb")
    # retrieved_phrases_list = pickle.load(save_phrases_list)
    #
    # # save and retrieve --linkage matrix--:
    # save_linkage_matrix = open("data.pkl.linkage", "wb")
    # pickle.dump(Z, save_linkage_matrix)
    # save_linkage_matrix.close()
    #
    # save_linkage_matrix = open("data.pkl.linkage", "rb")
    # retrieved_linkage_matrix = pickle.load(save_linkage_matrix)
    #
    # # save and retrieve --cluster's dictionary--:
    # save_clusters_dic = open("data.pkl.cluster_dic", "wb")
    # pickle.dump(clusters, save_clusters_dic)
    # save_clusters_dic.close()
    #
    # save_clusters_dic = open("data.pkl.cluster_dic", "rb")
    # retrieved_clusters_dic = pickle.load(save_clusters_dic)
    print()
    print()
    # Closing file:
    f.close()
    print()


if __name__ == "__main__":
    main()