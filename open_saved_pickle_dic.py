import pickle
import json
import numpy as np
import random
import scipy.cluster.hierarchy as shc
import pandas as pd


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

        phrases_and_distance = {'Phrases': this_clust, 'Distance': distance}
        clusters[cluster_n] = phrases_and_distance
    return clusters


def print_clusters_LastToFirst(cluster_dic, steps, precent_of_dic):
    dic_keys = list(cluster_dic.keys())
    dic_keys_LastToFirst = dic_keys[::-1]
    dic_keys_LastToFirst_with_steps = dic_keys_LastToFirst[0:round(precent_of_dic*len(dic_keys_LastToFirst)):steps]
    clusters_diluted = {}
    for cluster in dic_keys_LastToFirst_with_steps:
        print('Distance: ', cluster_dic[cluster]['Distance'])
        print('Phrases: ', cluster_dic[cluster]['Phrases'])
        clusters_diluted[cluster_dic[cluster]['Distance']] = cluster_dic[cluster]['Phrases']
    return clusters_diluted


def create_fcluster(linkage_matrix, distance_criterion):
    T = shc.fcluster(linkage_matrix, t=distance_criterion, criterion='distance')
    return T


def dup_list_fcluster(T_fcluster, phrases_list):
    T_original = T_fcluster.tolist()
    T_tosort = T_fcluster
    T_tosort.sort()

    new_list = sorted(set(T_tosort))
    dup_list = []

    for i in range(len(new_list)):
        if (T_tosort.tolist()).count(new_list[i]) > 1:
            dup_list.append(new_list[i])

    clusterlists = dict()
    num_of_cluster = 1
    for cluster in dup_list:
        indices = np.where(T_original == cluster)
        list = [phrases_list[index] for index in indices[0]]
        clusterlists[str(num_of_cluster)] = [list]
        num_of_cluster += 1
    return clusterlists


def print_clusters_LastToFirst_length(cluster_dic):
    """
    to examine is the clusters made by "fcluster" are making sense. check if the criterion is right.
    take dictionary that contains the clusters and turn it into a dataframe, where the clusters are ordered according to their length
    (number of phrases in each cluster). the aim is to see if the clusters make sense, that eans if the phrases are actually
    close to each other in their meaning.
    :param cluster_dic:
    :return:
    """
    # dic_keys = list(cluster_dic.keys())
    # dic_keys_length = []
    # [dic_keys_length.append(len(cluster_dic[0])) for key in dic_keys]
    heads = ['Cluster_length', 'Cluster_id', 'Phrases']
    df = pd.DataFrame(columns=heads)
    columns = list(df)
    data = []
    # new_dict = {}
    for d in cluster_dic:
        solo_dic = cluster_dic[d][0]
        length_solo_dic = len(solo_dic)
        values = [length_solo_dic, d, solo_dic]
        zipped = zip(columns, values)
        a_dictionary = dict(zipped)
        data.append(a_dictionary)
        # new_dict[length_solo_dic] = solo_dic
        # d = {'Cluster_length': length_solo_dic, 'Cluster_id': d, 'Phrases': solo_dic}
        # df = pd.DataFrame(data=d)
    df = df.append(data, True)
    df_sorted = df.sort_values(by='Cluster_length', ascending=False)
    return df_sorted


def main():
    # Opening JSON file:
    f = open('wiki_data_out.json', )

    # returns JSON object as a dictionary:
    data = json.load(f)
    # list_data = list(data)
    # data0 = random.choice(list_data)
    data0 = random.choices(data, k=3)

    save_phrases_list = open("data.pkl.phrases_list", "rb")
    retrieved_phrases_list = pickle.load(save_phrases_list)

    save_linkage_matrix = open("data.pkl.linkage", "rb")
    retrieved_linkage_matrix = pickle.load(save_linkage_matrix)

    save_clusters_dic = open("data.pkl.cluster_dic", "rb")
    retrieved_clusters_dic = pickle.load(save_clusters_dic)

    # df_sorted = print_clusters_LastToFirst_length(retrieved_clusters_dic)
    T_fcluster = create_fcluster(retrieved_linkage_matrix, 0.2)
    clusterlists = dup_list_fcluster(T_fcluster, retrieved_phrases_list)

    # T_fcluster_0_3 = create_fcluster(retrieved_linkage_matrix, 0.3)
    # clusterlists_0_3 = dup_list_fcluster(T_fcluster_0_3, retrieved_phrases_list)

    df_sorted = print_clusters_LastToFirst_length(clusterlists)

    # clusters = extract_clusters(retrieved_linkage_matrix, retrieved_phrases_list, 0, 0.4)
    # clusters_diluted = print_clusters_LastToFirst(clusters, steps=10, precent_of_dic=0.8)

    print()

if __name__ == "__main__":
    main()

# cluster_last = retrieved_clusters_dic[last_key]
# cluster_last_phrases = cluster_last['Phrases']
# cluster_last_distance = cluster_last['Distance']
# print(len(retrieved_clusters_dic))
print()