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
# import tensorflow_hub as hub
# import tensorflow as tf

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
# from sentence_transformers import SentenceTransformer
# from models import InferSent
# import torch
import pickle
import random

nlp = spacy.load("en_core_web_sm")

# # Model = Universal Sentence Encoder:
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# model_Universal = hub.load(module_url)
# print("module %s loaded" % module_url)


# def dependency_matcher():
#     pattern = [
#         {
#             "RIGHT_ID": "OBJECT",
#             "RIGHT_ATTRS": {"POS": "VERB"}
#         },
#         {
#             "LEFT_ID": "OBJECT",
#             "REL_OP": ">",
#             "RIGHT_ID": "PREDICATE",
#             "RIGHT_ATTRS": {"DEP": "dobj"},
#         },
#         {
#             "LEFT_ID": "PREDICATE",
#             "REL_OP": ">",
#             "RIGHT_ID": "PREDICATE_modifier",
#             "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}},
#         }
#     ]
#
#     pattern_sec = [
#         {
#             "RIGHT_ID": "OBJECT",
#             "RIGHT_ATTRS": {"POS": "VERB"}
#         },
#         {
#             "LEFT_ID": "OBJECT",
#             "REL_OP": ">",
#             "RIGHT_ID": "PREDICATE",
#             "RIGHT_ATTRS": {"DEP": "dobj"},
#         }
#     ]
#
#     pattern_third = [
#         {
#             "RIGHT_ID": "OBJECT",
#             "RIGHT_ATTRS": {"POS": "VERB"}
#         },
#         {
#             "LEFT_ID": "OBJECT",
#             "REL_OP": ">",
#             "RIGHT_ID": "PREDICATE",
#             "RIGHT_ATTRS": {"DEP": "dobj"},
#         },
#         {
#             "LEFT_ID": "PREDICATE",
#             "REL_OP": ">",
#             "RIGHT_ID": "PREDICATE_modifier",
#             "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}},
#         },
#         {
#             "LEFT_ID": "PREDICATE_modifier",
#             "REL_OP": ".",
#             "RIGHT_ID": "PREDICATE_modifier_modifier",
#             "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}},
#         }
#     ]
#
#     matcher.add("OBJECT", [pattern])
#     matcher.add("OBJECT", [pattern_sec])
#     matcher.add("OBJECT", [pattern_third])
#     return matcher


def dependency_matcher():
    # matcher = DependencyMatcher(nlp.vocab, validate=True)
    matcher = DependencyMatcher(nlp.vocab)

    OBJECT_Dobj = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "Dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"},
        }
    ]

    # OBJECT_middle_Dobj = [
    #     {
    #         "RIGHT_ID": "OBJECT",
    #         # "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "NOUN"]}}
    #         "RIGHT_ATTRS": {"POS": "VERB"}
    #
    #     },
    #     {
    #         "LEFT_ID": "OBJECT",
    #         "REL_OP": ">>",
    #         "RIGHT_ID": "Dobj",
    #         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj", "iobj"]}},
    #     },
    #     {
    #         "LEFT_ID": "OBJECT",
    #         "REL_OP": ".",
    #         "RIGHT_ID": "mod/comp",
    #         "RIGHT_ATTRS": {"DEP": {"IN": ["xcomp", "amod", "compound"]}}
    #         # "RIGHT_ATTRS": {"DEP": {"IN": ["xcomp", "amod", "compound", "prep"]}}
    #     }
    # ]
    OBJECT_middle_Dobj = [
        {
            "RIGHT_ID": "OBJECT",
            # "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "NOUN"]}}
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "Dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"},
        },
        {
            "LEFT_ID": "Dobj",
            "REL_OP": ">",
            "RIGHT_ID": "mod/comp",
            "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}}
            # "RIGHT_ATTRS": {"DEP": {"IN": ["xcomp", "amod", "compound"]}}
        }
    ]

    OBJECT_middle_Dobj_opposite = [
        {
            "RIGHT_ID": "OBJECT",
            # "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "NOUN"]}}
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "Dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"},
            # "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj", "iobj"]}},
        },
        {
            "LEFT_ID": "Dobj",
            "REL_OP": "<",
            "RIGHT_ID": "mod/comp",
            "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}}
            # "RIGHT_ATTRS": {"DEP": {"IN": ["xcomp", "amod", "compound"]}}
        }
    ]

    # OBJECT_prep_Dobj = [
    #     {
    #         "RIGHT_ID": "Dobj",
    #         # "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "NOUN"]}}
    #         "RIGHT_ATTRS":  {"DEP": {"IN": ["dobj", "pobj", "iobj"]}}
    #
    #     },
    #     {
    #         "LEFT_ID": "Dobj",
    #         "REL_OP": "<",
    #         "RIGHT_ID": "prep",
    #         "RIGHT_ATTRS": {"DEP": "prep"},
    #     },
    #     {
    #         "LEFT_ID": "prep",
    #         "REL_OP": ".",
    #         "RIGHT_ID": "OBJECT",
    #         "RIGHT_ATTRS": {"POS": "VERB"}
    #     }
    # ]

    # OBJECT_prep_Dobj = [
    #     {
    #         "RIGHT_ID": "OBJECT",
    #         # "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "NOUN"]}}
    #         "RIGHT_ATTRS": {"POS": "VERB"}
    #     },
    #     {
    #         "LEFT_ID": "OBJECT",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "Prep",
    #         "RIGHT_ATTRS": {"DEP": "prep"},
    #     },
    #     {
    #         "LEFT_ID": "Prep",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "Dobj",
    #         "RIGHT_ATTRS": {"DEP": "pobj"}
    #     }
    # ]

    # OBJECT_prep_Dobj = [
    #     {
    #         "RIGHT_ID": "OBJECT",
    #         # "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "NOUN"]}}
    #         "RIGHT_ATTRS": {"POS": "VERB"}
    #
    #     },
    #     {
    #         "LEFT_ID": "OBJECT",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "pobj",
    #         "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "dobj"]}},
    #         # "RIGHT_ATTRS": {"DEP": "pobj"},
    #     },
    #     {
    #         "LEFT_ID": "pobj",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "prep",
    #         "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "xcomp"]}}
    #         # "RIGHT_ATTRS": {"DEP": "prep"}
    #         # "RIGHT_ATTRS": {"DEP": {"IN": ["xcomp", "amod", "compound"]}}
    #     }
    # ]
    #
    # OBJECT_prep_Dobj_opposite = [
    #     {
    #         "RIGHT_ID": "OBJECT",
    #         # "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "NOUN"]}}
    #         "RIGHT_ATTRS": {"POS": "VERB"}
    #
    #     },
    #     {
    #         "LEFT_ID": "OBJECT",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "pobj",
    #         "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "dobj"]}},
    #         # "RIGHT_ATTRS": {"DEP": "pobj"},
    #     },
    #     {
    #         "LEFT_ID": "pobj",
    #         "REL_OP": "<",
    #         "RIGHT_ID": "prep",
    #         "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "xcomp"]}}
    #         # "RIGHT_ATTRS": {"DEP": "prep"
    #     }
    # ]
    OBJECT_prep_Dobj = [
        {
            "RIGHT_ID": "OBJECT",
            # "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "NOUN"]}}
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "prep",
            # "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "xcomp"]}},
            "RIGHT_ATTRS": {"DEP": "prep"},
        },
        {
            "LEFT_ID": "prep",
            "REL_OP": ".",
            "RIGHT_ID": "pobj",
            # "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "dobj"]}}
            "RIGHT_ATTRS": {"DEP": "pobj"}
        }
    ]

    OBJECT_prep_Dobj_opposite = [
        {
            "RIGHT_ID": "OBJECT",
            # "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "NOUN"]}}
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "prep",
            # "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "xcomp"]}},
            "RIGHT_ATTRS": {"DEP": "prep"},
        },
        {
            "LEFT_ID": "prep",
            "REL_OP": ".",
            "RIGHT_ID": "pobj",
            # "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "dobj"]}}
            "RIGHT_ATTRS": {"DEP": "pobj"}

        }
    ]

    aux_OBJECT_Dobj = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "Dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"},
            # "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj", "iobj"]}},

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ".",
            "RIGHT_ID": "aux verb",
            "RIGHT_ATTRS": {"DEP": "aux", "POS": "VERB"}
        }
    ]

    with_agent = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "AGENT",
            "RIGHT_ATTRS": {"DEP": "agent"},
        },
        {
            "LEFT_ID": "AGENT",
            "REL_OP": ">",
            "RIGHT_ID": "OBJ",
            "RIGHT_ATTRS": {"DEP": "pobj", "POS": "NOUN"},
        }
    ]

    matcher.add("OBJECT", [OBJECT_Dobj])
    matcher.add("OBJECT", [OBJECT_middle_Dobj])
    matcher.add("OBJECT", [OBJECT_middle_Dobj_opposite])
    matcher.add("OBJECT", [aux_OBJECT_Dobj])
    # matcher.add("OBJECT", [with_agent])
    # matcher.add("OBJECT", [OBJECT_prep_Dobj])
    # matcher.add("OBJECT", [OBJECT_prep_Dobj_opposite])

    return matcher


def depMatch_match_and_print(doc):
    phrase_list = []

    matcher = dependency_matcher()
    matches = matcher(doc)
    # print("matches: ", len(matches))  # [(4851363122962674176, [6, 0, 10, 9])]
    # Each token_id corresponds to one pattern dict
    match_list = []
    [match_list.append(matches[match][1]) for match in range(len(matches))]
    match_list = [list(x) for x in set(frozenset(i) for i in [set(i) for i in match_list])]
    # print(match_list)

    print('\n++Dependency Phrases++: ')
    if len(matches) == 0:
        print('No Phrases')
    for match in match_list:
        # if len(matches) != 0:
        token_ids = match
        token_ids.sort()
        phrase = []
        for token_id in token_ids:
            aaaaaa = doc[token_id] # used to examine the properties of each token
            if doc[token_id].pos_ == "VERB":
                if (doc[token_id].suffix_ == "ing") and (token_id != token_ids[0]):
                    phrase.append(doc[token_id].lower_)
                elif doc[token_id].right_edge.pos_ == "SYM":
                    break
                else:
                    phrase.append(doc[token_id].lemma_)
            elif doc[token_id].pos_ == "PRON":
                phrase.append("something")
            elif doc[token_id].right_edge.pos_ == "SYM":
                break
            else:
                phrase.append(doc[token_id].lower_)

        # [phrase.append(doc[token_id].lemma_) for token_id in token_ids]
        if len(phrase) == 0:
            break
        else:
            phrase = ' '.join(phrase)
            phrase_list.append(phrase)
            print(phrase)
    return phrase_list



# def doc2phrase_random(data_json, num_random):
#     phrase_list = []
#     random_data = random.choices(data_json, k=num_random)
#
#     for dict in random_data:
#         sentence = dict['words']
#         # word tokenizeing and part-of-speech tagger
#         document = ' '.join(sentence)
#
#         print('\n', doc)
#
#         verbs = dict['verbs']
#         print('\nFrom QAsrl:')
#         for verb in verbs:
#             print('\nThe verb: ', verb['verb'])
#             for qa in verb['qa_pairs']:
#                 print('\nQ: ', qa['question'])
#                 for span in qa['spans']:
#                     print('Ans: ', span['text'])
#
#         # displacy.serve(doc, style="dep")  # http://localhost:5000/ copy to web page to see the visualization
#         matcher = dependency_matcher()
#         matches = matcher(doc)
#         # print("matches: ", len(matches))  # [(4851363122962674176, [6, 0, 10, 9])]
#         # Each token_id corresponds to one pattern dict
#         match_list = []
#         [match_list.append(matches[match][1]) for match in range(len(matches))]
#         match_list = [list(x) for x in set(frozenset(i) for i in [set(i) for i in match_list])]
#         # print(match_list)
#
#         print('\nDependency Phrases: ')
#         if len(matches) == 0:
#             print('No Phrases')
#         for match in match_list:
#             # if len(matches) != 0:
#             token_ids = match
#             token_ids.sort()
#             phrase = []
#             [phrase.append(doc[token_id].lemma_) for token_id in token_ids]
#             phrase = ' '.join(phrase)
#             phrase_list.append(phrase)
#             print(phrase)
#         # print(len(phrase_list))
#         print('#######################################################################################################')
#     phrase_list = list(set(phrase_list))
#     return phrase_list


def doc2phrase_random(data_json, num_random):
    """
    use to find cues in the QAsrl data to help me understand if the phrases generated with the algorithm of dependency
    matcher are suitable.
    choose randomly number of random sentences from the 'wiki_data' dictionary ('num_random'), and print the relevant
    data from the QAsrl and the phrses created from the algorithm of dependency parser which I created.
    :param data_json:
    :param num_random: number of random sentences to draw from 'wiki_data' dictionary.
    :return:
    """
    phrase_list = []
    data_json_indx = range(len(data_json))
    data_json_indx_list = random.choices(data_json_indx, k=num_random)
    # random_data = random.choices(data_json, k=num_random)
    print(data_json_indx_list)
    for dict_num in data_json_indx_list:
        dict = data_json[dict_num]
        sentence = dict['words']
        # word tokenizeing and part-of-speech tagger
        document = ' '.join(sentence)

        doc = nlp(document)
        print('doc_num: ', dict_num)
        print('\n', doc)

        verbs = dict['verbs']
        print('\n__From QAsrl__:')
        for verb in verbs:
            print('\n*The verb*: ', verb['verb'])
            for qa in verb['qa_pairs']:
                print('\nQ: ', qa['question'])
                for span in qa['spans']:
                    print('Ans: ', span['text'])

        # displacy.serve(doc, style="dep")  # http://localhost:5000/ copy to web page to see the visualization

        # print(len(phrase_list))
        phrase_list = depMatch_match_and_print(doc)
        print('###########################################################################')

    phrase_list = list(set(phrase_list))
    return phrase_list


def analyze_doc(data_json, doc_num):
    """
    use to analyze specific doc from "wiki_data" - the QAsrl file from the BARCODE's data
    the goal is to understand if the phrases generated from the dependancy parser are relevet to target the problem
    from the 'wiki_data' dataset.
    :param data_json: 'wiki_data' json file
    :param doc_num: the nubmer of the key of the dictionary of 'wiki_data'
    :return:
    """

    dict = data_json[doc_num]
    sentence = dict['words']
    # word tokenizeing and part-of-speech tagger
    document = ' '.join(sentence)

    doc = nlp(document)
    print('doc_num: ', doc_num)
    print('\n', doc)

    verbs = dict['verbs']
    print('\n__From QAsrl__:')
    for verb in verbs:
        print('\n*The verb*: ', verb['verb'])
        for qa in verb['qa_pairs']:
            print('\nQ: ', qa['question'])
            for span in qa['spans']:
                print('Ans: ', span['text'])

    # displacy.serve(doc, style="dep")  # http://localhost:5000/ copy to web page to see the visualization
    phrase_list = depMatch_match_and_print(doc)
    print('###########################################################################\n')

    phrase_list = list(set(phrase_list))
    return phrase_list


def main():
    # Opening JSON file:
    f = open('wiki_data_out.json', )

    # returns JSON object as a dictionary:
    data = json.load(f)
    # extracts phrases list:
    phrases_list = doc2phrase_random(data, 20)

    # phrases_of_doc = analyze_doc(data, 48)
    # phrases_of_doc = analyze_doc(data, 3960)
    # phrases_of_doc = analyze_doc(data, 3989)
    # phrases_of_doc = analyze_doc(data, 8723)
    # phrases_of_doc = analyze_doc(data, 2494)  # not optimal results
    # phrases_of_doc = analyze_doc(data, 16992)  # not good
    # phrases_of_doc = analyze_doc(data, 7759)
    # phrases_of_doc = analyze_doc(data, 17213)  # no phrases TN
    # phrases_of_doc = analyze_doc(data, 20173)
    # phrases_of_doc = analyze_doc(data, 22129)
    # phrases_of_doc = analyze_doc(data, 10875)
    # phrases_of_doc = analyze_doc(data, 18340)
    # phrases_of_doc = analyze_doc(data, 2564)
    # phrases_of_doc = analyze_doc(data, 13095)
    # phrases_of_doc = analyze_doc(data, 16072)
    # phrases_of_doc = analyze_doc(data, 12192)
    # phrases_of_doc = analyze_doc(data, 7298)
    # phrases_of_doc = analyze_doc(data, 2919)
    # phrases_of_doc = analyze_doc(data, 17724)
    # phrases_of_doc = analyze_doc(data, 770)





    # phrases_of_doc = analyze_doc(data, 48)

    # Closing file:
    f.close()
    print()


if __name__ == "__main__":
    main()