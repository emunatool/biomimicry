import json
import random
import spacy
import re
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import scipy.cluster.hierarchy as shc
import torch
# import nltk


# from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

# Model = Universal Sentence Encoder:
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model_Universal = hub.load(module_url)
print("module %s loaded" % module_url)


# lemmatizer = nltk.WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

# gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#

# def compute_score(phrase):
#     tokens_tensor = gpt2_tokenizer.encode(phrase, add_special_tokens=False, return_tensors="pt")
#     loss=gpt2_model(tokens_tensor, labels=tokens_tensor)[0]
#     score = np.exp(loss.cpu().detach().numpy())
#     return score


def clean_phrases(phrase):
    # lower case text
    newString = phrase.lower()
    # newString = re.sub(r"'s\b", "", newString)

    # remove punctuations
    newString = re.sub("[^a-zA-Z' ']", "", newString)
    newString = re.sub("['  ']", " ", newString)

    # remove specific words
    unwanted_words = ['is', 'a', 'at', 'the', 'any'] # maybe to add 'both' 'in'
    newStringwords = newString.split()

    resultwords = [word for word in newStringwords if word not in unwanted_words]
    result = ' '.join(resultwords)
    return result


# def QAsrl_phrases(data_json, num_random):
#     phrase_list = []
#     data_json_indx = range(len(data_json))
#     data_json_indx_list = random.choices(data_json_indx, k=num_random)
#     # random_data = random.choices(data_json, k=num_random)
#     print(data_json_indx_list)
#     for dict_num in data_json_indx_list:
#         dict = data_json[dict_num]
#         sentence = dict['words']
#         # word tokenizeing and part-of-speech tagger
#         document = ' '.join(sentence)
#
#         # doc = nlp(document)
#         print('doc_num: ', dict_num)
#         print('\n', document)
#
#         verbs = dict['verbs']
#         print('\n__From QAsrl__:')
#         for verb in verbs:
#             print('\n*The verb*: ', verb['verb'])
#             for qa in verb['qa_pairs']:
#                 question = (qa['question']).split()
#                 wh_question =question[0].lower()
#                 if wh_question != 'when' or 'where' or 'who':
#                     print('\nQ: ', qa['question'])
#                     for span in qa['spans']:
#                         print('Ans: ', span['text'])
#                         phrase_list.append(lemmatizer.lemmatize(verb['verb'], pos="v") + ' ' + span['text'])
#
#
#         # phrase_list = depMatch_match_and_print(doc)
#         print('###########################################################################')
#     phrase_list = list(set(phrase_list))
#     pass


# def QAsrl_analyze_doc(data_json, dict_num):
#     phrase_list = []
#
#     dict = data_json[dict_num]
#     sentence = dict['words']
#     # word tokenizeing and part-of-speech tagger
#     document = ' '.join(sentence)
#
#     # doc = nlp(document)
#     print('doc_num: ', dict_num)
#     print('\n', document)
#
#     verbs = dict['verbs']
#     # print the QAsrl data:
#     print('\n__From QAsrl__:')
#     for verb in verbs:
#         print('\n*The verb*: ', verb['verb'])
#         for qa in verb['qa_pairs']:
#             print('\nQ: ', qa['question'])
#             for span in qa['spans']:
#                 print('Ans: ', span['text'])
#
#     # extract the phrase list:
#     for verb in verbs:
#         for qa in verb['qa_pairs']:
#             question = (qa['question']).split()
#             wh_question =question[0].lower()
#             if wh_question not in ['when', 'where', 'who']:
#             # if wh_question not in ['when', 'who']:
#                 the_verb = nlp(verb['verb'])[0].lemma_
#                 last_token = nlp(question[-1])[0].lemma_
#                 # the_verb = verb_lemma + '?'
#                 # last_token = question[-1]
#                 if the_verb == last_token:
#                     verb_lemma = nlp(verb['verb'])[0].lemma_
#                     qa_first_span = qa['spans'][0]['text']
#                     # remove 'that the'
#                     qa_first_span_tokenaized = qa_first_span.split()
#                     if len(qa_first_span_tokenaized) < 2:
#                         if nlp(qa_first_span_tokenaized[0])[0].pos_ == 'NOUN':
#                             # clean_phrase = clean_phrases(verb_lemma + ' ' + qa_first_span)
#                             # phrase_list.append(clean_phrase)
#                             phrase_list.append(verb_lemma + ' ' + qa_first_span)
#                     elif qa_first_span_tokenaized[0] + ' ' + qa_first_span_tokenaized[1] != 'that the':
#                         # clean_phrase = clean_phrases(verb_lemma + ' ' + qa_first_span)
#                         # phrase_list.append(clean_phrase)
#                         phrase_list.append(verb_lemma + ' ' + qa_first_span)
#
#                     # for span in qa['spans']:
#                     #     verb_lemma = nlp(verb['verb'])[0].lemma_
#                     #     phrase_list.append(verb_lemma + ' ' + span['text'])
#
#     ## need to perform clean_phrases function!
#     ## I set the below prints to be comments as it might slow down the running
#     print('---------------------------------------------------------------------')
#     # phrase_list = list(set(phrase_list)) # remove duplicates
#     print("Extracted phrases:\n")
#     if len(phrase_list) == 0:
#         print("No phrases")
#     else:
#         [print(clean_phrases(phrase), compute_score(phrase)) for phrase in phrase_list]
#     print('#####################################################################')
#     clean_phrases_list = [clean_phrases(phrase) for phrase in phrase_list]
#     return clean_phrases_list


def get_phrases_from_doc(data_json, dict_num):
    phrase_list = []

    dict = data_json[dict_num]
    print('doc_num: ', dict_num)
    verbs = dict['verbs']
    # extract the phrase list:
    for verb in verbs:
        for qa in verb['qa_pairs']:
            question = (qa['question']).split()
            wh_question = question[0].lower()
            if wh_question not in ['when', 'where', 'who']:
                # if wh_question not in ['when', 'who']:
                the_verb = nlp(verb['verb'])[0].lemma_
                last_token = nlp(question[-1])[0].lemma_
                # the_verb = verb_lemma + '?'
                # last_token = question[-1]
                if the_verb == last_token:
                    # verb_lemma = nlp(verb['verb'])[0].lower_
                    verb_lemma = nlp(verb['verb'])[0].lemma_
                    qa_first_span = qa['spans'][0]['text']
                    # remove 'that the'
                    qa_first_span_tokenaized = qa_first_span.split()
                    if len(qa_first_span_tokenaized) < 2:
                        if nlp(qa_first_span_tokenaized[0])[0].pos_ == 'NOUN':
                            # clean_phrase = clean_phrases(verb_lemma + ' ' + qa_first_span)
                            # phrase_list.append(clean_phrase)
                            phrase_list.append(verb_lemma + ' ' + qa_first_span)
                    elif qa_first_span_tokenaized[0] + ' ' + qa_first_span_tokenaized[1] != 'that the':
                        # clean_phrase = clean_phrases(verb_lemma + ' ' + qa_first_span)
                        # phrase_list.append(clean_phrase)
                        phrase_list.append(verb_lemma + ' ' + qa_first_span)

    clean_phrases_list = [clean_phrases(phrase) for phrase in phrase_list]
    return clean_phrases_list


def collecting_all_phrases(data_json):
    all_phrases = []
    QAsrl_docNphrases = {}
    data_json_indx = range(len(data_json))
    for doc in data_json_indx:
        doc_phrase_list = get_phrases_from_doc(data_json, doc)
        all_phrases += doc_phrase_list
        # all_phrases.append(get_phrases_from_doc(data_json, doc))
        QAsrl_docNphrases[doc] = {'doc_num': doc, 'phrases': doc_phrase_list}
    all_phrases = list(set(all_phrases))  # remove duplicates
    return all_phrases, QAsrl_docNphrases


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

def main():
    # Opening JSON file:
    f = open('wiki_data_out.json', )

    # returns JSON object as a dictionary:
    data = json.load(f)
    # extracts phrases list:
    # phrases_list = doc2phrase_random(data, 20)

    # QAsrl_phrases(data, 1)
    # phrase_list = QAsrl_analyze_doc(data, 1057)
    # phrase_list = QAsrl_analyze_doc(data, 48)
    # phrase_list = QAsrl_analyze_doc(data, 3960)
    # phrase_list = QAsrl_analyze_doc(data, 3989)
    # phrase_list = QAsrl_analyze_doc(data, 8723)
    # phrase_list = QAsrl_analyze_doc(data, 16992)
    # phrase_list = QAsrl_analyze_doc(data, 7759)
    # phrase_list = QAsrl_analyze_doc(data, 17213)
    # phrase_list = QAsrl_analyze_doc(data, 20173)
    # phrase_list = QAsrl_analyze_doc(data, 22129)
    # phrase_list = QAsrl_analyze_doc(data, 10875)
    # phrase_list = QAsrl_analyze_doc(data, 2494)
    # phrase_list = QAsrl_analyze_doc(data, 18340)
    # phrase_list = QAsrl_analyze_doc(data, 13095)
    # phrase_list = QAsrl_analyze_doc(data, 16072)

    # all_phrases_list = collecting_all_phrases(data)
    #
    # # save and retrieve --phrases_list--:
    # save_phrases_list = open("data.pkl.QAsrl_docNphrases", "wb")
    # pickle.dump(all_phrases_list, save_phrases_list)
    # save_phrases_list.close()

    save_phrases_list = open("data.pkl.QAsrl_docNphrases", "rb")
    retrieved_phrases_list = pickle.load(save_phrases_list)
    full_phrases_list = retrieved_phrases_list[0]
    print()
    # create sentence embeddings and arrange them in DataFrame:
    df_Universal = df_model_embed(model_Universal, full_phrases_list, full_phrases_list)
    print("dataframe is done")

    # Agglomerative clustering - compute linkage matrix from sentences embeddings:
    Z = shc.linkage(df_Universal, method='average', metric='cosine')
    print("linkage matrix is done")

    # save and retrieve --linkage matrix--:
    save_linkage_matrix = open("data.pkl.QAsrl_linkage", "wb")
    pickle.dump(Z, save_linkage_matrix)
    save_linkage_matrix.close()

    print()

    # Closing file:
    f.close()
    print()


if __name__ == "__main__":
    main()