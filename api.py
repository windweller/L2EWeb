import os
import sys
import csv
import time
import json
import argparse
from os.path import join as pjoin
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import logging
import nltk

from seq2seq import PhenomenonEncoder, L2EDecoder
from parser import setup_corenlp, get_parse, Sentence
import spacy
import nltk

import uuid
import pickle

from onmt.model_builder import build_base_model
from onmt import inputters

from pattern.en import conjugate, lemma, lexeme,PRESENT,PAST,PL,SG

setup_corenlp("en")
nlp = spacy.load("en_core_web_sm")

def replacements(parse, subs):
    old_string = subs["old"]
    new_string = subs["new"]
    pos_type = subs["pos"]
    indices_to_replace = [t["index"] for t in parse.new_tokens if t["word"] == old_string and t["pos"][0] == pos_type]
    for i in indices_to_replace:
        parse.new_tokens[i - 1]["word"] = new_string
        parse.new_tokens[i - 2]["after"] = " "
    sentence = "".join(t["word"] + t["after"] for t in parse.new_tokens).replace("  ", " ")
    return sentence, parse


def replace_apostrophes(parse):
    # replace "'s" with "is" where appropriate
    return replacements(parse, {"old": "'s", "new": "is", "pos": "V"})


def get_subj(s, root_index, lemmas):
    if s.token(root_index)["pos"][0] == "V":
        root_verb_index = root_index
        # Find where the subject NP starts.
        # First, find any children of the root where the dep is nsubj or nsubjpass
        # and grap the corresponding dependent.
        subj_head_indices = s.find_children(root_index, filter_types=["nsubj", "nsubjpass"])
        if len(subj_head_indices) > 0:
            subj_head_index = subj_head_indices[0]
        else:
            return ("ERROR: can't transform: missing nsubj")
    else:
        # assume the first noun is subj (this is a hack)
        noun_indices = [t["index"] for t in s.tokens if t["pos"][0] == "N"]
        determiner_indices = [t["index"] for t in s.tokens if t["pos"][0] == "D"]
        noun_indices = noun_indices + determiner_indices
        if len(noun_indices) > 0:
            subj_head_index = min(noun_indices)
        else:
            return ("ERROR: can't transform: weird root")

    # Then, get the index where this phrase begins.
    subj_end_index = max(s.get_subordinate_indices([subj_head_index], [subj_head_index]))
    subj_head = s.token(subj_head_index)["word"]
    subj_is_plural = (subj_head.lower() != lemmas[subj_head.lower()])

    return subj_end_index, subj_is_plural


def conjugate_main_verb(s, do_form, root_index):
    if do_form == "do":
        # lemma is fine
        s.cut(2)
        return s
    else:
        if do_form == "did":
            # past tense
            tense = PAST
            number = SG
        elif do_form == "does":
            # third person singular
            tense = PRESENT
            number = SG
        orig_verb = s.new_tokens[root_index - 1]["word"]
        s.new_tokens[root_index - 1]["word"] = conjugate(verb=orig_verb, tense=tense, number=number)
        # remove "did" or "does"
        s.cut(2)
        return s


def cleanup_statement(q):
    q = str(q)
    q = q.replace(" i ", " I ")
    return q[0].upper() + q[1:] + " ."


def transform_q2s(question):
    #### Input: question $q$,

    # # lower case everything (why?)
    # question = question.lower()

    #### dependency parsed.
    q = Sentence(get_parse(question), question, "en", print_tokens="new")
    doc = nlp(question)
    lemmas = {token.text.lower(): token.lemma_ for token in doc}
    past_tense = {token.text: token.tag_ == "VBD" for token in doc}

    # if question doesn't start with "why", stop there
    if q.word(1).lower() != "why":
        return ("ERROR: can't transform: does not start with 'why'")

    # replace "'s" with "is" and "'d" with "did" where appropriate
    question, q = replacements(q, {"old": "'s", "new": "is", "pos": "V"})
    question, q = replacements(q, {"old": "'re", "new": "are", "pos": "V"})
    question, q = replacements(q, {"old": "'d", "new": "did", "pos": "M"})
    question, q = replacements(q, {"old": "'ve", "new": "have", "pos": "V"})

    # remove the final question mark
    if q.word(len(q.tokens)) == "?":
        q.cut(len(q.tokens))

    # if question starts with "why is it that", return the rest
    if (q.words(1, 5).lower() == "why is it that ") or (q.words(1, 5).lower() == "why was it that "):
        for i in range(4):
            q.cut(1)
        return cleanup_statement(q)

    #### Start at the root of $q$
    all_indices = [t["index"] for t in q.tokens]
    root_index = [i for i in all_indices if "ROOT" in q.find_dep_types(i)][0]

    #### $subj$ = \textsc{nsubj} or \textsc{nsubjpass} %dependent of the root
    subj_end_index, subj_is_plural = get_subj(q, root_index, lemmas)
    #### $vp^{(\text{lemma})}$ = all remaining dependents
    # (everything after subj_end_index is the lemmatized vp)

    #### if $aux$ in [``do'', ``does'', ``did'']
    if (q.word(2).lower() in ["do", "does", "did"]):
        #### $vp$ = apply tense/person of $aux$ to $vp^{(\text{lemma})}$
        q = conjugate_main_verb(q, q.word(2), root_index)
    else:
        #### else $vp$ = $aux$ $vp^{(\text{lemma})}$
        # if it's not "do", just move the single helping word to after the subj NP
        q.move(2, subj_end_index)

    #### $s$ = $subj$ $vp$

    #### Remove "Why".
    q.cut(1)

    return cleanup_statement(q)

model_file_path = "model/dissent_step_80000.pt"
# model_file_path = "model/dissent_step_200000.pt"
temp_dir = "/tmp/"

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

encoder = PhenomenonEncoder(model_file_path, temp_dir, logger)

decoder = L2EDecoder(encoder)

def decode_sent(sent, is_why=False):

    if is_why:
        sent = transform_q2s(sent)
    else:
        doc = nltk.word_tokenize(sent)
        sent = ' '.join([str(w) for w in doc][:-1]) + " ."  # we remove "because".
        #  In the website we ask users to end sentence with "because"
        sent = sent.capitalize()

    decoded_tups = decoder.decode_sentences([sent])
    print(decoded_tups)
    return decoded_tups[0]
