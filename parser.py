# -*- coding: utf-8 -*-

"""
Will read all files from a directory, map to discourse markers
initiate dependency parsing
This creates (s1, s2, marker) by each marker
"""

import numpy as np
import argparse
import io
import pickle
import requests
import re
import logging

import pprint

pp = pprint.PrettyPrinter(indent=1)


import sys

import platform


from os.path import join as pjoin

import json

from copy import deepcopy as cp

EN_PORT = 12345

np.random.seed(123)

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

# for key in logging.Logger.manager.loggerDict:
#	print(key)

logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)

argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
argparser.add_argument("--lang", type=str, default='en', help="en|ch|es")

# parser = argparse.ArgumentParser(description='Split by discourse marker using dependency patterns')
# parser.add_argument("--lang", type=str, default="en", help="en|ch|es")
# args, _ = parser.parse_known_args()

logging.getLogger('requests').setLevel(logging.CRITICAL)

PUNCTUATION = u'.,:;—'

# dependency_patterns = None

"""
Given (p, s), previous sentence and current sentence
1. For each pair, parse s with corenlp, get json with dependency parse
"""


def setup_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def capitalize(s):
    return s[0].capitalize() + s[1:]


def cleanup(s, lang="en"):
    s = s.replace(" @-@ ", "-")
    if len(s) > 0:
        s = capitalize(s)
    if lang == "en":
        s = s.replace(" i ", " I ")
        s = s.replace(" im ", " I'm ")
        if len(s) > 3:
            if s[0:3] == "im ":
                s = "I'm " + s[3:]
        s = re.sub(' " (.*) " ', ' "\\1" ', s)
    # elif lang == "sp":
    #    s = s.replace(" del ", " de el ")
    return s


def standardize_sentence_output(s, lang="en"):
    if len(s) == 0:
        return None
    else:
        s = s.strip()
        s = capitalize(s)

        if lang == "sp":
            # remove all punctuation
            for p in PUNCTUATION:
                # print s
                s = s.replace(p, "")
            return s
        else:
            # strip original final punctuation
            while s[-1] in (PUNCTUATION + '"\''):

                # keep final quotations and don't add an additional .
                if len(s) > 3 and s[-3:] in ", ', \". \". '":
                    return s
                else:
                    # otherwise, strip ending punct
                    s = s[:-1].strip()
                    if len(s) == 0:
                        return None
            # add new standard . at end of sentence
            return s + " ."


# this was chosen for english, but it's probably fine for other languages, too
# basically, if there are multiple discourse markers,
# throw them all out and just keep the sentence they attach to
top_level_deps_to_ignore_if_extra = [
    ("mark", "IN"),
    # ("advmod", "WRB") ## this would solve several minor problems but introduce a few major problems
]


def is_verb_tag(tag):
    return tag[0] == "V" and not tag[-2:] in ["BG", "BN"]


# https://stackoverflow.com/a/18669080
def get_indices(lst, element, case="sensitive"):
    result = []
    starting_index = -1
    while True:
        try:
            found_index = lst.index(element, starting_index + 1)
            starting_index = found_index
        except ValueError:
            return result
        result.append(found_index)


def get_nearest(lst, element):
    distances = [abs(e - element) for e in lst]
    return lst[np.argmin(distances)]


def redo_tokenization(lst, lang="en"):
    if lang == "en":
        s = " ".join(lst)
        separated_s = re.sub(" @([^ ]+)@ ", " @ \1 @ ", s)
        separated_s = re.sub(" 't", " ' t", separated_s)
        separated_s = re.sub(' "', ' " ', separated_s)
        separated_s = re.sub('" ', ' " ', separated_s)
        return separated_s.split()
    else:
        return lst


"""
parsed tokenization is different from original tokenization.
try to re-align and extract the correct words given the
extraction_indices (which are 1-indexed into parsed_words)
fix me to catch more cases?
"""


def extract_subphrase(orig_words, parsed_words, extraction_indices, lang="en"):
    extraction_indices = [i - 1 for i in extraction_indices]

    if lang == "sp":
        return " ".join([parsed_words[i] for i in extraction_indices])

    orig_words = redo_tokenization(orig_words, lang=lang)

    if len(orig_words) == len(parsed_words):
        return " ".join([orig_words[i] for i in extraction_indices])
    else:
        first_parse_index = extraction_indices[0]
        first_word_indices = get_indices(orig_words, parsed_words[first_parse_index])

        last_parse_index = extraction_indices[-1]

        last_word_indices = get_indices(orig_words, parsed_words[last_parse_index])

        if len(first_word_indices) > 0 and len(last_word_indices) > 0:
            first_orig_index = get_nearest(first_word_indices, first_parse_index)
            last_orig_index = get_nearest(last_word_indices, last_parse_index)
            if last_orig_index - first_orig_index == last_parse_index - first_parse_index:
                # maybe it's just shifted
                shift = first_orig_index - first_parse_index
                extraction_indices = [i + shift for i in extraction_indices]
                return " ".join([orig_words[i] for i in extraction_indices])
            else:
                # or maybe there's funny stuff happening inside the subphrase
                # in which case T-T
                print("wonky subphrase:")
                print(orig_words)
                print(parsed_words)
                return None
        else:
            if len(first_word_indices) > 0 and abs(last_parse_index - len(parsed_words)) < 3:
                # the end of the sentence is always weird. assume it's aligned

                # grab the start of the subphrase
                first_orig_index = get_nearest(first_word_indices, first_parse_index)
                # shift if necessary
                shift = first_orig_index - first_parse_index
                extraction_indices = [i + shift for i in extraction_indices]

                if len(orig_words) > extraction_indices[-1]:
                    # extend to the end of the sentence if we're not already there
                    extraction_indices += range(extraction_indices[-1] + 1, len(orig_words))
                else:
                    extraction_indices = [i for i in extraction_indices if i < len(orig_words)]

                return " ".join([orig_words[i] for i in extraction_indices])

            else:
                # or maybe the first and/or last words have been transformed,
                # in which case T-T
                return None


"""
use corenlp server (see https://github.com/erindb/corenlp-ec2-startup)
to parse sentences: tokens, dependency parse
"""


def get_parse(sentence, lang="en", depparse=True):
    if lang == 'en':
        sentence = sentence.replace("'t ", " 't ")
        port = EN_PORT

    if depparse:
        url = "http://localhost:" + str(port) + "?properties={annotators:'tokenize,ssplit,pos,depparse'}"
    else:
        url = "http://localhost:" + str(port) + "?properties={annotators:'tokenize,ssplit,pos'}"

    data = sentence

    parse_string = requests.post(url, data=data).text

    parse_string = parse_string.replace('\r\n', '')
    parse_string = parse_string.replace('\x19', '')

    try:
        parsed_output = json.loads(parse_string)
        sentences = parsed_output["sentences"]
        if len(sentences) > 0:
            return sentences[0]
        else:
            print("error in parse:")
            print(sentences)
            return None

    except ValueError:
        try:
            if lang == "en":
                return json.loads(re.sub("[^A-z0-9.,!:?\"'*&/\{\}\[\]()=+-]", "", parse_string))["sentences"][0]
            elif lang == "sp":
                return json.loads(re.sub("[^áéíóúñÑü¿?¡!ÁÉÍÓÚÜªºA-z0-9.,:\"'*&/\{\}\[\]()=+-]", "", parse_string))[
                    "sentences"][0]
        except:
            print("error loading json:")
            print(sentence)
            return None


class Sentence():
    def __init__(self, json_sentence, original_sentence, lang, print_tokens="original"):
        self.json = json_sentence
        self.dependencies = json_sentence["basicDependencies"]
        self.tokens = json_sentence["tokens"]
        self.original_sentence = original_sentence
        self.lang = lang
        self.new_tokens = self.tokens
        self.print_tokens = print_tokens

    def indices(self, word):
        if len(word.split(" ")) > 1:
            words = word.split(" ")
            indices = [i for lst in [self.indices(w) for w in words] for i in lst]
            return indices
        else:
            # print " ".join([t["word"].lower() for t in self.tokens])
            # print word.lower()
            return [i + 1 for i in get_indices([t["word"].lower() for t in self.tokens], word)]

    def token(self, index):
        return self.tokens[int(index) - 1]

    def word(self, index):
        return self.token(index)["word"]
    
    def words(self, start_index, end_index):
        words_string = ""
        for i in range(start_index, end_index):
            words_string += self.word(i)
            words_string += self.token(i)["after"]
        return words_string

    def find_parents(self, index, filter_types=False, needs_verb=False):
        deps = self.find_deps(index, dir="parents", filter_types=filter_types)

        if needs_verb:
            deps = [d for d in deps if self.gov_is_verb(d)]

        return [d["governor"] for d in deps]

    def find_children(self, index, filter_types=False, exclude_types=False, needs_verb=False,
                      exclude_type_and_POS=False):
        deps = self.find_deps(
            index,
            dir="children",
            filter_types=filter_types,
            exclude_types=exclude_types,
            exclude_type_and_POS=exclude_type_and_POS
        )

        if needs_verb:
            deps = [d for d in deps if self.dep_is_verb(d)]

        # print(deps)
        return [d["dependent"] for d in deps]

    def is_punct(self, index):
        pos = self.token(index)["pos"]
        # return pos in '.,"-RRB--LRB-:;'
        return pos in PUNCTUATION

    def is_verb(self, index):
        pos = self.token(index)["pos"]
        if pos[0] == "V":
            return True
        else:
            cop_relations = self.find_deps(index, dir="children", filter_types="cop")
            has_cop_relation = len(cop_relations) > 0
            if has_cop_relation:
                return True
            else:
                return False

    def gov_is_verb(self, d):
        index = d["governor"]
        return self.is_verb(index)

    def dep_is_verb(self, d):
        index = d["dependent"]
        return self.is_verb(index)

    def find_deps(self, index, dir=None, filter_types=False, exclude_types=False, exclude_type_and_POS=False):
        deps = []
        if dir == "parents" or dir == None:
            deps += [{"dep": d, "index": d['governor']} for d in self.dependencies if d['dependent'] == index]
        if dir == "children" or dir == None:
            deps += [{"dep": d, "index": d['dependent']} for d in self.dependencies if d['governor'] == index]

        if filter_types:
            deps = [d for d in deps if d["dep"]["dep"] in filter_types]
        if exclude_types:
            deps = [d for d in deps if not d["dep"]["dep"] in exclude_types]

        if exclude_type_and_POS:
            deps = [d for d in deps if (d["dep"]["dep"], self.token(d["index"])["pos"]) not in exclude_type_and_POS]

        return [d["dep"] for d in deps]

    def find_dep_types(self, index, dir=None, filter_types=False):
        deps = self.find_deps(index, dir=dir, filter_types=filter_types)
        return [d["dep"] for d in deps]

    def __str__(self):
        if self.print_tokens == "original":
            tokens = self.tokens
        elif self.print_tokens == "new":
            tokens = self.new_tokens
        if self.lang == "ch":
            return "".join([t["word"] for t in tokens])
        else:
            return " ".join([t["word"] for t in tokens])

    def get_subordinate_indices(self, acc, explore, depth=0, exclude_indices=[], exclude_types=[]):
        # print("acc: {}\nexplore: {}\ndepth: {}\nexclude_indices: {}".format(acc, explore, depth, exclude_indices))
        exclude_indices.sort()
        acc.sort()
        explore.sort()

        if depth > 15:
            return None
        # print("exclude: " + " ".join([self.tokens[t_ind-1]["word"] for t_ind in exclude_indices]))
        # print("acc: " + " ".join([self.tokens[t_ind-1]["word"] for t_ind in acc]))
        # print("explore: " + " ".join([self.tokens[t_ind-1]["word"] for t_ind in explore]))
        # print("*****")

        if depth == 0:
            all_children = [c for i in explore for c in self.find_children(i, exclude_types=exclude_types,
                                                                           exclude_type_and_POS=top_level_deps_to_ignore_if_extra)]
        else:
            all_children = [c for i in explore for c in self.find_children(i, exclude_types=exclude_types)]

        # exclude indices
        children = [c for c in all_children if not c in exclude_indices]

        # delete commas before excluded indices
        children = [c for c in children if not (c + 1 in exclude_indices and self.word(c) == ",")]

        # delete commas after excluded indices
        children = [c for c in children if not (c - 1 in exclude_indices and self.word(c) == ",")]

        if len(children) == 0:
            return acc
        else:
            return self.get_subordinate_indices(
                acc=acc + children,
                explore=children,
                depth=depth + 1,
                exclude_indices=exclude_indices,
                exclude_types=exclude_types
            )

    def get_phrase_from_head(self, head_index, exclude_indices=[], exclude_types=[]):

        # given an index,
        # grab every index that's a child of it in the dependency graph
        subordinate_indices = self.get_subordinate_indices(
            acc=[head_index],
            explore=[head_index],
            exclude_indices=exclude_indices,
            exclude_types=exclude_types
        )
        # print subordinate_indices
        # print " ".join([self.word(i) for i in subordinate_indices])

        if not subordinate_indices:
            return None
        subordinate_indices.sort()

        # print subordinate_indices
        # print self
        # exclude any punctuation not followed by a non-punctuation token
        while len(subordinate_indices) > 0 and self.is_punct(subordinate_indices[-1]):
            subordinate_indices = subordinate_indices[:-1]
        if len(subordinate_indices) == 0:
            return None

        # while self.is_punct(subordinate_indices[0]):
        #            subordinate_indices = subordinate_indices[1:]

        if self.lang == "ch":
            subordinate_phrase = "".join([self.word(i) for i in subordinate_indices])
        else:
            # filter so that lengths are nicely behaved, unless it's chinese where tokenization is harder...

            # make string of subordinate phrase from parse
            parse_subordinate_string = " ".join([self.word(i) for i in subordinate_indices])

            # if "ONU" in parse_subordinate_string:
            #     print parse_subordinate_string

            # correct subordinate phrase from parsed version to wikitext version
            # (tokenization systems are different)
            orig_words = self.original_sentence.split()
            parsed_words = [t["word"] for t in self.new_tokens]
            # print subordinate_indices

            # if "estudios" in parse_subordinate_string:
            #     # print orig_words
            #     # print len(orig_words)
            #     # print parsed_words
            #     # print len(parsed_words)
            #     for i in range(len(parsed_words)):
            #         try:
            #             print parsed_words[i],
            #             print orig_words[i]
            #         except:
            #             print parsed_words[i]

            subordinate_phrase = extract_subphrase(orig_words, parsed_words, subordinate_indices, lang="sp")

        # if "ONU" in subordinate_phrase:
        #     print subordinate_phrase

        # make a string from this to return
        if subordinate_phrase:
            return standardize_sentence_output(subordinate_phrase, lang=self.lang)
        else:
            return None

    def get_valid_marker_indices(self, marker, dep_pattern):
        pos = dep_pattern["POS"]
        if "head" in dep_pattern:
            marker_head = dep_pattern["head"]
        else:
            marker_head = marker

        valid_marker_indices = [i for i in self.indices(marker_head) if self.token(i)["pos"] in pos]
        # if marker=="so":
        #     for i in valid_marker_indices:
        #         print self.find_children(i)
        valid_marker_indices = [i for i in valid_marker_indices if
                                len(self.find_children(i)) == (len(marker.split(" ")) - 1)]
        return valid_marker_indices

    def get_candidate_S2_indices(self, marker, marker_index, dep_pattern, needs_verb=False):
        connection_types = [dep_pattern["S2"]]
        # Look for S2
        return self.find_parents(marker_index, filter_types=connection_types, needs_verb=needs_verb)

    def get_candidate_S1_indices(self, marker, s2_head_index, dep_pattern, needs_verb=False):
        valid_connection_types = [dep_pattern["S1"]]
        return self.find_parents(
            s2_head_index,
            filter_types=valid_connection_types,
            needs_verb=needs_verb
        ) + self.find_children(
            s2_head_index,
            filter_types=valid_connection_types,
            needs_verb=needs_verb
        )

    def add(self, token):
        new_new_tokens = [token]
        index = token["index"]
        for token in self.new_tokens:
            if token["index"] >= index:
                token["index"] = token["index"] + 1
            new_new_tokens.append(token)
        new_new_tokens.sort(key=lambda x: x["index"])
        self.new_tokens = new_new_tokens

    def cut(self, index):
        new_tokens = []
        for token in self.new_tokens:
            new_token = dict(token)
            if token["index"] != index:
                if token["index"] > index:
                    new_token["index"] = new_token["index"] - 1
                new_tokens.append(new_token)
        self.new_tokens = new_tokens

    def move(self, original_index, new_index):
        if (original_index != new_index):
            new_new_tokens = []
            for token in self.new_tokens:
                current_index = token["index"]
                if current_index == original_index:
                    token["index"] = new_index
                elif original_index < new_index:
                    # moving word to later in the sentence
                    # if current_index < original_index:
                    #    #do nothing
                    if original_index < current_index and current_index <= new_index:
                        # so words in between the new and old indices go back one index,
                        # including the word that is currently located at that new index
                        token["index"] = token["index"] - 1
                else:
                    # moving word to earlier in the sentence
                    if new_index <= current_index and current_index < original_index:
                        token["index"] = token["index"] + 1
                new_new_tokens.append(token)
            new_new_tokens.sort(key=(lambda x: x["index"]))
            self.new_tokens = new_new_tokens

def setup_corenlp(lang="en"):
    try:
        test_sentences = {"en": "The quick brown fox jumped over the lazy dog."}
        get_parse(test_sentences[lang], lang=lang)
    except:
        # TODO
        # run the server if we can
        # otherwise ask to install the server and install it if we can
        # raise Exception('corenlp parser needs to be running. see https://github.com/erindb/corenlp-ec2-startup')
        raise Exception(
            "corenlp parser needs to be running for language '{}'. see https://github.com/erindb/corenlp-ec2-startup".format(
                lang))


# def depparse_ssplit(sentence, previous_sentence, marker):
#    sentence = cleanup(sentence)
#    parse = get_parse(sentence)
#    if parse:
#    	# print(json.dumps(parse, indent=4))
#    	sentence = Sentence(parse, sentence)
#    	return(sentence.find_pair(marker, "any", previous_sentence))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    setup_corenlp()