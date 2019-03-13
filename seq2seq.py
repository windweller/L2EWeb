"""
If the goal is to integrate this code into the system,
using a Model might be a better idea

To make this code compatible with latest OpenNMT version
Simply import missing functions from data.py
"""

import uuid
import os
import argparse

from os.path import join as pjoin
import numpy as np
import torch
from onmt.model_builder import build_base_model
from onmt import inputters

import onmt
import onmt.translate
from onmt.translate import Translator
import onmt.opts as opts

# from data import build_dataset, make_features

def _run_encoder(model, batch, data_type='text'):
    src = inputters.make_features(batch, 'src', data_type)
    src_lengths = None
    if data_type == 'text':
        _, src_lengths = batch.src
    elif data_type == 'audio':
        src_lengths = batch.src_lengths
    enc_states, memory_bank, src_lengths = model.encoder(
        src, src_lengths)
    if src_lengths is None:
        assert not isinstance(memory_bank, tuple), \
            'Ensemble decoding only supported for text data'
        src_lengths = torch.Tensor(batch.batch_size) \
            .type_as(memory_bank) \
            .long() \
            .fill_(memory_bank.size(0))

    # enc_states: (hidden_state, cell_state)
    # hidden_state: (layer_size, batch, dim)

    # * enc_states: final state of the encoder
    # * memory_bank `[src_len x batch_size x model_dim]`
    return src, enc_states, memory_bank, src_lengths


class PhenomenonEncoder(object):
    def __init__(self, model_file, temp_dir, logger, silent=False):
        self.model_file = model_file
        self.temp_dir = temp_dir

        if not silent:
            logger.info("Start loading the model")

        # checkpoint = torch.load(self.model_file,
        #                         map_location=lambda storage, location: 'cpu')
        checkpoint = torch.load(self.model_file, map_location='cpu')

        # fields = inputters.load_fields_from_vocab(checkpoint['vocab'])
        fields = inputters.load_fields_from_vocab(checkpoint['vocab'])
        model_opt = checkpoint['opt']

        # OpenNMT changed their configuration...
        model_opt.enc_rnn_size = model_opt.rnn_size
        model_opt.dec_rnn_size = model_opt.rnn_size
        # model_opt.max_relative_positions = 0  # default is 0
        # model_opt.model_dtype = 'fp32'  # not half model

        model = build_base_model(model_opt, fields, gpu=False, checkpoint=checkpoint)
        model.eval()
        model.generator.eval()

        self.model = model
        self.fields = fields
        self.model_opt = model_opt

        if not os.path.exists(pjoin(self.temp_dir, "l2e")):
            os.makedirs(pjoin(self.temp_dir, "l2e"))

        if not silent:
            logger.info("Model built")

    def generate_vectors(self, list_of_sentences, batch_size=1, cuda=False):
        """
        list_of_sentences: [str]
        batch_size: int
        :return [np.array] numpy vectors of sentences in the same order
        """
        unique_filename = str(uuid.uuid4())

        # delete repeating tmp files
        tmp_files = os.listdir(pjoin(self.temp_dir, "l2e"))

        if len(tmp_files) > 10:
            for f_n in tmp_files:
                os.remove(pjoin(self.temp_dir, "l2e", f_n))

        with open(pjoin(self.temp_dir, "l2e", '{}.txt'.format(unique_filename)), 'w') as f:
            for s in list_of_sentences:
                f.write(s.strip() + '\n')

        data = inputters.build_dataset(self.fields,
                                       src_path=pjoin(self.temp_dir, "l2e", '{}.txt'.format(unique_filename)),
                                       data_type='text',
                                       use_filter_pred=False)  # src_seq_length=50, dynamic_dict=False)

        if cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        all_vecs = []

        for batch in data_iter:
            # translation model just translates, here we generate vectors instead
            src, enc_states, memory_bank, src_lengths = _run_encoder(self.model, batch, 'text')
            # enc_states[0]: (layer_size, batch_size, hid)
            reshaped_hid_states = enc_states[0].reshape(batch_size, self.model_opt.enc_layers,
                                                        self.model_opt.enc_rnn_size)
            # reshaped_hid_states: (batch_size, layer_size, hid)
            # we only append the 2nd layer
            all_vecs.append(reshaped_hid_states[:, -1, :].data.cpu().numpy())

        all_vecs = np.vstack(all_vecs)

        return all_vecs

def build_translator(model, fields, model_opt, beam_size=1, n_best=1):
    # model, fields, model_opt are all loaded from start of script
    model_opt.beta = -0.0
    model_opt.coverage_penalty = "none"
    model_opt.length_penalty = "none"

    scorer = onmt.translate.GNMTGlobalScorer(alpha=0.0, beta=-0.0, cov_penalty="none", length_penalty="none")

    dummy_parser = argparse.ArgumentParser(description='translate.py')
    opts.translate_opts(dummy_parser)
    dummy_translate_opt = dummy_parser.parse_known_args("-model dummy -src dummy".split())[0]
    dummy_translate_opt.beam_size = beam_size
    dummy_translate_opt.cuda= False

    translator = Translator(model, fields, beam_size, n_best,
                            global_scorer=scorer, out_file=None,
                            report_score=False, gpu=False, replace_unk=True)
    # translator.beam_size = beam_size
    # translator.n_best = n_best

    return translator


class L2EDecoder(object):
    def __init__(self, phenom_encoder: PhenomenonEncoder):
        """
        Twitter Decoder requries the pass in of a Twitter Encoder.
        :param phenom_encoder:
        """
        self.translator = build_translator(phenom_encoder.model,
                                           phenom_encoder.fields, phenom_encoder.model_opt, beam_size=5,
                                           n_best=1)
        self.translator.cuda = False  # there's a bug here, we need to manually set this
        self.fields = phenom_encoder.fields
        self.temp_dir = phenom_encoder.temp_dir

    def decode_sentences(self, sents, cuda=False):
        """
        Takes in a list of sentences and returns a list of sentences
        decode_sentences(['this is fun !', "this is not fun"])
        [('this is fun !', 'I 'm not a this .', -12.412576675415039),
        ('this is not fun', 'I 'm not sure .', -10.160457611083984)]
        :param sents: [str]
        :return: [(src, tgt, log-likelihood-score)]
        """
        unique_filename = str(uuid.uuid4())

        # delete repeating tmp files
        tmp_files = os.listdir(pjoin(self.temp_dir, "l2e"))

        if len(tmp_files) > 10:
            for f_n in tmp_files:
                os.remove(pjoin(self.temp_dir, "l2e", f_n))

        with open(pjoin(self.temp_dir, "l2e", '{}.txt'.format(unique_filename)), 'w') as f:
            for s in sents:
                f.write(s.strip() + '\n')

        data = inputters.build_dataset(self.fields,
                                       src_path=pjoin(self.temp_dir, "l2e", '{}.txt'.format(unique_filename)),
                                       data_type='text',
                                       use_filter_pred=False,
                                       dynamic_dict=False)  # src_seq_length=50, dynamic_dict=False)

        if cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=1, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.fields, n_best=1, replace_unk=True, has_tgt=False)

        # this is not really beam-search...
        decoded_sents = []  # (src, tgt, score)

        # we don't keep statistics / scores or anything

        for batch in data_iter:
            batch_data = self.translator.translate_batch(batch, data, fast=False)
            translations = builder.from_batch(batch_data)

            # going through each sentence in a batch
            for trans in translations:
                n_best_preds = [" ".join(pred) for pred in trans.pred_sents[:self.translator.n_best]]
                for i in range(len(n_best_preds)):
                    decoded_sents.append((' '.join(trans.src_raw), n_best_preds[i], trans.pred_scores[i].item()))

        return decoded_sents
