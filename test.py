#!/usr/bin/env python
# encoding:utf-8
# -----------------------------------------#
# Filename:     test.py
#
# Description:
# Version:      1.0
# Created:      2020/5/14 16:44
# Author:       chenxiang@myhexin.com
# Company:      www.iwencai.com
#
# -----------------------------------------#

import tensorflow as tf
from tensorflow.contrib import data as contrib_data


def input_fn_builder(input_file, seq_length, is_training,
                     drop_remainder, use_tpu, bsz, is_v2):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }
    # p_mask is not required for SQuAD v1.1
    if is_v2:
        pass
        # name_to_features["p_mask"] = tf.FixedLenFeature([seq_length], tf.int64)

    if is_training:
        # name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        # name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["y_idx"] = tf.FixedLenFeature([seq_length], tf.int64)
        name_to_features["has_answer"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        if use_tpu:
            batch_size = params["batch_size"]
        else:
            batch_size = bsz

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            contrib_data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d