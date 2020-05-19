#!/bin/bash
data_root="gs://squad_c/albert_et"
pretrained_models="gs://squad_c/pretrained_models/albert_xxlarge_chinese"
albert_config_file_gs=$pretrained_models"/albert_config.json"
output_dir_gs=$data_root"/output_a"
train_file_gs=""
predict_file_gs=""
train_feature_file_gs=$data_root"/features/tf.record.train.1"
predict_feature_file_gs=$data_root"/features/tf.record.test.1"
predict_feature_left_file_gs=$data_root"/features/features.pkl.1"
predict_example_file_gs=$data_root"/features/examples.pkl.1"
init_checkpoint_gs=$pretrained_models"/model.ckpt-best"
vocab_file_gs=$pretrained_models"/vocab_chinese.txt"
tag_info_file_gs=$data_root"/features/tag_info.yaml.1"
export PYTHONPATH="../":$PYTHONPATH
python3 run_extraction.py \
  --albert_config_file=$albert_config_file_gs \
  --output_dir=$output_dir_gs \
  --train_file=$train_file_gs \
  --predict_file=$predict_file_gs \
  --train_feature_file=$train_feature_file_gs \
  --predict_feature_file=$predict_feature_file_gs \
  --predict_feature_left_file=$predict_feature_left_file_gs \
  --predict_example_file=$predict_example_file_gs \
  --init_checkpoint=$init_checkpoint_gs \
  --vocab_file=$vocab_file_gs \
  --do_lower_case \
  --max_seq_length=256 \
  --doc_stride=64 \
  --do_train \
  --do_predict \
  --train_batch_size=40 \
  --predict_batch_size=40 \
  --learning_rate=5e-5 \
  --num_train_epochs=16.0 \
  --warmup_proportion=.1 \
  --save_checkpoints_steps=10000 \
  --n_best_size=20 \
  --max_answer_length=30 \
  --use_tpu \
  --tpu_name="cx" \
  --tag_info_file=$tag_info_file_gs
