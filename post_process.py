# coding=utf-8

import fire
import json
import tensorflow as tf

empty_samples = []  # 预测为空的样本
multi_samples = []


def process(test_file, output_file, pred_part1_file=None, pred_part2_file=None, do_part2=False):
    if do_part2:  # 结合part1(processed)和part2的输出，以及预测文件，得到最终提交文件
        pass
    else:  # 结合part1的输出和预测文件,得到处理后的part1结果文件
        with tf.gfile.Open(test_file, 'r') as ft, tf.gfile.Open(pred_part1_file, 'r') as fp1, \
                tf.gfile.Open(output_file, 'w') as fo:
            pred_part1_data = json.load(fp1)
            for line in ft:
                test_data = json.loads(line)
                key = test_data["id"]
                text = test_data["text"]
                result = pred_part1_data[key]
                if result:
                    index = 0
                    for event_type, center_words in result.items():
                        assert len(center_words) == 1
                        center_word = center_words[0]
                        center_word_text = center_word["text"]
                        center_word_start = center_word["start"]
                        center_word_end = center_word["end"]
                        key_new = str(key) + "$$" + str(index)
                        output_json = {"key": key_new, "text": text, "center_word": {"start": center_word_start,
                                                                                     "end": center_word_end,
                                                                                     "text": center_word_text}}
                        fo.write(json.dumps(output_json, ensure_ascii=False) + "\n")
                        index += 1
                else:
                    empty_samples.append(test_data)

            print(empty_samples)
            print("empty_samples:", len(empty_samples))


if __name__ == '__main__':
    fire.Fire(process)
