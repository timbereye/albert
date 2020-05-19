# coding=utf-8

import fire
import json
import tensorflow as tf

empty_samples = []  # 预测为空的样本
multi_samples = []


def process(test_file, output_file, pred_part1_file=None, pred_part2_file=None, do_part2=False):
    if do_part2:  # 结合part1(processed)和part2的输出，以及预测文件，得到最终提交文件
        with tf.gfile.Open(test_file, 'r') as ft, tf.gfile.Open(pred_part1_file, 'r') as fp1, \
                tf.gfile.Open(pred_part2_file, 'r') as fp2, tf.gfile.Open(output_file, 'w') as fo:
            event_type_dct = {}
            for pred1 in fp1:
                pred1_data = json.loads(pred1)
                key = pred1_data["id"]
                event_type = pred1_data["event_type"]
                event_type_dct[key] = event_type
            pred_part2_data = json.load(fp2)
            for line in ft:
                test_data = json.loads(line)
                key = test_data["id"]
                text = test_data["text"]
                event_list = []
                index = 0
                while True:
                    key_ind = key + "$$" + str(index)
                    if key_ind in event_type_dct:
                        event_type = event_type_dct[key_ind]
                        part2_result = pred_part2_data[key_ind]
                        if part2_result:
                            arguments = []
                            for role, info in part2_result.items():
                                assert len(info) == 1, print(part2_result)
                                argument = info[0]["text"]
                                arguments.append({"role": role, "argument": argument})
                            event_list.append({"event_type": event_type, "arguments": arguments})
                        else:
                            empty_samples.append(line)
                        index += 1
                    else:
                        break
                fo.write(json.dumps({"id": key, "event_list": event_list}, ensure_ascii=False) + "\n")

            print(empty_samples)
            print("empty_samples:", len(empty_samples))
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
                        output_json = {"id": key_new, "text": text, "event_type": event_type, "center_word": {"start": center_word_start,
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
