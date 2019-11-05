# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:58
# @File : DataConll2003_Loader.py
# @Last Modify Time : 2018/1/30 15:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :
    FUNCTION :
"""
import os
import re
import random
import torch
from Dataloader.Instance import Instance

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class DataLoaderHelp(object):
    """
    DataLoaderHelp
    """

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _normalize_word(word):
        """
        :param word:
        :return:
        """
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    @staticmethod
    def _sort(insts):
        """
        :param insts:
        :return:
        """
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.words_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=True)
        for key, value in dict:
            sorted_insts.append(insts[key])
        print("Sort Finished.")
        return sorted_insts

    @staticmethod
    def _write_shuffle_inst_to_file(insts, path):
        """
        :return:
        """
        w_path = ".".join([path, shuffle])
        if os.path.exists(w_path):
            os.remove(w_path)
        file = open(w_path, encoding="UTF-8", mode="w")
        for id, inst in enumerate(insts):
            for word, label in zip(inst.words, inst.labels):
                file.write(" ".join([word, label, "\n"]))
            file.write("\n")
        print("write shuffle insts to file {}".format(w_path))


class DataLoader(DataLoaderHelp):
    """
    DataLoader
    """
    def __init__(self, path, shuffle, config):
        """
        :param path: data path list
        :param shuffle:  shuffle bool
        :param config:  config
        """
        #
        print("Loading Data......")
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle
        # char feature
        self.pad_char = [char_pad, char_pad]
        # self.pad_char = []
        self.max_char_len = config.max_char_len

    def dataLoader(self):
        """
        :return:
        """
        path = self.path
        shuffle = self.shuffle
        assert isinstance(path, list), "Path Must Be In List"
        print("Data Path {}".format(path))
        for id_data in range(len(path)):
            print("Loading Data Form {}".format(path[id_data]))
            insts = self._Load_Each_Data(path=path[id_data], shuffle=shuffle)
            random.shuffle(insts)
            self._write_shuffle_inst_to_file(insts, path=path[id_data])
            self.data_list.append(insts)
        # return train/dev/test data
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]

    def _Load_Each_Data(self, path=None, shuffle=False):
        """
        :param path:
        :param shuffle:
        :return:
        """
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        with open(path, encoding="UTF-8") as f:
            inst = Instance()
            for line in f.readlines():
                line = line.strip()
                if line == "" and len(inst.words) != 0:
                    inst.words_size = len(inst.words)
                    insts.append(inst)
                    inst = Instance()
                else:
                    line = line.strip().split(" ") ## line:['EU'. 'S-ORG']
                    word = line[0]
                    char = self._add_char(word)## char是个列表，是每个单词的字母组成
                    word = self._normalize_word(word)##返回的是一个字符串，就是把输入word字符串中的数字转为'0':engl7sh->engl0sh
                    inst.chars.append(char)## inst.chars本身是一个list，把现在这个单词对应的char列表加进去
                    inst.words.append(word)## inst.words本身是一个lsit，把修改之后的word加进去
                    inst.labels.append(line[-1])
                if len(insts) == self.max_count:## 控制读取数据量，一般设置为-1，也就是读取全部数据
                    break
            if len(inst.words) != 0:
                inst.words_size = len(inst.words)
                insts.append(inst)
            # print("\n")
        return insts

    def _add_char(self, word): ## 返回一个列表，列表元素是单词中的字母，超过最大的，取原始单词的前一半和后一半，没超过的我们使用填充
        """
        :param word:
        :return:
        """
        char = []
        # char feature
        for i in range(len(word)):
            char.append(word[i])
        if len(char) > self.max_char_len:
            half = self.max_char_len // 2
            word_half = word[:half] + word[-(self.max_char_len - half):]
            char = word_half
        else:
            for i in range(self.max_char_len - len(char)):
                char.append(char_pad)
        return char


