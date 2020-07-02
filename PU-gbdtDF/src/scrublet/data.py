#!/usr/bin/env python
from random import sample
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pandas as pd


class DataSet:

    def __init__(self, counts_matrix):  ## just for csv data format
        self.df = pd.DataFrame(counts_matrix)
        self.instances = self.df.to_dict('index')
        self.field_names = self.df.columns

    def describe(self):
        info = self.df.describe()
        print(info)

    def get_instances_idset(self):
        return set(self.df.index)

    def is_real_type_field(self, name):
        if name not in self.field_names:
            raise ValueError(" field name not in the dictionary of dataset")
        # return (name in self.distinct_valueset);
        return is_numeric_dtype(self.df[name])

    def get_label_size(self, name="label"):
        if name not in self.field_names:
            raise ValueError(" there is no class label field!")
        # 因为训练样本的label列的值可能不仅仅是字符类型，也可能是数字类型
        # 如果是数字类型则field_type[name]为空
        return self.df[name].count()

    def get_label_valueset(self, name="label"):
        """返回具体分离label"""
        if name not in self.field_names:
            raise ValueError(" there is no class label field!")
        return set(self.df[name])

    def size(self):
        """返回样本个数"""
        return self.df.shape[0]

    def get_instance(self, Id):
        """根据ID获取样本"""
        if Id not in self.instances:
            raise ValueError("Id not in the instances dict of dataset")
        return self.instances[Id]

    def get_attributes(self):
        """返回所有features的名称"""
        field_names = [x for x in self.field_names if x != "label"]
        return tuple(field_names)

    def get_distinct_valueset(self, name):
        if name not in self.field_names:
            raise ValueError("the field name not in the dataset field dictionary")
        if self.is_real_type_field(name):
            return set(self.df[name])
        else:
            return set(self.df[name])

    def datasetP_id(self, p2u_pro, train_rate):
        train_size = int(self.size() * train_rate)
        P_size = round(train_size * p2u_pro / (p2u_pro + 1))
        train_data_id = sample(self.get_instances_idset(), train_size)
        test_data_id = set(self.get_instances_idset()) - set(train_data_id)
        datasetp_id = set()
        for k in train_data_id:
            if int(self.get_instance(k)['label']) == 1:
                datasetp_id.add(k)
        p_cnt = len(datasetp_id)
        if p_cnt >= P_size:
            datasetp_id = (list(datasetp_id))[:P_size]
        else:
            for k in test_data_id:
                if int(self.get_instance(k)['label']) == 1:
                    datasetp_id.add(k)
                    p_cnt += 1
                if p_cnt >= P_size:
                    break
        return set(datasetp_id)

    def train_data_id(self, p2u_pro, train_rate):
        train_size = int(self.size() * train_rate)
        P_size = round(train_size * p2u_pro / (p2u_pro + 1))
        train_data_id = set(self.datasetP_id(p2u_pro, train_rate)) | set(sample(
            set(self.get_instances_idset()) - set(self.datasetP_id(p2u_pro, train_rate)),
            train_size - P_size))
        return set(train_data_id)

    def test_data_id(self, p2u_pro, train_rate):
        test_data_id = set(self.get_instances_idset()) - set(self.train_data_id(p2u_pro, train_rate))
        return set(test_data_id)


if __name__ == "__main__":
    from sys import argv

    data = DataSet(argv[1])
    print("instances size=", len(data.instances))
    print(data.instances[1])
