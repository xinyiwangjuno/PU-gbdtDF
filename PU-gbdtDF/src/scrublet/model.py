#!/usr/bin/env python
from datetime import datetime
from random import sample
from math import exp, log
from mytree import construct_decision_tree
from random import sample
import numpy as np
from sklearn.metrics import roc_auc_score

class Model:
    def __init__(self, max_iter, sample_rate, learn_rate, max_depth, split_points=0):
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.split_points = split_points
        self.trees = dict()

    def train(self, dataset, train_data, stat_file, test_data=None):
        accuracy = 0.0
        f = dict()  ## for train instances
        self.initialize(f, dataset)
        for iter in range(1, self.max_iter + 1):
            subset = train_data
            if 0 < self.sample_rate < 1:
                subset = sample(subset, int(len(subset) * self.sample_rate))
            self.trees[iter] = dict()
            residual = self.compute_residual(dataset, subset, f)
            # print("resiudal of iteration",iter,"###",residual)
            leafNodes = []
            targets = residual
            ## for debug
            # print "targets of iteration:",iter,"and label=",label,"###",targets;
            tree = construct_decision_tree(dataset, subset, targets, 0, leafNodes, self.max_depth,
                                               self.split_points)
            # if label==sample(label_valueset,1)[0]:
            #    print tree.describe("#"*30+"Tree Description"+"#"*30+"\n");
            self.trees[iter] = tree
            self.update_f_value(f, tree, leafNodes, subset, dataset)
            ## for debug
            # print "residual=",residual;
            if test_data is not None:
                auc_score, accuracy, ave_risk = self.test(dataset, test_data)
            train_loss = self.compute_loss(dataset, train_data, f, 0.16)  # lambda can be changed
            test_loss = self.compute_loss(dataset, test_data, f, 0.5)
            stat_file.write(str(iter) + "\t" + str(train_loss) + "\t" + str(accuracy) + "\t" + str(test_loss) + "\n")
            if iter % 1 == 0:
                print("accuracy=%f,average train_loss=%f,average test_loss=%f" % (accuracy, train_loss, test_loss))
                label = "0"
                print("stop iteration:", iter, "time now:", datetime.now())
                print("\n")

    def find_positive_id(self, dataset, subset):
        pos_subset = set()
        for id in subset:
            if int(dataset.get_instance(id)['label']) == 1:
                pos_subset.add(id)
        return set(pos_subset)

    def compute_instance_f_value(self, instance):
        f_value = 0.0
        for iter in self.trees:
            tree = self.trees[iter]
            f_value += self.learn_rate * tree.get_predict_value(instance)
        return f_value

    def compute_loss(self, dataset, subset, f, mylambda):
        sum1 = sum(f[id] for id in subset)
        # print(sum1)
        mean1 = log(sum1 / len(subset))
        subsetp = self.find_positive_id(dataset, subset)
        sum2 = sum(log(f[id]) for id in subset)
        mean2 = sum2 / len(subsetp)
        loss = mean1 - (1 + mylambda) * mean2
        return loss

    def initialize(self, f, dataset):
        for id in dataset.get_instances_idset():
            f[id] = 0.95

    def update_f_value(self, f, tree, leafNodes, subset, dataset):
        data_idset = set(dataset.get_instances_idset())
        subset = set(subset)
        for node in leafNodes:
            for id in node.get_idset():
                f[id] += self.learn_rate * node.get_predict_value()
        ## for id not in subset, we have to predict by retrive the tree
        for id in data_idset - subset:
            f[id] += self.learn_rate * tree.get_predict_value(dataset.get_instance(id))

    def compute_residual(self, dataset, subset, f):
        residual = {}
        sum1 = 0.0
        for id in subset:
            sum1 = sum1 + f[id]
        # print(sum1)
        subsetp = self.find_positive_id(dataset, subset)
        nP = len(subsetp)
        for id in subset:
            if id in subsetp:
                residual[id] = 1 / (sum1) - 1 / (nP * (f[id]+0.0001))
            else:
                residual[id] = 1 / (sum1)
        return residual

    def predict_label(self, instance):
        f_value = self.compute_instance_f_value(instance)
        # print(f_value)
        probs = dict()
        predict_label = None
        #f_normalize = exp(f_value)/(exp(f_value)+exp(1-f_value))
        p_neg = 1/(1+np.exp(-2*f_value))
        p_pos = 1-p_neg
        if p_pos >= p_neg:
            predict_label = 1
        else:
            predict_label = -1
        probs['1'] = p_pos
        probs['-1'] = p_neg
        return predict_label, probs

    def test(self, dataset, test_data):
        right_predition = 0
        label_valueset = dataset.get_label_valueset()
        # print(label_valueset)
        risk = 0.0
        predict_label = []
        for id in test_data:
            instance = dataset.get_instance(id)
            predict_label_test, probs = self.predict_label(instance)
            predict_label.append(predict_label_test)
            # print(probs)
            single_risk = 0.0
            for label in probs:
                if int(float(label)) == int(float(instance["label"])):
                    single_risk = single_risk + (1.0 - probs[label])
                else:
                    single_risk = single_risk + probs[label]
            # print probs,"instance label=",instance["label"],"##single_risk=",single_risk/len(probs)
            risk = risk + single_risk / len(probs)
            if int(float(instance["label"])) == int(float(predict_label_test)):
                right_predition = right_predition + 1
        print("test data size=%d,test accuracy=%f"%(len(test_data), float(right_predition)/len(test_data)))
        # print(len(predict_label))
        return predict_label, float(right_predition) / len(test_data), risk / len(test_data)


