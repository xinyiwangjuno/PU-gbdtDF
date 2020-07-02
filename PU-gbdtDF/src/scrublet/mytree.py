# encoding:gbk
from random import sample


class Tree:
    def __init__(self):
        self.split_feature = None
        self.leftTree = None
        self.rightTree = None
        # 对于real value的条件为<，对于类别值得条件为=
        # 将满足条件的放入左树
        self.real_value_feature = True
        self.conditionValue = None
        self.leafNode = None

    def get_predict_value(self, instance):
        if self.leafNode:  # 到达叶子节点
            return self.leafNode.get_predict_value()
        if not self.split_feature:
            raise ValueError("the tree is null")
        if self.real_value_feature and instance[self.split_feature] < self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        elif not self.real_value_feature and instance[self.split_feature] == self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        return self.rightTree.get_predict_value(instance)

    def describe(self, addtion_info=""):
        if not self.leftTree or not self.rightTree:
            return self.leafNode.describe()
        leftInfo = self.leftTree.describe()
        rightInfo = self.rightTree.describe()
        info = addtion_info+"{split_feature:"+str(self.split_feature)+",split_value:"+str(self.conditionValue)+"[left_tree:"+leftInfo+",right_tree:"+rightInfo+"]}"
        return info


class LeafNode:
    def __init__(self, idset):
        self.idset = idset
        self.predictValue = None

    def describe(self):
        return "{LeafNode:" + str(self.predictValue) + "}"

    def get_idset(self):
        return self.idset

    def get_predict_value(self):
        return self.predictValue

    def update_predict_value(self, targets, step):
        b = sum([targets[x] for x in self.idset])/len(self.idset)
        if b == 0:
            self.predictValue = 0
        else:
            try:
                self.predictValue = step*b
            except ZeroDivisionError:
                print("zero division")
                print("targets are:", [targets[x] for x in self.idset])
                raise
        ## for debug
        # print "targets=",[targets[x] for x in self.idset];
        # print "sum1=",sum1,"sum2=",sum2;
        # print "predict value=",self.predictValue;


def compute_min_loss(values):
    if len(values) < 2:
        return 0
    mean = sum(values) / float(len(values))
    loss = 0.0
    for v in values:
        loss = loss + (mean - v) * (mean - v)
    return loss


## if split_points is larger than 0, we just random choice split_points to evalute minLoss when consider real-value split
def construct_decision_tree(dataset, remainedSet, targets, depth, leafNodes, max_depth, split_points=0):
    # print "start process,depth=",depth;
    if depth < max_depth:
        attributes = dataset.get_attributes()
        loss = -1
        selectedAttribute = None
        conditionValue = None
        selectedLeftIdSet = []
        selectedRightIdSet = []
        for attribute in attributes:
            # print "start process attribute=",attribute;
            is_real_type = dataset.is_real_type_field(attribute)
            attrValues = dataset.get_distinct_valueset(attribute)
            if is_real_type and 0 < split_points < len(
                    attrValues):  ## need subsample split points to speed up
                attrValues = sample(attrValues, split_points)
            for attrValue in attrValues:
                leftIdSet = []
                rightIdSet = []
                for Id in remainedSet:
                    instance = dataset.get_instance(Id)
                    value = instance[attribute]
                    if (is_real_type and value < attrValue) or (
                            not is_real_type and value == attrValue):  ## fall into the left
                        leftIdSet.append(Id)
                    else:
                        rightIdSet.append(Id)
                leftTargets = [targets[id] for id in leftIdSet]
                rightTargets = [targets[id] for id in rightIdSet]
                sumLoss = compute_min_loss(leftTargets) + compute_min_loss(rightTargets)
                if loss < 0 or sumLoss < loss:
                    selectedAttribute = attribute
                    conditionValue = attrValue
                    loss = sumLoss
                    selectedLeftIdSet = leftIdSet
                    selectedRightIdSet = rightIdSet
            # print "for attribute:",attribute," min loss=",loss;
        # print "process over, get split attribute=",selectedAttribute;
        if selectedAttribute is None or loss < 0:
            raise ValueError("cannot determine the split attribute.")
        tree = Tree()
        tree.split_feature = selectedAttribute
        tree.real_value_feature = dataset.is_real_type_field(selectedAttribute)
        tree.conditionValue = conditionValue
        tree.leftTree = construct_decision_tree(dataset, selectedLeftIdSet, targets, depth + 1, leafNodes, max_depth)
        tree.rightTree = construct_decision_tree(dataset, selectedRightIdSet, targets, depth + 1, leafNodes, max_depth)
        # print "build a tree,min loss=",loss,"conditon value=",conditionValue,"attribute=",tree.split_feature;
        return tree
    else:  # is a leaf node
        node = LeafNode(remainedSet)
        step = 0.03
        node.update_predict_value(targets, step)
        leafNodes.append(node)  # add a leaf node
        tree = Tree()
        tree.leafNode = node
        return tree
