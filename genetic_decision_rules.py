from random import randint, uniform, choice
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score
'''arff: https://pythonhosted.org/liac-arff/'''
import arff
import numpy as np
from math import sqrt
import time

operator2function = {
    "=": lambda x, y : x == y,
    "!=": lambda x, y : x != y,
    "<": lambda x, y : x < y,
    ">": lambda x, y : x > y,
    "<=": lambda x, y : x <= y,
    ">=": lambda x, y : x >= y
}

accuracy_functions = [
    lambda ind, X, y : balanced_accuracy_score(y, [ind.apply(row) for row in X]),
    lambda ind, X, y : f1_score(y, [ind.apply(row) for row in X], average="macro"),
    lambda ind, X, y : accuracy(ind, X, y)
]

interpretability_functions = [
    lambda ind, X, y : interpretability(ind) * overlap_score(ind, X),
    lambda ind, X, y : interpretability(ind)
]

boolean_space = [True, False]

'''
a simple clause pertaining one operand and a value
operand: the name of the operand (used when printing the clause)
operator: the operator (=, !=, <, >, <=, >=)
value: the second operand which is a constant value
'''
class Clause:
    def __init__(self, operand:str, operator, value) -> None:
        self.operand:str = operand
        self.operator = operator
        self.value = value
    
    def __str__(self) -> str:
        operator = self.operator
        if operator == "=":
            operator = "is"
        elif operator == "!=":
            operator = "is not"
        return f"{self.operand} {operator} {self.value}"
    
    def apply(self, value):
        try:
            # if argument can be interpreted as a number, convert to float
            value = float(value)
        except ValueError:
            pass
        return operator2function[self.operator](value, self.value)

    def __eq__(self, __value: object) -> bool:
        return (
            type(__value) is Clause and
            self.operand == __value.operand and
            self.operator == __value.operator and
            self.value == __value.value
        )

'''
a conjunction of simple clauses
'''
class Conjunction:
    def __init__(self, *clauses:Clause) -> None:
        self.clauses:tuple[Clause] = clauses

    def __str__(self) -> str:
        return " AND ".join(str(clause) for clause in self.clauses)
    
    def apply(self, values):
        for clause in self.clauses:
            if not clause.apply(values[clause.operand]):
                return False
        return True

    def __iter__(self):
        yield from self.clauses
    
    def __eq__(self, __value: object) -> bool:
        if type(__value) is not Conjunction or len(self.clauses) != len(__value.clauses):
            return False
        for x, y in zip(self.clauses, __value.clauses):
            if x != y:
                return False
        return True
    
    def __len__(self):
        return len(self.clauses)
        

'''a decision rule'''
class Rule:
    def __init__(self, output, *clauses:Clause) -> None:
        self.condition = Conjunction(*clauses)
        self.output = output

    def __str__(self) -> str:
        return f"IF {self.condition} THEN {self.output}"

    def apply(self, values):
        return self.output if self.condition.apply(values) else None

    def __eq__(self, __value: object) -> bool:
        return (
            type(__value) is Rule and
            self.condition == __value.condition and
            self.output == __value.output
        )
    
    '''
    returns a mapping of the operands and operators to the list of constants they occur with in the rule
    '''
    def summary(self) -> dict[str, dict[str, list]]:
        output = dict()
        for clause in self.condition:
            if clause.operand not in output:
                output[clause.operand] = dict()
            sub = output[clause.operand]
            if clause.operator not in sub:
                sub[clause.operator] = []
            sub[clause.operator].append(clause.value)
        return output
    
    def __len__(self):
        return len(self.condition)

'''a decision rule list'''
class RuleList:
    def __init__(self, default, *rules) -> None:
        self.default = default
        self.rules = rules
        self.attribute_names = None
    
    def __str__(self) -> str:
        return "\nELSE ".join(str(rule) for rule in self.rules) + f"\nELSE {self.default}"

    def apply(self, values):
        if type(values) is not dict:
            values = dict(zip(self.attribute_names, values))
        for rule in self.rules:
            output = rule.apply(values)
            if output is not None:
                return output
        return self.default

    def __iter__(self):
        yield from self.rules
    
    def __eq__(self, __value: object) -> bool:
        if type(__value) is not RuleList or self.default != __value.default or len(self.rules) != len(__value.rules):
            return False
        for x, y in zip(self.rules, __value.rules):
            if x != y:
                return False
        return True
    
    def __len__(self):
        return len(self.rules)

class EncodeError(Exception):
    def __init__(self, msg:str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg

'''
facilitates the encoding of rule lists to chromosomes
a space is a list of possible values or a tuple of numerical bounds
output_space: the space to which the output belongs
spaces: the spaces to which the inputs belong
'''
class Encoding:
    def __init__(self, output_space, **spaces) -> None:
        self.output_space = output_space
        self._space_dict = spaces
        self.spaces = tuple(spaces.items())
        self._space_dict["output"] = output_space
        for space in self.spaces:
            if type(space[1]) != list and (type(space[1]) != tuple or len(space[1]) != 2):
                raise ValueError("a space should be a list of values or a binary tuple of numerical boundaries")
        # maps position in chromosome rule to the space of the value at that position
        self._field2space = []
        for _, space in self.spaces:
            if type(space) is tuple:
                for _ in range(2):
                    self._field2space.append(space)
            else:
                self._field2space.append(space)
                for _ in range(len(space)):
                    self._field2space.append(boolean_space)
        self._field2space.append(self.output_space)
        self._encoded_rule_size = len(self._field2space)
        self._gene_prob = 1 / sum(len(space[1]) for space in spaces)
        self.attribute_names = None
    
    def __getitem__(self, key:str):
        return self._space_dict[key]
    
    def __str__(self):
        return str(self._space_dict)
    
    def get_space(self, chrom:list, index:int):
        if index == len(chrom) - 1:
            return self.output_space
        return self._field2space[index % self._encoded_rule_size]
    
    def encoded_size(self, space):
        if type(space) is tuple:
            return 2
        if type(space) is list:
            return len(space) + 1
    
    def _encode_rule(self, rule:Rule, output:list):
        summary = rule.summary()
        for attribute, space in self.spaces:
            if attribute in summary:
                if type(space) is tuple:
                    # numeric attribute
                    bounds = [None, None]
                    # assume few clauses
                    for clause in rule.condition:
                        if clause.operand == attribute:
                            if clause.operator[0] == ">":
                                bounds[0] = clause.value
                            elif clause.operator[0] == "<":
                                bounds[1] = clause.value
                            elif clause.operator == "=":
                                bounds[0] = clause.value
                                bounds[1] = clause.value
                    output.append(bounds[0])
                    output.append(bounds[1])
                else:
                    # categorical attribute
                    ops = summary[attribute]
                    if "=" in ops:
                        if len(ops["="]) > 1:
                            raise EncodeError("more than one value assigned to a categorical attribute")
                        output.append(ops["="][0])
                    else:
                        output.append(None)
                    if "!=" in ops:
                        for value in space:
                            if value in ops["!="]:
                                output.append(True)
                            else:
                                output.append(False)
                    else:
                        for _ in range(len(space)):
                            output.append(False)
            else:
                # attribute not used in clause
                for _ in range(self.encoded_size(space)):
                    output.append(None)
        output.append(rule.output)
        
    def encode(self, rules:RuleList) -> list:
        output = []
        for rule in rules:
            self._encode_rule(rule, output)
        output.append(rules.default)
        return output

    def decode(self, chrom:list) -> RuleList:
        try:
            chrom_len = len(chrom)
        except TypeError:
            chrom = tuple(chrom)
            chrom_len = len(chrom)
        rule_count = chrom_len // self._encoded_rule_size
        rules = []
        # pointer to the current position in the chromosome
        ptr = 0
        for _ in range(rule_count):
            clauses = []
            for attribute, space in self.spaces:
                if type(space) is tuple:
                    # expect a lower bound followed by an upper bound
                    # if the lower bound is greater than the upper bound, both bounds are latent
                    if chrom[ptr] is None or chrom[ptr+1] is None or chrom[ptr] <= chrom[ptr+1]:
                        if chrom[ptr] == chrom[ptr+1] and chrom[ptr] is not None:
                            clauses.append(Clause(attribute, "=", chrom[ptr]))
                            ptr += 1
                        else:
                            if chrom[ptr] is not None:
                                clauses.append(Clause(attribute, ">=", chrom[ptr]))
                            ptr += 1
                            if chrom[ptr] is not None:
                                clauses.append(Clause(attribute, "<=", chrom[ptr]))
                    else:
                        ptr += 1
                else:
                    # expect an included value followed by a boolean for every possible excluded value
                    # if there is an included value, the excluded values are latent
                    base_ptr = ptr + 1
                    if chrom[ptr] is None:
                        # excluded values
                        for _ in range(len(space)):
                            ptr += 1
                            if chrom[ptr]:
                                clauses.append(Clause(attribute, "!=", space[ptr - base_ptr]))
                    else:
                        # included value
                        clauses.append(Clause(attribute, "=", chrom[ptr]))
                        # excluded values are latent
                        for _ in range(len(space)):
                            ptr += 1
                # next attribute
                ptr += 1
            if len(clauses) > 0:
                rules.append(Rule(chrom[ptr], *clauses))
            # next rule
            ptr += 1
        output = RuleList(chrom[ptr], *rules)
        output.attribute_names = self.attribute_names
        return output

    '''
    increase the length of a chromosome without altering it's meaning
    '''
    def _lengthen(self, chrom:list, num_extra_rules:int):
        output = chrom[:-1]
        for _ in range(num_extra_rules):
            for _ in range(self._encoded_rule_size - 1):
                output.append(None)
            # the output of a rule may never be None so we assign a latent value in stead
            output.append(self._random_value(self.output_space))
        output.append(chrom[-1])
        return output
    
    '''
    exchange values between 2 chromosomes
    the original chromosomes remain unchanged, the new chromosomes are returned
    '''
    def crossover(self, chrom1:list, chrom2:list) -> tuple[list]:
        # chromosomes must be of equal length
        if len(chrom1) < len(chrom2):
            chrom1 = self._lengthen(chrom1, (len(chrom2) - len(chrom1)) // self._encoded_rule_size)
        elif len(chrom2) < len(chrom1):
            chrom2 = self._lengthen(chrom2, (len(chrom1) - len(chrom2)) // self._encoded_rule_size)
        length = len(chrom1)
        # minimal size of the exchanged part of the chromosomes
        min_cross = length // 10
        start = randint(0, length - min_cross)
        end = randint(start + min_cross, length)
        offspr1 = chrom1[:start] + chrom2[start:end] + chrom1[end:]
        offspr2 = chrom2[:start] + chrom1[start:end] + chrom2[end:]
        return (offspr1, offspr2)
    
    def _random_value(self, space):
        if type(space) is tuple:
            return uniform(space[0], space[1])
        else:
            return choice(space)
    
    def _random_gene(self, space):
        if type(space) is tuple:
            x = None
            y = None
            if uniform(0, 1) < self._gene_prob:
                x = uniform(space[0], space[1])
            if uniform(0, 1) < self._gene_prob:
                if x is None:
                    y = uniform(space[0], space[1])
                else:
                    y = uniform(x, space[1])
            yield x
            yield y
        else:
            if uniform(0, 1) < self._gene_prob:
                yield choice(space)
            else:
                yield None
            for _ in range(len(space)):
                if uniform(0, 1) < self._gene_prob:
                    yield choice(boolean_space)
                else:
                    yield None

    def random_chromosome(self, rule_count:int):
        output = []
        for _ in range(rule_count):
            for _, space in self.spaces:
                for e in self._random_gene(space):
                    output.append(e)
            output.append(self._random_value(self.output_space))
        output.append(self._random_value(self.output_space))
        return output
    
    '''
    make some random changes in a chromosome
    '''
    def mutate(self, chrom:list, mutation_prob=0.01):
        for i in range(len(chrom) - 1):
            val = uniform(0, 1)
            if val < mutation_prob / 2:
                # substitute value
                space = self.get_space(chrom, i)
                chrom[i] = self._random_value(space)
            elif i % self._encoded_rule_size < self._encoded_rule_size - 1 and val < mutation_prob:
                # delete value
                chrom[i] = None
        val = uniform(0, 1)
        if val < mutation_prob:
            chrom[-1] = self._random_value(self.output_space)

class Pareto:
    def __init__(self, *values) -> None:
        if len(values) == 1:
            self._values = values[0]
        else:
            self._values = values
    
    def __str__(self):
        return str(self._values)
    
    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return self._values.__hash__()

    def __len__(self):
        return len(self._values)
    
    def __getitem__(self, i):
        return self._values[i]

    def __add__(self, other):
        if len(self) != len(other):
            raise ValueError("vectors should have same size")
        return Pareto(x + y for x, y in zip(self, other))
    
    def __iadd__(self, other):
        if len(self) != len(other):
            raise ValueError("vectors should have same size")
        for i in range(len(self)):
            self._values[i] += other._values[i]
        return self
    
    def __sub__(self, other):
        if len(self) != len(other):
            raise ValueError("vectors should have same size")
        return Pareto(x - y for x, y in zip(self, other))
    
    def __isub__(self, other):
        if len(self) != len(other):
            raise ValueError("vectors should have same size")
        for i in range(len(self)):
            self._values[i] -= other._values[i]
        return self

    def __mul__(self, other):
        return Pareto(x * other for x in self)
    
    def __imul__(self, other):
        for i in range(len(self)):
            self._values[i] *= other
        return self
    
    def __truediv__(self, other):
        return Pareto(x / other for x in self)
    
    def __itruediv__(self, other):
        for i in range(len(self)):
            self._values[i] /= other
        return self

    def __iter__(self):
        return self._values.__iter__()
    
    '''
    pareto dominance:
    a vector x dominates a vector y if there is no i such that x[i] < y[i]
    and there is at least one i such that x[i] > y[i]
    '''
    def __lt__(self, other):
        if len(self) != len(other):
            raise ValueError("vectors must be of same length")
        i = 0
        less = False
        while i < len(other) and self[i] <= other[i]:
            if self[i] < other[i]:
                less = True
            i += 1
        return i == len(self) and less
    
    def __gt__(self, other):
        return other < self
    
    def __eq__(self, other):
        return self._values == other._values
    
    def __neq__(self, other):
        return self._values != other._values
    
    def __le__(self, other):
        return self == other or self < other
    
    def __ge__(self, other):
        return self == other or self > other
    
    def norm(self):
        return sqrt(sum(x**2 for x in self._values))

class ParetoSet:
    def __init__(self):
        self._content = dict()
    
    def __str__(self):
        return str(self._content.items())
    
    def __iter__(self):
        return self._content.items().__iter__()
    
    def __len__(self):
        return len(self._content)
    
    def add(self, item:tuple) -> bool:
        dominated = False
        marked = set()
        # pareto dominance is transitive
        for e in self:
            if item[0] <= e[0]:
                # if dominated or equal, do not add to pareto set
                dominated = True
                break
            elif item[0] > e[0]:
                # when added to pareto set, remove dominated items
                marked.add(e[0])
        if not dominated:
            for key in marked:
                self._content.pop(key)
            self._content[item[0]] = item[1]
        return not dominated

    def clear(self):
        self._content.clear()

def remove_duplicate_pareto(pop:list[tuple]):
    marked = set()
    i = len(pop) - 1
    while i >= 0:
        if pop[i][0] in marked:
            pop.pop(i)
        else:
            marked.add(pop[i][0])
        i -= 1

'''
1. count the amount of times individuals in the population are dominated
2. for each individual find all individuals that are dominated by it
'''
def dominance_count(pop:list[tuple]) -> tuple[list[int], list[set[int]]]:
    count = [0 for _ in range(len(pop))]
    dom = [set() for _ in range(len(pop))]
    for i in range(len(pop)):
        for j in range(len(pop)):
            if pop[i][0] < pop[j][0]:
                count[i] += 1
                dom[j].add(i)
    return (count, dom)

'''
sort the population based on pareto ranking
'''
def pareto_rank_sort(pop:list[tuple]) -> list[list[tuple]]:
    output = []
    remove_duplicate_pareto(pop)
    count, dom = dominance_count(pop)
    delta = [0 for _ in range(len(pop))]
    done = 0
    while done < len(pop):
        sub = []
        for i in range(len(pop)):
            if count[i] == 0:
                sub.append(pop[i])
                done += 1
                count[i] = -1
                for j in dom[i]:
                    delta[j] += 1
        for i in range(len(pop)):
            count[i] -= delta[i]
            delta[i] = 0
        output.append(sub)
    return output

"""
sort a pareto set based on the crowding distance
"""
def crowding_distance_sort(pareto_set, normalization=None):
    l = len(pareto_set)
    vec_ind_dist = [[vec, ind, 0] for vec, ind in pareto_set]
    dim = len(vec_ind_dist[0][0])
    if normalization is None:
        normalization = [1 for _ in range(dim)]
    for i in range(dim):
        vec_ind_dist.sort(key=lambda x : x[0][i])
        vec_ind_dist[0][2] = np.inf
        vec_ind_dist[l-1][2] = np.inf
        for j in range(1, l-1):
            vec_ind_dist[j][2] += abs(vec_ind_dist[j+1][0][i] - vec_ind_dist[j-1][0][i]) / normalization[i]
    vec_ind_dist.sort(key=lambda x : x[2], reverse=True)
    return [(vec, ind) for vec, ind, _ in vec_ind_dist]

'''
compute the average of pareto sets
'''
def pareto_average(pareto_sets:list[list[Pareto]]):
    output = []
    n = max(len(par) for par in pareto_sets)
    dim = len(pareto_sets[0][0])
    for i in range(n):
        par_mean = Pareto([0 for _ in range(dim)])
        m = 0
        for par in pareto_sets:
            if i < len(par):
                par_mean += par[i]
                m += 1
        par_mean /= m
        output.append(par_mean)
    return output


'''
arff_data: dictionary returned by arff.load (https://pythonhosted.org/liac-arff/)
output_name: name of the output attribute
returns the data as a numpy array along with the corresponding feature spaces and encoding scheme
note:   data with missing values are removed
note:   attributes of type STRING are ignored, all possible values must be provided in a list,
        attributes that can hold any string are unusable
'''
def process_arff(arff_data:dict, output_name:str):
    attributes = [(name.replace("_", " "), space) for name, space in arff_data["attributes"]]
    data = arff_data["data"]
    real_attributes = []
    for index, attribute_space in enumerate(attributes):
        space = attribute_space[1]
        # all numerical attributes are treated as real
        if space == "REAL" or space == "INTEGER":
            real_attributes.append(index)
    spaces = dict()
    marked = []
    # construct spaces for real attributes
    # the bounds are the smallest and largest values found in the data
    for i, row in enumerate(data):
        for index in real_attributes:
            attribute = attributes[index][0]
            value = row[index]
            if value is None or np.isnan(value):
                # rows with missing values are marked for deletion
                if len(marked) == 0 or marked[-1] != i:
                    marked.append(i)
            else:
                if attribute in spaces:
                    space = spaces[attribute]
                    if value < space[0]:
                        space[0] = value
                    elif value > space[1]:
                        space[1] = value
                else:
                    # needs to be a tuple, but tuples are immutable
                    spaces[attribute] = [value, value]
    for i in marked[::-1]:
        data.pop(i)
    marked.clear()
    # remove the remaining missing values
    for i, row in enumerate(data):
        for value in row:
            if value is None and (len(marked) == 0 or marked[-1] != i):
                # rows with missing values are marked for deletion
                marked.append(i)
    for i in marked[::-1]:
        data.pop(i)
    # convert to tuples once the bounds are known
    for attribute in spaces.keys():
        spaces[attribute] = tuple(spaces[attribute])
    # add categorical attributes
    for attribute, space in attributes:
        if type(space) is list:
            spaces[attribute] = space
    # the output space is seperate
    output_space = spaces[output_name]
    del spaces[output_name]
    return (
        np.array(data),
        attributes,
        Encoding(output_space, **spaces)
    )

'''
adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#tree-structure
'''
def decision_tree_path_iter(clf:DecisionTreeClassifier, node:int=0, path:list=[]):
    if clf.tree_.children_left[node] == clf.tree_.children_right[node]:
        # leaf node
        # append class label
        path.append(clf.classes_[np.argmax(clf.tree_.value[node])])
        yield tuple(path)
        # return to parent
        path.pop()
    else:
        # append (direction, feature, split value) 
        path.append(("left", clf.tree_.feature[node], clf.tree_.threshold[node]))
        # not leaf node -> 2 children
        yield from decision_tree_path_iter(clf, node=clf.tree_.children_left[node], path=path)
        # flip direction
        path[-1] = ("right", path[-1][1], path[-1][2])
        yield from decision_tree_path_iter(clf, node=clf.tree_.children_right[node], path=path)
        # return to parent
        path.pop()

def path2rule(path, feature_names, output_classes, index2class=True) -> Rule:
    clauses = []
    for dir, feature_id, split in path[:-1]:
        try:
            feature, value = feature_names[feature_id].split("_")
            operator = "!=" if dir == "left" else "="
        except ValueError:
            feature = feature_names[feature_id]
            operator = "<=" if dir == "left" else ">"
            value = split
        clauses.append(Clause(feature, operator, value))
    if index2class:
        return Rule(output_classes[int(path[-1])], *clauses)
    else:
        return Rule((path[-1]), *clauses)
        
def decision_tree2rule_list(tree:DecisionTreeClassifier, feature_names, output_classes, index2class=True) -> RuleList:
    return RuleList(
        output_classes[0],
        *(path2rule(path, feature_names, output_classes, index2class=index2class) for path in decision_tree_path_iter(tree))
    )

'''
X: the data as a numpy matrix
y: the expected output as a numpy matrix
spaces: (feature, space) pairs where the space is a list of possible values if the feature is categorical
        numeric data may have any non-list space
        the order of the spaces must correspond to the order of the values in data
        the output must be the last feature in the sequence
'''
def create_initial_population(
        X:np.matrix,
        y:np.matrix,
        spaces,
        pop_size:int=10,
        max_rule_size:int=4,
        max_rule_count:int=16
    ) -> list[RuleList]:
    feature_names = [key for key, _ in spaces[:-1]]
    categorical_cols = []
    numeric_cols = []
    for index, pair in enumerate(spaces[:-1]):
        space = pair[1]
        if type(space) is list:
            categorical_cols.append(index)
        else:
            numeric_cols.append(index)
    rfc = RandomForestClassifier(n_estimators=pop_size, max_depth=max_rule_size, max_leaf_nodes=max_rule_count)
    if len(categorical_cols) > 0:
        # categorical data needs to be one hot encoded
        reordered_cols = categorical_cols + numeric_cols
        categorical_data = X[:, categorical_cols]
        numeric_data = X[:, numeric_cols]
        encoder = OneHotEncoder()
        encoder.fit(categorical_data)
        encoder.feature_names_in_ = [feature_names[i] for i in reordered_cols]
        enc_feature_names = list(encoder.get_feature_names_out()) + [feature_names[i] for i in numeric_cols]
        del encoder.feature_names_in_
        enc_categorical_data = encoder.transform(categorical_data).toarray()
        encoded_data = np.column_stack((enc_categorical_data, numeric_data))
        rfc.fit(encoded_data, y)
        rulelists = [decision_tree2rule_list(tree, enc_feature_names, rfc.classes_) for tree in rfc.estimators_]
    else:
        rfc.fit(X, y)
        rulelists = [decision_tree2rule_list(tree, feature_names, rfc.classes_) for tree in rfc.estimators_]
    for rulelist in rulelists:
        rulelist.attribute_names = feature_names
    return rulelists

def CART_decision_list(
        X:np.matrix,
        y:np.matrix,
        spaces
    ) -> RuleList:
    feature_names = [key for key, _ in spaces[:-1]]
    categorical_cols = []
    numeric_cols = []
    for index, pair in enumerate(spaces[:-1]):
        space = pair[1]
        if type(space) is list:
            categorical_cols.append(index)
        else:
            numeric_cols.append(index)
    rfc = DecisionTreeClassifier()
    if len(categorical_cols) > 0:
        # categorical data needs to be one hot encoded
        reordered_cols = categorical_cols + numeric_cols
        categorical_data = X[:, categorical_cols]
        numeric_data = X[:, numeric_cols]
        encoder = OneHotEncoder()
        encoder.fit(categorical_data)
        encoder.feature_names_in_ = [feature_names[i] for i in reordered_cols]
        enc_feature_names = list(encoder.get_feature_names_out()) + [feature_names[i] for i in numeric_cols]
        del encoder.feature_names_in_
        enc_categorical_data = encoder.transform(categorical_data).toarray()
        encoded_data = np.column_stack((enc_categorical_data, numeric_data))
        rfc.fit(encoded_data, y)
        rulelist = decision_tree2rule_list(rfc, enc_feature_names, rfc.classes_, index2class=False)
    else:
        rfc.fit(X, y)
        rulelist = decision_tree2rule_list(rfc, feature_names, rfc.classes_, index2class=False)
    rulelist.attribute_names = feature_names
    return rulelist

def accuracy(rulelist:RuleList, X, y) -> float:
    return sum(1 if exp == rulelist.apply(row) else 0 for exp, row in zip(y, X)) / len(y)

def interpretability(rulelist:RuleList) -> float:
    try:
        return 1 / sum(len(rule) for rule in rulelist)
    except ZeroDivisionError:
        return 0

"""
returns the indices of the rows in the data to which the rule applies
"""
def get_cover(attribute_names, rule:Rule, X) -> set[int]:
    output = set()
    for i, row in enumerate(X):
        if rule.apply(dict(zip(attribute_names, row))) is not None:
            output.add(i)
    return output
"""
returns one minus the mean relative overlap between rule covers
1 -> no overlap
0 -> full overlap (each row covered by all rules)
"""
def overlap_score(rulelist:RuleList, X) -> float:
    covers = [get_cover(rulelist.attribute_names, rule, X) for rule in rulelist]
    if len(covers) < 2:
        return 1
    overlap_sizes = [len(covers[i].intersection(covers[j])) for i in range(len(covers)) for j in range(i+1, len(covers))]
    mean_overlap = sum(overlap_sizes)/len(overlap_sizes)
    return 1 - mean_overlap/len(X)

def score(acc:float, interpretability:float) -> float:
    return acc * interpretability**3

'''
returns an index with probability defined by the provided list
'''
def probabilistic_selection(prob:list[float]) -> int:
    num = uniform(0, 1)
    i = 0
    while i < len(prob) and num >= prob[i]:
        i += 1
    return i

def select_pair(prob:list[float]) -> tuple[int]:
    return (probabilistic_selection(prob), probabilistic_selection(prob))

'''
creates probabilities used in rank selection
(divides the interval [0, 1] into pieces such that the width equals the probability)
'''
def create_rank_prob(pop_size):
    total_rank = sum(range(1, pop_size + 1))
    prob = []
    weight = 0
    for i in range(pop_size, 1, -1):
        weight += i
        prob.append(weight / total_rank)
    return prob

def selection(collection, prob, count):
    for _ in range(count):
        i1, i2 = select_pair(prob)
        yield (collection[i1], collection[i2])

def train_interpretability(
        encoding:Encoding,
        initial_population:list[RuleList],
        X,
        y,
        max_epochs:int=200,
        pop_inc=50,
        mutation_prob=0.01
    ):
    # inherit the attribute names from the initial population
    encoding.attribute_names = initial_population[0].attribute_names
    pop_size = len(initial_population)
    # (chromosome, score, accuracy, interpretability)
    pop = [
        (encoding.encode(ind), score(acc, interp), acc, interp)
        for ind, acc, interp
        in ((ind, accuracy(ind, X, y), interpretability(ind)) for ind in initial_population)
    ]
    pop.sort(key=lambda x : x[1], reverse=True)
    pareto_set = ParetoSet()
    for chrom, _, acc, interp in pop:
        pareto_set.add((Pareto(acc, interp), encoding.decode(chrom)))
    prob = create_rank_prob(len(pop))
    i = 0
    # continue until the best and worst individuals converge or the maximum amount of epochs have passed
    while i < max_epochs and pop[0][1] != pop[-1][1]:
        for item1, item2 in selection(pop, prob, pop_inc // 2):
            chrom1, chrom2 = encoding.crossover(item1[0], item2[0])
            encoding.mutate(chrom1, mutation_prob)
            encoding.mutate(chrom2, mutation_prob)
            rulelist1 = encoding.decode(chrom1)
            rulelist2 = encoding.decode(chrom2)
            acc1 = accuracy(rulelist1, X, y)
            acc2 = accuracy(rulelist2, X, y)
            interp1 = interpretability(rulelist1)
            interp2 = interpretability(rulelist2)
            pop.append((chrom1, score(acc1, interp1), acc1, interp1))
            pop.append((chrom2, score(acc2, interp2), acc2, interp2))
            pareto_set.add((Pareto(acc1, interp1), rulelist1))
            pareto_set.add((Pareto(acc2, interp2), rulelist2))
        pop.sort(key=lambda x : x[1], reverse=True)
        pop = pop[:pop_size]
        i += 1
    return (
        pareto_set,
        [(encoding.decode(chrom), score, acc) for chrom, score, acc, _ in pop]
    )

def train_accuracy_interpretability(
        encoding:Encoding,
        initial_population:list[RuleList],
        X,
        y,
        max_epochs:int=1000,
        timeout:int=100,
        pop_inc=50,
        mutation_prob=0.01,
        track_history=False,
        accuracy_score="accuracy",
        interpretability_score="size",
        crowding_distance=False
    ):
    accuracy_score = ["balanced_accuracy", "mean_f1", "accuracy"].index(accuracy_score)
    interpretability_score = ["size_overlap", "size"].index(interpretability_score)
    start_time = time.time()
    pop_size = len(initial_population)
    rank_prob = dict()
    # inherit the attribute names from the initial population
    encoding.attribute_names = initial_population[0].attribute_names
    acc_f = accuracy_functions[accuracy_score]
    inter_f = interpretability_functions[interpretability_score]
    pop = [(Pareto(acc_f(ind, X, y), inter_f(ind, X, y)), encoding.encode(ind)) for ind in initial_population]
    prev_pop_set = {e[0] for e in pop}
    pop = pareto_rank_sort(pop)
    if crowding_distance:
        for i in range(len(pop)):
            pop[i] = crowding_distance_sort(pop[i])
    prev_pareto_set = {e[0] for e in pop[0]}
    if track_history:
        pareto_history = []
    i = 0
    # continue until timeout is reached or the maximum amount of epochs have passed
    while i < max_epochs:
        if crowding_distance:
            # flatten the population
            _pop = [e for group in pop for e in group]
        else:
            _pop = pop
        if len(_pop) not in rank_prob:
            rank_prob[len(_pop)] = create_rank_prob(len(_pop))
        for selection1, selection2 in selection(_pop, rank_prob[len(_pop)], pop_inc // 2):
            if crowding_distance:
                # selection == chromosome
                chrom1, chrom2 = encoding.crossover(selection1[1], selection2[1])
            else:
                # selection == group
                chrom1, chrom2 = encoding.crossover(choice(selection1)[1], choice(selection2)[1])
            encoding.mutate(chrom1, mutation_prob)
            encoding.mutate(chrom2, mutation_prob)
            rulelist1 = encoding.decode(chrom1)
            rulelist2 = encoding.decode(chrom2)
            pop[-1].append((Pareto(acc_f(rulelist1, X, y), inter_f(rulelist1, X, y)), chrom1))
            pop[-1].append((Pareto(acc_f(rulelist2, X, y), inter_f(rulelist2, X, y)), chrom2))
        pop = pareto_rank_sort([e for group in pop for e in group])
        if crowding_distance:
            for j in range(len(pop)):
                pop[j] = crowding_distance_sort(pop[j])
        if len(pop) == 1:
            # pareto set exceeds current population size
            # population size increases
            pop_size = len(pop)
        else:
            # cull the population
            while sum(len(group) for group in pop) > pop_size:
                pop[-1].pop()
                if len(pop[-1]) == 0:
                    pop.pop()
        if track_history:
            pareto_set = {e[0] for e in pop[0]}
            if (
                len(pareto_set) != len(prev_pareto_set) or
                len(pareto_set.intersection(prev_pareto_set)) != len(pareto_set)
            ):
                # pareto set has changed
                pareto_history.append((i, pareto_set))
                prev_pareto_set = pareto_set
        i += 1
        if i % timeout == 0:
            pop_set = {e[0] for group in pop for e in group}
            if len(pop_set) == len(prev_pop_set) and len(pop_set.intersection(prev_pop_set)) == len(pop_set):
                # the population hasn't changed in 'timeout' epochs
                break
            prev_pop_set = pop_set
    end_time = time.time()
    output = {
        "population": pop,
        "epochs": i,
        "time": round(end_time - start_time)
    }
    if track_history:
        output["pareto_history"] = pareto_history
    return output