from scipy import spatial
import random
import math
import numpy as np
import collections


class PersonalityTemplate:

    def generate_random_personality(self, personality, accuracy):
        new_personality = {}
        for d in self.dimensions:
            for x in range(1, self.number_of_facets + 1):
                key = d + str(x)
                new_personality[key] = accuracy_value(personality[key], accuracy)
        return new_personality

    def facet_percentage(self, facets, personality):
        total = self.max_facet_score * len(facets)
        score = 0
        for f in facets:
            score += (personality[f] if personality[f] is not None else 0)
        return score/total

    def set_all_facets_in_dimension(self, personality, dimension, value):
        for x in range(1, self.number_of_facets + 1):
            personality[dimension + str(x)] = value

    def set_values(self, personality, key_value_list):
        for kv in key_value_list:
            facet_or_dimension_list, value = kv
            for fd in facet_or_dimension_list:
                if len(fd) == 1:
                    #dimension
                    self.set_all_facets_in_dimension(personality, fd, value)
                elif len(fd) == 2:
                    #facet
                    personality[fd] = value

        return personality

    def dimension_total(self, personality, dimension_letter):
        tot = 0
        for x in range(1, self.number_of_facets + 1):
            tot += personality[dimension_letter + str(x)]
        return tot

    def dimension_percentage(self, personality, dimension_letter):
        tot = self.dimension_total(personality, dimension_letter)
        return tot / (self.number_of_facets * self.max_facet_score)

    def personality_facet_similarity(self, personality, wanted_facets, unwanted_facets):
        listx = []
        listy = []
        for key in wanted_facets:
            listy.append(self.max_facet_score)
            listx.append(personality[key])
        for key in unwanted_facets:
            listy.append(1)
            listx.append(personality[key])
        result = 1 - spatial.distance.cosine(listx, listy)
        return result

    def random_personality(self):
        personality = {}
        for d in self.dimensions:
            for x in range(1, self.number_of_facets + 1):
                personality[d + str(x)] = random.randrange(1,self.max_facet_score)
        return personality


    def personality_similarity(self, personality_one, personality_two):
        list1 = []
        list2 = []
        for key in personality_one:
            list1.append(personality_one[key])
            list2.append(personality_two[key])
        result = 1 - spatial.distance.cosine(list1, list2)
        return round(result,5)

    def get_personality(self, dimension_to_facet_values):
        personality_dictionary = {}
        for dimension in dimension_to_facet_values:
            dimension_values = dimension_to_facet_values[dimension]
            for i in range(len(dimension_values)):
                fv = dimension_values[i]
                personality_dictionary[dimension + str(i + 1)] = fv
        return personality_dictionary

    def __init__(self, dimensions, number_of_facets, max_facet_score):
        self.dimensions = dimensions
        self.number_of_facets = number_of_facets
        self.max_facet_score = max_facet_score


class HexacoPersonality(PersonalityTemplate):

    @staticmethod
    def psychopath():
        return {
            "H1": 10,
            "H2": 10,
            "H3": 10,
            "H4": 10,
            "E1": 10,
            "E2": 10,
            "E3": 10,
            "E4": 10,
            "X1": 50,
            "X2": 50,
            "X3": 50,
            "X4": 50,
            "A1": 10,
            "A2": 10,
            "A3": 10,
            "A4": 10,
            "C1": 90,
            "C2": 90,
            "C3": 90,
            "C4": 90,
            "O1": 50,
            "O2": 50,
            "O3": 50,
            "O4": 90,
        }

    @staticmethod
    def average():
        return {
            "H1": 50,
            "H2": 50,
            "H3": 50,
            "H4": 50,
            "E1": 50,
            "E2": 50,
            "E3": 50,
            "E4": 50,
            "X1": 50,
            "X2": 50,
            "X3": 50,
            "X4": 50,
            "A1": 50,
            "A2": 50,
            "A3": 50,
            "A4": 50,
            "C1": 50,
            "C2": 50,
            "C3": 50,
            "C4": 50,
            "O1": 50,
            "O2": 50,
            "O3": 50,
            "O4": 50,
        }

    def text(self, personality):
        text = ""
        for j in range(len(self.dimensions)):
            c = self.dimensions[j]
            dimension = '('
            for i in range(1, self.number_of_facets + 1):
                p = personality[c + str(i)]
                t = '0' + str(p) if p < 10 else str(p)
                dimension += t
                if i < self.number_of_facets:
                    dimension += ","
            dimension += ')'
            text += dimension
            if j < 4:
                text += ' | '
        return text

    def as_list(self, personality):
        lst = []
        for d in self.dimensions:
            for i in range(1,self.number_of_facets+1):
                k = d + str(i)
                lst.append(personality[k])
        return lst

    def dimension_text(self,personality):
        text = "("
        for i in range(len(self.dimensions)):
            d = self.dimensions[i]
            text += str(round(100*self.dimension_percentage(personality,d)))
            if i < len(self.dimensions)-1:
                text += ","
        text += ")"
        return text

    def __init__(self):
        PersonalityTemplate.__init__(self, ['H', 'E', 'X', 'A', 'C', 'O'], 4, 100)

    def get_personality(self, h, e, x, a, c, o):
        personality_dict = {'H': h, 'E': e, 'X': x, 'A': a, 'C': c, 'O': o}
        return PersonalityTemplate.get_personality(self, personality_dict)


def apply_prob_distribution(score, start_prob, score_dif, prob_incr):
    p = start_prob
    i = 1
    while p < 100:
        p = start_prob * pow(1 + prob_incr / 100, i-1)
        incr_p = 0.5*p
        decr_p = 0.5*p
        incr_s = score * pow(1 + score_dif/100, i)
        decr_s = score * pow(1 - score_dif/100, i)

        if decr_s < 1 and incr_s > 34:
            return score

        if decr_s < 1:
            incr_p = p
            decr_p = 0
        elif incr_s > 34:
            incr_p = 0
            decr_p = p

        rand = random.randrange(0, 100)
        if rand < incr_p:
            return incr_s
        elif rand < incr_p + decr_p:
            return decr_s

        i += 1
    return score


def random_boolean_variable(probability):
    return random.random() < probability


def contains(lists_of_lists, list):
    counter = collections.Counter(list)
    for a in lists_of_lists:
        if collections.Counter(a) == counter:
            return True
    return False


def accuracy_value(value, accuracy):
    # pick 5 random values between 1 and value
    weights = []
    values = []

    dif = 1

    if value < 10:
        if value == 0:
            value = 0.1
        value *= 10
        dif = 10

    value = 1 + int(value)

    for x in range(5):
        #print(value)
        values.append(random.randrange(1,value))
        weights.append((1-accuracy)/5)

    values.append(value)
    weights.append(accuracy)
    return random.choices(population=values,weights=weights,k=1)[0] / dif

def similarity(first, second):
    return round(1 - spatial.distance.cosine([first], [second]),5)

def logistic_update_status_weight(n, c, m, z):
    return c - (c-m) / (1 + math.exp(-0.1 * (100 * n / z - 50)))