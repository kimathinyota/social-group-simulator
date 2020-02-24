from scipy import spatial
import random
import math
import numpy as np


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
        return result

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


def random_boolean_variable(truth_likelyhood):
    k = random.randrange(1000)

    v = truth_likelyhood* 1000

    if k > v:
        return False

    return True


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


def logistic_update_status_weight(n, c, m, z):
    return c - (c-m) / (1 + math.exp(-0.1 * (100 * n / z - 50)))