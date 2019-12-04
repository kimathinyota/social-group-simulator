from scipy import spatial
import random
import math


def personality_dictionary(n, e, o, c, a):
    return {
                'N1': n[0],
                'N2': n[1],
                'N3': n[2],
                'N4': n[3],
                'N5': n[4],
                'N6': n[5],
                'E1': e[0],
                'E2': e[1],
                'E3': e[2],
                'E4': e[3],
                'E5': e[4],
                'E6': e[5],
                'O1': o[0],
                'O2': o[1],
                'O3': o[2],
                'O4': o[3],
                'O5': o[4],
                'O6': o[5],
                'C1': c[0],
                'C2': c[1],
                'C3': c[2],
                'C4': c[3],
                'C5': c[4],
                'C6': c[5],
                'A1': a[0],
                'A2': a[1],
                'A3': a[2],
                'A4': a[3],
                'A5': a[4],
                'A6': a[5]
    }


def dimension_total(personality,dimension_letter):
    tot = 0
    for x in range(1, 7):
        tot += personality[dimension_letter + str(x)]
    return tot


def dimension_percentage(personality, dimension_letter):
    tot = dimension_total(personality, dimension_letter)
    return tot/(6*35)


def personality_facet_similarity(personality, wanted_facets, unwanted_facets):
    listx = []
    listy = []
    for key in wanted_facets:
        listy.append(35)
        listx.append(personality[key])
    for key in unwanted_facets:
        listy.append(1)
        listx.append(personality[key])
    result = 1 - spatial.distance.cosine(listx, listy)
    return result


def personality_similarity(personality_one, personality_two):
    list1 = []
    list2 = []
    for key in personality_one:
        list1.append(personality_one[key])
        list2.append(personality_two[key])
    result = 1 - spatial.distance.cosine(list1, list2)
    return result


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


def action_from_personality(personality):
    action = {}
    for key in personality:
        facet_score = personality[key]
        act = apply_prob_distribution(facet_score, 55, 7.5, 7)
        action[key] = act
    return action


def logistic_update_status_weight(n, c, m, z):
    return c - (c-m)/(1 + math.exp(-0.1*(100*n/z - 50)))


def estimate_personality(action_old, action_observed, weight):
    action_new = {}
    for key in action_observed:
        old = action_old[key]
        observed = action_observed[key]
        score = weight * old + observed * (1 - weight)
        action_new[key] = score
    return action_new


