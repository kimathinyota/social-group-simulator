import matplotlib.pyplot as plt
import networkx as nx
from src.Agent import *
from src.GoldMiningEnvironment import Friendship, Mentorship, Theft, Help
from networkx.algorithms.community import k_clique_communities, greedy_modularity_communities
from networkx.algorithms.community.label_propagation import *
from src.Helper import *
import difflib
import statistics
from networkx.algorithms.centrality import *


class GroupAgent:

    def get_agents(self):
        return self.agents

    def memberOf(self, agent):
        return agent in self.agents

    def __repr__(self):
        return "G" + str(self.agents)

    def __init__(self, agents):
        self.agents = agents
        self.generation_id = str(sorted(agents))


class PowerMetrics:

    @staticmethod
    def get_counts(list_of_list):
        aidmap = {}
        for aid in [elem for lst in list_of_list for elem in lst]:
            aidmap[aid] = 1 if aid not in aidmap else aidmap[aid] + 1
        return aidmap

    @staticmethod
    def merge(type_infos):
        # t -> Strength, Percentage, [Agent Counts]
        type_to_value = {'None': (0, 0, [], 0),
                         'Democracy': (0, 0, [], 0),
                         'Dictatorship': (0, 0, [], 0),
                         'RulingClass': (0, 0, [], 0),
                         'ServantClass': (0, 0, [], 0),
                         'Slavery': (0, 0, [], 0)}

        for type_info in type_infos:
            if type_info is not None:
                t, cd, a, s = type_info
                ts, tp, ta, c = type_to_value[t]
                ts += cd if not None else 0 + s if not None else 0
                tp += 1
                ta += [a]
                c += 1
                type_to_value[t] = (ts, tp, ta, c)

        for t in type_to_value:
            s, p, a, c = type_to_value[t]
            aidmap = PowerMetrics.get_counts(a)
            p /= len(type_infos)
            s = s / c if c > 0 else s

            type_to_value[t] = (s, p, aidmap)

        return type_to_value

    def __repr__(self):
        return str((self.type_info, self.std))

    def type_of_distribution(self, class_division_thresh=1.5, democracy_thresh=0.2, ruling_class_thresh=0.2,
                             servant_class_thresh=0.2):

        valuesgo = [v for v in self.agent_to_power_distribution.values() if v > 0]
        if len(valuesgo) == 0 or len(self.agent_to_power_distribution.keys()) < 2:
            return None
        lowest = min(valuesgo)
        d = self.std/lowest
        if d < democracy_thresh:
            return "Democracy", (1 - d), list(self.agent_to_power_distribution.keys()), None
        positives, negatives = [], []
        for agent in self.agent_to_deviation:
            v = self.agent_to_deviation[agent]
            v = v/self.std
            if v > 0:
                positives.append((agent, v))
            else:
                negatives.append((agent, v))

        ap = sum([v for (a,v) in positives]) / len(positives)
        an = sum([v for (a,v) in negatives]) / len(negatives)
        division_strength = abs(ap - an)

        largest = max(self.agent_to_power_distribution.values())

        if division_strength > class_division_thresh and len(positives) < len(negatives):
            # Contender for Ruling Class
            p = ruling_class_thresh * self.std
            agents = [agent for (agent, v) in positives if self.agent_to_power_distribution[agent] > (largest - p)]
            if len(agents) == 1:
                # Dictatorship
                dictator_strength = self.agent_to_deviation[agents[0]] / self.std
                return "Dictatorship", division_strength, [agents[0]], dictator_strength
            return "RulingClass", division_strength, agents, None

        if division_strength > class_division_thresh and len(negatives) <= len(positives):
            # Contender for Servant Class
            p = servant_class_thresh * self.std
            agents = [agent for (agent, v) in negatives if self.agent_to_power_distribution[agent] < (lowest + p)]

            if len(agents) == 1:
                # Slavery
                slavery_strength = self.agent_to_deviation[agents[0]] / self.std
                return "Slavery", division_strength, [agents[0]], abs(slavery_strength)
            return "ServantClass", division_strength, agents, None

        closeness_to_distribution = min(0, 1-d, (class_division_thresh - division_strength)/class_division_thresh)

        return 'None', closeness_to_distribution, list(self.agent_to_power_distribution.keys()), None

    def __init__(self, agent_to_power_distribution, round=None, class_division_thresh=1.5, democracy_thresh=0.2,
                 ruling_class_thresh=0.2, servant_class_thresh=0.2):
        self.agent_to_power_distribution = agent_to_power_distribution
        self.std = None
        self.mean = None
        self.agent_to_deviation = None
        self.type_info = None
        if len(self.agent_to_power_distribution) >= 2:
            self.std = statistics.stdev(list(self.agent_to_power_distribution.values()))
            self.mean = statistics.mean(list(self.agent_to_power_distribution.values()))
            self.agent_to_deviation = {a: (self.agent_to_power_distribution[a] - self.mean)
                                       for a in agent_to_power_distribution}

            self.type_info = self.type_of_distribution(class_division_thresh, democracy_thresh, ruling_class_thresh,
                                                       servant_class_thresh)
        self.round = round


class PowerStability:

    def __repr__(self):
        return str(self.stability_info)

    @staticmethod
    def merge(stability_info_data):
        total_stability_values = []
        type_to_value = {'None': ([], [], [], 0),
                         'Democracy':([], [], [], 0),
                         'Dictatorship':([], [], [], 0),
                         'RulingClass':([], [], [], 0),
                         'ServantClass':([], [], [], 0),
                         'Slavery':([], [], [], 0)}
        for stability_info in stability_info_data:
            if stability_info is None:
                for t in type_to_value:
                    a, b, c, d = type_to_value[t]
                    type_to_value[t] = (a + [0], b + [0], c, d + 0)
                total_stability_values.append(0)
            else:
                total, ttv = stability_info
                total_stability_values.append(total)
                for t in type_to_value:
                    a1, b1, c1, d1 = ttv[t]
                    b1 = b1/ d1 if d1 > 0 else b1
                    a, b, c, d = type_to_value[t]
                    type_to_value[t] = (a + [a1], b + [b1], c + c1, d + d1)

        for t in type_to_value:
            a, b, c, d = type_to_value[t]
            aidmap = PowerMetrics.get_counts(c)
            ams = SocialAnalysis.get_mean_similarity(a)
            bms = SocialAnalysis.get_mean_similarity(b)
            type_to_value[t] = (ams, bms, aidmap, d)

        total = SocialAnalysis.get_mean_similarity(total_stability_values)

        return total, type_to_value

    def stability_information(self):
        if len(self.stability_metrics) == 0:
            return None
        type_to_info = {'None': (0, 0, [], 0), 'Democracy': (0, 0, [], 0), 'Dictatorship': (0, 0, [], 0),
                        'RulingClass': (0, 0, [], 0), 'ServantClass': (0, 0, [], 0), 'Slavery': (0, 0, [], 0)}
        length = self.last_round - self.starting_round

        total = 0

        for entry_metric in self.stability_metrics:
            entry, metric = entry_metric
            p, n, avg, avg_power = entry
            total += (n-p)
            if metric.type_info is not None:
                t, cs, ags, ts = metric.type_info
                total_stab, type_strength, agents, count = type_to_info[t]
                total_stab += (n-p)
                type_strength += cs if not None else 0 + ts if not None else 0
                count += 1
                if ags is not None:
                    agents.append(ags)

                type_to_info[t] = (total_stab, type_strength, agents, count)

        for t in type_to_info:
            total_stab, type_strength, agents, count = type_to_info[t]
            type_to_info[t] = (total_stab/length, type_strength, agents, count)

        total /= length

        return total, type_to_info

    def __init__(self, round_to_agent_to_power, agents, thresh=0.8, class_division_thresh=1.5, democracy_thresh=0.2,
                 ruling_class_thresh=0.2, servant_class_thresh=0.2):

        self.class_division_thresh = class_division_thresh
        stability = SocialAnalysis.hierarchy_stablity(round_to_agent_to_power, agents, thresh)
        self.starting_round = min(round_to_agent_to_power.keys())
        self.last_round = max(round_to_agent_to_power.keys())
        self.stability_metrics = []

        for entry in stability:
            p, n, avg, average_power = entry
            metric = PowerMetrics(average_power, None, class_division_thresh, democracy_thresh, ruling_class_thresh,
                                  servant_class_thresh)
            self.stability_metrics.append((entry, metric))

        self.stability_info = self.stability_information()


# Purpose: Have access to all of the analysis data - it can be combined with others in a database to find correlations
class SocialMetrics:

    def __repr__(self):
        return str((("Anti-Social", "Productivity", "Cooperation", "Synergy"),
                    (self.anti_social, self.productivity, self.cooperative, self.introduced_gold)))

    def __init__(self, interactions_to_earnings, agent_to_mined, agents):
        self.anti_social, self.productivity, self.cooperative = SocialAnalysis.group_metrics(interactions_to_earnings)
        self.introduced_gold = SocialAnalysis.introduced_wealth(interactions_to_earnings, agent_to_mined, agents)


class SocialStructureAnalysis:

    def __init__(self, social_structure, start_round, end_round, round_to_interactions_to_earnings,
                 round_to_agent_to_wealth, round_to_agent_mines, contains_all_agents=False):

        # General
        all_interactions_to_earnings = {}
        agent_to_mined = {}
        agent_to_wealth = {}
        round_to_agent_to_power = {}
        for round in range(start_round, end_round + 1):
            interactions_to_earnings = round_to_interactions_to_earnings[round]
            c = round_to_agent_mines[round]
            for agent in c:
                res = c[agent]
                agent_to_mined[agent] = agent_to_mined[agent] + res if agent in agent_to_mined else res

            c = round_to_agent_to_wealth[round]
            for agent in c:
                res = c[agent]
                agent_to_wealth[agent] = agent_to_wealth[agent] + res if agent in agent_to_wealth else res
            interactions = list(interactions_to_earnings.keys())
            limited = interactions
            if not contains_all_agents:
                limited = SocialAnalysis.limit_interactions_to_agents(interactions, social_structure)

            round_to_agent_to_power[round] = SocialAnalysis.power_distribution(limited, agent_earnings=c)
            for interaction in interactions_to_earnings:
                all_interactions_to_earnings[interaction.copy()] = interactions_to_earnings[interaction]

        interactions = SocialAnalysis.limit_interactions_to_agents(list(all_interactions_to_earnings.keys()), social_structure)

        self.general_social_metrics = SocialMetrics({i: all_interactions_to_earnings[i] for i in interactions},
                                                    agent_to_mined,
                                                    social_structure)

        self.wealth_stability = SocialAnalysis.hierarchy_stablity(round_to_agent_to_wealth, social_structure)

        self.wealth_stability_info = SocialAnalysis.get_hierarchy_stability_info(start_round, end_round, self.wealth_stability)

        self.general_power_metric = PowerMetrics(SocialAnalysis.power_distribution(interactions, agent_earnings=agent_to_wealth))

        self.power_stability = PowerStability(round_to_agent_to_power, social_structure)


class CompetingSocialStructureAnalysis:

    def __init__(self, social_structures, round_to_interactions_to_earnings, round_to_agent_to_earnings,
                 round_to_agent_mines):

        rie = {}
        rae = {}
        ram = {}
        all_groups = []
        for round in round_to_interactions_to_earnings:
            ite = round_to_interactions_to_earnings[round]
            ate = round_to_agent_to_earnings[round]
            atm = round_to_agent_mines[round]

            subsets = [group for (group, start, end) in social_structures if start <= round <= end]
            non_subsets = SocialAnalysis.non_subset_groups(subsets)

            if len(non_subsets) > 0:
                ite, atm, ate, gs = SocialAnalysis.social_groups_as_interactions(ite, non_subsets, atm, ate)
                all_groups += [g.generation_id for g in gs if g.generation_id not in all_groups]
                rie[round] = ite
                rae[round] = ate
                ram[round] = atm
            else:
                rie[round], rae[round], ram[round] = {}, {}, {}

        self.analysis = SocialStructureAnalysis(all_groups, min(list(rie.keys())), max(list(rie.keys())), rie, rae, ram, True)


class SocialAnalysisResult:

    def __repr__(self):
        return str(self.get_data())

    @staticmethod
    def similar_structures(group1, group2, sim_thresh=0.8):
        # Return (g,g2) pairs with sim > 0.8
        found_groups = []
        group_found = {}
        first = group1 if len(group1) > len(group2) else group2
        second = group2 if first == group1 else group1
        for g in first:
            for g2 in second:
                sim = SocialAnalysis.sim(g, g2)
                if sim >= sim_thresh:
                    group_found[str(g)] = group_found[str(g)] + 1 if str(g) in group_found else 1
                    # similar enough to be near equal
                    if sim == 1:
                        found_groups += [g]
                    else:
                        found_groups += [g, g2]
        score = sum(group_found.values()) / len(first) if len(first) > 0 else 0
        return found_groups, score

    @staticmethod
    def social_structures_consistency(social_structure_data_sets):
        # ([A,B,C], [1,2]), ([B,C,D,[1,2])
        # Find similar groupings identified in each data set
        groups = [[g[0] for g in group] for group in social_structure_data_sets]
        avg_similarity = 0
        c = 0
        social_structures = []

        for i in range(len(groups)):
            g = groups[i]
            for j in range(i+1, len(groups)):
                g2 = groups[j]
                found, s = SocialAnalysisResult.similar_structures(g, g2)
                social_structures += [g for g in found if g not in social_structures]
                avg_similarity += s
                c += 1
        consistency_score = avg_similarity / c if c > 0 else 0
        return social_structures, consistency_score

    @staticmethod
    def merge(data_set):
        anti, prod, coop, ig = [], [], [], []
        power_stability_info = []
        wealth_stability = []
        power_metrics = []
        atv = {}
        atg = {}
        social_structures = []
        between_average_metrics = [[],[],[],[]]
        canti, cprod, ccoop, cig = [], [], [], []
        comp_type_to_count = {'Democracy': 0, 'RulingClass':0, 'ServantClass':0, 'Dictatorship':0, 'Slavery':0, 'None': 0}
        avg_comp_stability = []
        for data in data_set:
            sm = data["SocialMetrics"]
            a, p, c, i = sm
            anti, prod, coop, ig = anti + [a], prod + [p], coop + [c], ig + [i]
            power_stability_info.append(data["PowerStability"])
            wealth_stability.append(data["WealthStability"][0])
            power_metrics.append(data["PowerMetric"])
            ahp = data["AgentToHierarchyPosition"]
            if ahp is not None:
                for a in ahp:
                    atv[a] = ahp[a] if a not in atv else ahp[a] + atv[a]
            social_structures.append(data["SocialStructures"])
            agroup = data["AgentToGroupCount"]
            for a in agroup:
                atg[a] = agroup[a] if a not in atg else atg[a] + agroup[a]

            sgm = data["SocialGroupMetrics"]
            avgs, rangs = sgm
            a, p, c, i = 0, 0, 0, 0
            if avgs is not None:
                a, p, c, i = avgs
            al, pl, cl, il = between_average_metrics
            al, pl, cl, il = al + [a], pl + [p], cl + [c], il + [i]
            between_average_metrics = [al, pl, cl, il]

            gps = data["GroupPowerStability"]

            if gps is not None:
                avg_stab, tti = gps
                avg_comp_stability.append(avg_stab)
                for t in tti:
                    comp_type_to_count[t] += 1
            else:
                avg_comp_stability.append(0)

            csm = data["CompetingGroupMetrics"]
            if csm is None:
                a, p, c, i = 0, 0, 0, 0
            else:
                a, p, c, i = csm
            canti, cprod, ccoop, cig = canti + [a], cprod + [p], ccoop + [c], cig + [i]

        return {
            "SocialMetrics": [SocialAnalysis.get_mean_similarity(v) for v in [anti, prod, coop, ig]],
            "PowerStability": PowerStability.merge(power_stability_info),
            "WealthStability": SocialAnalysis.get_mean_similarity(wealth_stability),
            "PowerMetric": PowerMetrics.merge(power_metrics),
            "AgentToHierarchyPosition": atv,
            "SocialStructures": SocialAnalysisResult.social_structures_consistency(social_structures),
            "AgentToGroupCount": atg,
            "SocialGroupMetrics": [SocialAnalysis.get_mean_similarity(v) for v in between_average_metrics],
            "GroupPowerStability": [SocialAnalysis.get_mean_similarity(avg_comp_stability), comp_type_to_count],
            "CompetingGroupMetrics": [SocialAnalysis.get_mean_similarity(v) for v in [canti, cprod, ccoop, cig]]
        }

    def get_data(self):
        g = self.general.general_social_metrics
        return {"SocialMetrics": [g.anti_social, g.productivity, g.cooperative, g.introduced_gold],
                "PowerStability": self.general.power_stability.stability_info,
                "WealthStability": self.general.wealth_stability_info,
                "PowerMetric": self.general.general_power_metric.type_info,
                "AgentToHierarchyPosition": self.agent_to_hierarchy_position,
                "SocialStructures": self.social_structures,
                "AgentToGroupCount": self.agents_to_number_of_social_structures,
                "SocialGroupMetrics": [self.between_social_average_metrics, self.between_social_range_metrics],
                "GroupPowerStability": self.average_power_stability,
                "CompetingGroupMetrics": self.competing_social_metrics,
                "CompetingPowerStability": [self.competing_power_stability_percentage,
                                            self.competing_power_stability_counts]
                }

    def __init__(self, round_to_interactions_to_earnings, round_to_agent_to_earnings, round_to_agent_to_mines, agents,
                 min_community_size=3, sim_thresh=0.8, stability_thresh=3):

        rounds = sorted(list(round_to_agent_to_earnings.keys()))
        self.general = SocialStructureAnalysis(agents, rounds[0], rounds[len(rounds)-1],
                                               round_to_interactions_to_earnings, round_to_agent_to_earnings,
                                               round_to_agent_to_mines, True)

        # Find social structures

        rte = {r: list(round_to_interactions_to_earnings[r].keys()) for r in round_to_interactions_to_earnings}
        self.social_structures = SocialAnalysis.find_social_structures(rte, min_community_size, sim_thresh,
                                                                       stability_thresh)
        self.general_wealth_stability = None
        self.agent_to_hierarchy_position = None
        if len(self.general.wealth_stability) > 0:
            self.general_wealth_stability = self.general.wealth_stability_info
            self.wealth_stability_percentage = 0
            percentage, atv = self.general_wealth_stability
            self.wealth_stability_percentage = percentage
            self.agent_to_hierarchy_position = {a: 0 for a in agents}
            for a in atv:
                p = atv[a]
                self.agent_to_hierarchy_position[a] += p

        self.social_structure_analysis = []
        self.competing_social_analysis = None

        start_round = min(round_to_interactions_to_earnings.keys())
        end_round = max(round_to_interactions_to_earnings.keys())

        self.agents_to_number_of_social_structures = {a: 0 for a in agents}
        self.competing_power_stability_counts = None
        self.competing_power_stability_percentage = 0
        self.competing_social_metrics = None
        self.between_social_average_metrics = None
        self.between_social_range_metrics = None
        self.average_power_stability = None

        if len(self.social_structures) > 0:
            # Within Social Structure Analysis
            avg_group_stab = 0
            avg_type_to_info = {'Democracy': 0, 'RulingClass':0, 'ServantClass':0, 'Dictatorship':0, 'Slavery':0, 'None': 0}
            avg_anti, avg_prod, avg_coop, avg_synergy = 0, 0, 0, 0
            antis, coops, prods, synergys = [], [], [], []
            for struct in self.social_structures:
                group, start, end = struct
                avg_group_stab += (end - start)/(end_round - start_round)
                analysis = SocialStructureAnalysis(group, start, end, round_to_interactions_to_earnings,
                                                   round_to_agent_to_earnings, round_to_agent_to_mines)
                sm = analysis.general_social_metrics
                avg_anti += sm.anti_social/len(self.social_structures)
                avg_coop += sm.cooperative/len(self.social_structures)
                avg_prod += sm.productivity/len(self.social_structures)
                avg_synergy += sm.introduced_gold/len(self.social_structures)

                antis.append(sm.anti_social)
                coops.append(sm.cooperative)
                prods.append(sm.productivity)
                synergys.append(sm.introduced_gold)

                for agent in group:
                    self.agents_to_number_of_social_structures[agent] += 1

                res = analysis.power_stability.stability_info
                if res is not None:
                    tot, type_to_info = res
                    for t in type_to_info:
                        c = type_to_info[t][3] / len(self.social_structures)
                        avg_type_to_info[t] += c

                self.social_structure_analysis.append((struct, analysis))

            avg_group_stab /= len(self.social_structures)

            self.average_power_stability = [avg_group_stab, avg_type_to_info]

            self.between_social_average_metrics = [avg_anti, avg_prod, avg_coop, avg_synergy]
            self.between_social_range_metrics = [max(antis) - min(antis), max(prods) - min(prods), max(coops) - min(coops),
                      max(synergys) - min(synergys)]

            # Competing
            self.competing_social_analysis = CompetingSocialStructureAnalysis(self.social_structures,
                                                                              round_to_interactions_to_earnings,
                                                                              round_to_agent_to_earnings,
                                                                              round_to_agent_to_mines)

            res = self.competing_social_analysis.analysis

            csm = res.general_social_metrics
            psi = res.power_stability.stability_info
            if psi is not None:
                percent, type_to_info = psi
                self.competing_power_stability_counts = {t: type_to_info[t][3] for t in type_to_info}
                self.competing_power_stability_percentage = percent
            self.competing_social_metrics = [csm.anti_social, csm.productivity, csm.cooperative, csm.introduced_gold]



class SocialAnalysis:

    @staticmethod
    def get_mean_similarity(values):
        if len(values) == 0:
            return None, None
        mean, std = SocialAnalysis.get_mean_range(values)
        return mean, (1 - std/mean) if mean > 0 else 1

    @staticmethod
    def get_mean_range(values):
        if len(values) == 0:
            return None, None
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        return mean, std

    @staticmethod
    def get_hierarchy_stability_info(start, end, hierarchy_stability):
        length = (end - start)
        tot = 0
        most_stable_hierarchy = None
        temp = 0
        for entry in hierarchy_stability:
            s, e, avg, atv = entry
            p = (e - s) / length * avg
            if p > temp:
                temp = p
                most_stable_hierarchy = atv
            tot += p
        return tot, most_stable_hierarchy


    @staticmethod
    def get_network(interactions, interaction_to_weight={}, default_weight=1):
        g = nx.MultiGraph()
        for interaction in interactions:
            p = interaction.proactive_agent
            r = interaction.reactive_agent
            d = default_weight
            w1, w2 = interaction_to_weight[interaction] if interaction in interaction_to_weight else (d, d)
            if isinstance(interaction, Friendship):
                g.add_edge(p.generation_id, r.generation_id, r="F", weight=w1)
                g.add_edge(r.generation_id, p.generation_id, r="F", weight=w2)
            elif isinstance(interaction, Mentorship):
                g.add_edge(p.generation_id, r.generation_id, r="M", weight=w1)
                g.add_edge(r.generation_id, p.generation_id, r="M", weight=w2)
            elif isinstance(interaction, Theft):
                g.add_edge(p.generation_id, r.generation_id, r="T", weight=w1)
                g.add_edge(r.generation_id, p.generation_id, r="T", weight=w2)
            elif isinstance(interaction, Help):
                g.add_edge(p.generation_id, r.generation_id, r="H", weight=w1)
                g.add_edge(r.generation_id, p.generation_id, r="H", weight=w2)

        return g

    @staticmethod
    def find_social_structures(round_to_interactions, min_community_size=3, sim_thresh=0.8, stability_thresh=3):
        round_communities = {}
        rounds = []
        # Find communities for interactions in each round
        for round in round_to_interactions:
            rounds.append(round)
            interactions = round_to_interactions[round]
            if len(interactions) > min_community_size:
                # convert interactions for current round into NetworkX
                network = SocialAnalysis.get_network(interactions)
                # Below are other community finding algorithms that can be used:
                #   kcc = list(k_clique_communities(network, 3))
                #   gmc = list(greedy_modularity_communities(network))
                # Chose below algorithm because it worked better when i tested it on other networks
                alc = list(asyn_lpa_communities(network))
                round_communities[round] = [list(a) for a in alc]
            else:
                round_communities[round] = []
        rounds.sort()
        rcsim = []

        # Compare directly consectutive found communities and store in rcsim
        for i in range(len(rounds) - 1):
            round = rounds[i]
            r1 = round_communities[round]
            r2 = round_communities[round + 1]
            for c in r1:
                for c2 in r2:
                    # compares how similar communities c and c2 are
                    sim = SocialAnalysis.sim(c, c2)
                    if len(c) > 0 and len(c2) > 2:
                        rcsim.append(((round, c), (round + 1, c2), sim))

        # Only keep entries with a higher similarity than sim_thresh
        communities = [rcsim[i] for i in range(len(rcsim)) if rcsim[i][2] > sim_thresh]

        # Chain together communities
        # ((x, lst), (y,lst2), 1), ((y, lst2), (z,lst3), 1)
        # to ((x, lst), (y,lst2), 1), (z,lst3), 1)

        chains = [[a, b] for (a, b, s) in communities]

        i= 0
        while i < len(chains):
            first = chains[i]
            end = first[len(first) - 1]

            j = 0
            while j < len(chains):
                second = chains[j]
                start = second[0]

                if start == end:
                    rep = first[0:len(first) - 1] + second
                    chains[i] = rep
                    chains.remove(second)
                    j -= 1
                    i -= 1

                j += 1

            i += 1

        result = []

        for group in chains:
            start = min(group, key=lambda item: item[0])[0]
            end = max(group, key=lambda item: item[0])[0]
            res = set([])
            for g in group:
                res = res.union(g[1])
            res = list(res)
            result.append((res, start, end))
        return [(res,start,end) for (res,start,end) in result if (end-start) >= stability_thresh]

    @staticmethod
    def power_distribution(interactions, interaction_to_weight={}, default_weight=1, agent_earnings={}, should_draw=False):
        # For each round of interactions
        # Look at centrality of each agent involved in interaction
        network = SocialAnalysis.get_network(interactions, interaction_to_weight, default_weight)
        centrality = closeness_centrality(network)
        tot = 0
        for a in agent_earnings:
            tot += agent_earnings[a]
        earnings = {a: agent_earnings[a]/tot for a in agent_earnings}
        power = {a: ((earnings[a] if a in earnings else 0) + centrality[a])/2 for a in centrality}
        if should_draw:
            pos = nx.spring_layout(network)
            nx.draw(network, pos, with_labels=True)
            edge_labels = nx.get_edge_attributes(network, 'r')
            edge_labels = {(e[0], e[1]): edge_labels[e] for e in edge_labels}
            nx.draw_networkx_edge_labels(network, pos, edge_labels=edge_labels)
            plt.show()
        return power

    @staticmethod
    def test():
        agents = []
        for i in range(10):
            n = "AG" + str(i)
            agents.append(Agent.random(n,n))
        interactions = []
        interaction_to_weight = {}

        # a, b, c, d = agents[0], agents[1], agents[2], agents[3]
        #
        # interactions += [Theft(a,b, None), Help(c,b, None), Mentorship(c, d, None), Friendship(a, d, None)]
        #
        #
        # interaction_to_weight = {interactions[0]: [10, -10], interactions[1]: [-15, 15], interactions[2]: [10, -10],
        #                          interactions[3]: [15, 20]}
        #

        types = [Friendship, Mentorship, Theft, Help]

        earnings = {agent.generation_id: random.randrange(10, 75) for agent in agents}

        print("Yeet", earnings)

        for i in range(int(0.2*len(agents) * len(agents))):
            a = agents[random.randrange(len(agents))]
            b = agents[random.randrange(len(agents))]
            typ = types[random.randrange(len(types))]
            if typ == Friendship:
                i = Friendship(a, b, None)
                interactions.append(i)
                e1, e2 = (random.randrange(10, 100), random.randrange(10, 100))
                earnings[a.generation_id] += e1
                earnings[b.generation_id] += e2
                interaction_to_weight[i] = [e1, e2]
            elif typ == Mentorship:
                i = Mentorship(a, b, None)
                interactions.append(i)
                n = random.randrange(10, 100)
                interaction_to_weight[i] = [n, -n]
                earnings[a.generation_id] += n
                earnings[b.generation_id] += -n
            elif typ == Theft:
                i = Theft(a, b, None)
                interactions.append(i)
                n = random.randrange(10, 100)
                interaction_to_weight[i] = [n, -n]
                earnings[a.generation_id] += n
                earnings[b.generation_id] += -n
            elif typ == Help:
                i = Help(a, b, None)
                interactions.append(i)
                n = random.randrange(10, 100)
                interaction_to_weight[i] = [-n, n]
                earnings[a.generation_id] += n
                earnings[b.generation_id] += -n

        earnings = {'AG0': 40, 'AG1':15, 'AG2': 5, 'AG3': 25}
        power = SocialAnalysis.power_distribution(interactions, interaction_to_weight,1,earnings)

        std = statistics.stdev(list(power.values()))
        mean = statistics.mean(list(power.values()))
        print(power)
        print("Std", std)
        powers = {a: (power[a] - mean)/std for a in power}

        a = max(powers, key=lambda key: powers[key])
        print("Max", a, powers[a])

    @staticmethod
    def type_earnings(interactions_to_earnings, exclude_self_interactions=True):
        total_theft = 0
        total_friend = 0
        total_mentorship = 0
        total_help = 0
        for interaction in interactions_to_earnings:
            if exclude_self_interactions and interaction.proactive_agent != interaction.reactive_agent:
                p, r = interactions_to_earnings[interaction]
                if isinstance(interaction, Friendship):
                    total_friend += p + r
                if isinstance(interaction, Theft):
                    total_theft += max(p, r)
                if isinstance(interaction, Help):
                    total_help += max(p, r)
                if isinstance(interaction, Mentorship):
                    total_mentorship += max(p, r)
        return total_friend, total_mentorship, total_help, total_theft

    @staticmethod
    def group_metrics(interactions_to_earnings):
        tf, tm, th, tt = SocialAnalysis.type_earnings(interactions_to_earnings)
        total = tf + tm + th + tt
        asocial = 0 if total == 0 else tt / total
        productivity = 0 if total == 0 else tm / total
        coop = 0 if total == 0 else (tf + th) / total
        return asocial, productivity, coop

    @staticmethod
    def to_hierarchy(round_to_agent_to_wealth):
        round_to_hierarchy = {}
        for round in round_to_agent_to_wealth:
            wealths = round_to_agent_to_wealth[round]
            keys = sorted(wealths, key=lambda key: wealths[key])
            round_to_hierarchy[round] = keys
        return round_to_hierarchy

    @staticmethod
    def hierarchy_stablity(round_to_agent_to_value, agents=None, thresh=0.8):
        rounds = sorted(list(round_to_agent_to_value.keys()))
        similarities = []

        round_to_hierarchy = {}
        for i in range(len(rounds) - 1):
            p, n = rounds[i], rounds[i + 1]

            if p in round_to_hierarchy:
                hp = round_to_hierarchy[p]
            else:
                hp = round_to_agent_to_value[p]
                hp = sorted(hp.keys(), key=lambda item: hp[item])
                hp = [x for x in hp if x in agents] if agents is not None else hp
                round_to_hierarchy[p] = hp

            if n in round_to_hierarchy:
                hn = round_to_hierarchy[n]
            else:
                hn = round_to_agent_to_value[n]
                hn = sorted(hn.keys(), key=lambda item: hn[item])
                hn = [x for x in hn if x in agents] if agents is not None else hn
                round_to_hierarchy[n] = hn

            s = SocialAnalysis.sim(hp, hn, False)
            if len(hp) > 0 and len(hn) > 0:
                similarities.append((p, n, s))

        # Find consecutive points where sim > 0.8
        consecutive = []
        temp = []
        for i in range(len(similarities)):
            p, n, s = similarities[i]
            if s > thresh:
                temp.append(similarities[i])

            if (not (s > thresh) or i == (len(similarities) - 1)) and len(temp) > 0:
                avg = sum([c[2] for c in temp]) / len(temp)
                start = temp[0][0]
                end = temp[len(temp) - 1][1]

                count = {}
                for r in range(start, end+1):
                    atv = round_to_agent_to_value[r]
                    count = {a: atv[a] + count[a] if a in count else 0 for a in atv}
                count = {a: count[a]/(end-start) for a in count}
                consecutive.append((start, end, avg, count))
                temp = []
        return consecutive

    @staticmethod
    def limit_interactions_to_agents(interactions, agents):
        i = []
        for interaction in interactions:
            p = interaction.proactive_agent.generation_id
            r = interaction.reactive_agent.generation_id
            if p in agents and r in agents:
                i.append(interaction)
        return i

    @staticmethod
    def non_subset_groups(groups):
        non_subsets = []
        for g in groups:
            is_subset = False
            for g2 in groups:
                if g != g2:
                    if set(g).issubset(g2):
                        is_subset = True
            if not is_subset:
                non_subsets.append(g)
        return non_subsets

    @staticmethod
    def social_groups_as_interactions(interactions_to_earnings, non_subsets, all_agents, agents_to_mine=None, agents_to_earned=None):
        interactions = list(interactions_to_earnings.keys())

        ite = {}

        # Agents not in any groups become seperate groups
        groups = [GroupAgent(g) for g in non_subsets]
        agents_in_group = []
        for group in groups:
            agents_in_group = set(agents_in_group).union(group.agents)

        for agent in all_agents:
            if agent not in agents_in_group:
                groups.append(GroupAgent([agent]))

        itupe = {}

        for interaction in interactions:
            p = interaction.proactive_agent.generation_id
            r = interaction.reactive_agent.generation_id

            gp = [g for g in groups if g.memberOf(p)]
            gr = [g for g in groups if g.memberOf(r)]

            for g in gp:
                for g2 in gr:
                    earnings = interactions_to_earnings[interaction] if interaction in interactions_to_earnings else 0
                    k = (type(interaction), g, g2)
                    res = itupe[k] if k in itupe else [0,0]
                    res[0] += earnings[0]
                    res[1] += earnings[1]
                    itupe[k] = res

        ate = {}
        atm = {}

        for itup in itupe:
            t, g, g2 = itup
            earning = itupe[itup]
            i = None
            if t == Friendship:
                i = Friendship(g, g2, None)
            elif t == Mentorship:
                i = Mentorship(g, g2, None)
            elif t == Theft:
                i = Theft(g, g2, None)
            elif t == Help:
                i = Help(g, g2, None)

            if i is not None:
                ite[i] = earning

        if agents_to_earned is None and agents_to_mine is None:
            return ite

        for g in groups:
            total_mine = 0
            total_earn = 0
            for agent in g.agents:
                if agents_to_mine is not None:
                    total_mine += (agents_to_mine[agent] if agent in agents_to_mine else 0)
                if agents_to_earned is not None:
                    total_earn += (agents_to_earned[agent] if agent in agents_to_earned else 0)

            if agents_to_mine is not None:
                atm[g.generation_id] = total_mine
            if agents_to_earned is not None:
                ate[g.generation_id] = total_earn

        return ite, atm, ate, groups

    @staticmethod
    def sim(a, b, should_sort=True):
        l = list(set().union(a, b))
        result = {b: a for a, b in enumerate(l)}
        first = [result[f] for f in a]
        second = [result[f] for f in b]
        if should_sort:
            first.sort()
            second.sort()
        sm = difflib.SequenceMatcher(None, first, second)
        return sm.ratio()

    @staticmethod
    def introduced_wealth(interactions_to_earnings, agent_to_mine, agents):
        agent_to_earn = {a: agent_to_mine[a] if a in agent_to_mine else 0 for a in agents}
        for interaction in interactions_to_earnings:
            p = interaction.proactive_agent.generation_id
            r = interaction.reactive_agent.generation_id
            if p in agents and r in agents:
                pe, re = interactions_to_earnings[interaction]
                agent_to_earn[p] += pe
                agent_to_earn[r] += re

        money_added = sum([agent_to_earn[a] for a in agents])
        money_taken = sum([agent_to_mine[a] for a in agents])
        return money_added - money_taken

    @staticmethod
    def network2():
        g = nx.MultiGraph()

        l = [(1, 3), (1,2), (3,1), (2,1), (4,1), (1, 2), (2, 4), (2, 4), (2, 4), (1,3), (1,3)]
        l += [(2, 5), (5, 6), (6, 7), (6, 7), (6, 9), (6, 9), (7, 9), (7, 9), (7, 8), (7, 8), (8, 9), (8, 9)]
        l += [(9, 10), (9, 10), (6, 11), (11, 12), (11, 12), (11, 13), (11, 14), (14, 15), (14, 15), (14, 15), (15, 13), (12, 13)]

        h = {}
        for e in l:
            a, b = e
            g.add_edge(a, b, r="A")
            h[e] = "A"

        pos = nx.spring_layout(g, scale=2)
        nx.draw(g, pos, with_labels=True)

        edge_labels = nx.get_edge_attributes(g, 'r')

        nx.draw_networkx_edge_labels(g, pos, edge_labels=h)

        print(edge_labels)



        c = list(k_clique_communities(g, 3))
        d = list(greedy_modularity_communities(g))
        e = list(asyn_lpa_communities(g))




        print("Cliques", c)
        print("Cliques2", d)
        print("Cliques3", e)



        plt.show()





        #print( g.edges )