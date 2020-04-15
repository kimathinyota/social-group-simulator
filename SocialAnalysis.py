import matplotlib.pyplot as plt
import networkx as nx
from src.Agent import *
from src.GoldMiningEnvironment import Friendship, Mentorship, Theft, Help
from networkx.algorithms.community import k_clique_communities, greedy_modularity_communities
from networkx.algorithms.community.label_propagation import *
from src.Helper import *
import difflib
import functools
from networkx.algorithms.centrality import *
from networkx.algorithms.community.centrality import girvan_newman


class SocialAnalysis:

    @staticmethod
    def get_network(interactions):
        g = nx.MultiGraph()
        for interaction in interactions:
            p = interaction.proactive_agent
            r = interaction.reactive_agent
            if isinstance(interaction, Friendship):
                g.add_edge(p.generation_id, r.generation_id, r="F")
                g.add_edge(r.generation_id, p.generation_id, r="F")
            elif isinstance(interaction, Mentorship):
                g.add_edge(p.generation_id, r.generation_id, r="M")
            elif isinstance(interaction, Theft):
                g.add_edge(p.generation_id, r.generation_id, r="T")
            elif isinstance(interaction, Help):
                g.add_edge(p.generation_id, r.generation_id, r="H")
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
        for i in range(len(rounds)-1):
            round = rounds[i]
            r1 = round_communities[round]
            r2 = round_communities[round+1]
            for c in r1:
                for c2 in r2:
                    # compares how similar communities c and c2 are
                    sim = SocialAnalysis.sim(c,c2)
                    rcsim.append(((round, c), (round + 1, c2), sim))


        # Only keep entries with a higher similarity than sim_thresh
        communities = [rcsim[i] for i in range(len(rcsim)) if rcsim[i][2] > sim_thresh]

        # Finding overlapping entries
        # Below will find the intersection between the communities of consecutive entries within rcsim
        # or in other words, finding and storing the agents involved within clusters of exchange relations
        # (or lack thereof) that persist across multiple rounds

        overlaps = []
        community = communities[0]
        rc, rc2, s = community
        r1, c1 = rc
        r2, c2 = rc2

        overlap = list(set(c1).intersection(c2))
        start_ends = []
        start = r1
        end = r2

        # find overlaping communities
        for i in range(1,len(communities)):
            community = communities[i]
            rc, rc2, s = community
            r1, c1 = rc

            if r1 == r2:
                overlap = list(set(overlap).intersection(c1))
            else:
                overlaps.append(overlap)
                start_ends.append((start, end))
                overlap = list(set(c1).intersection(rc2[1]))
                start = r1
            r2, c2 = rc2

            end = r2
            if i == len(communities) - 1:
                overlaps.append(overlap)
                start_ends.append((start, end))

        # stability_thresh = number of consecutive rounds that a community must persist for to be considered stable
        # will return all of the stable communities
        return [(overlaps[i], start_ends[i]) for i in range(len(overlaps)) if (start_ends[i][1] - start_ends[i][0])
                >= stability_thresh]

    @staticmethod
    def power_distribution(interactions):
        # For each round of interactions
        # Look at centrality of each agent involved in interaction

        network = SocialAnalysis.get_network(interactions)

        centrality = closeness_centrality(network)

        return centrality



    @staticmethod
    def sim(a, b):
        l = list(set().union(a, b))
        result = {b: a for a, b in enumerate(l)}
        first = [result[f] for f in a]
        second = [result[f] for f in b]
        first.sort()
        second.sort()
        sm = difflib.SequenceMatcher(None, first, second)
        return sm.ratio()

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