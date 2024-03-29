from src.Experiment import *
import unittest


class TestingDetectionSoftware(unittest.TestCase):

    @staticmethod
    def generate_round(interactions, atw, aids, mining_range, f_range, m_range, h_range, t_range):
        ite = {}
        s, f = mining_range
        atm = {i: random.randrange(s, f) for i in aids}
        atw = {i: atw[i] + atm[i] for i in aids}
        for interaction in interactions:
            pe, re = 0, 0
            if type(interaction) == Friendship:
                a = random.randrange(f_range[0], f_range[1])
                pe, re = a, a
            elif type(interaction) == Mentorship:
                a = random.randrange(m_range[0], m_range[1])
                pe, re = a, -a
            elif type(interaction) == Help:
                a = random.randrange(h_range[0], h_range[1])
                pe, re = -a, a
            else:
                a = random.randrange(t_range[0], t_range[1])
                pe, re = a, -a
            p, r = interaction.proactive_agent.generation_id, interaction.reactive_agent.generation_id
            atw[p] += pe
            atw[r] += re
            ite[interaction] = [pe, re]

        return ite, atm, atw

    @staticmethod
    def generate_pre_built_test_environment():
        a = []
        for i in range(1, 15):
            n = str(i)
            a.append(Agent.random(n, n))

        gid = {agent.generation_id: agent for agent in a}
        round_to_interactions = {}
        round_to_agent_to_mine = {}
        round_to_interactions_to_earnings = {}
        round_to_agent_to_wealth = {}
        atw = {a: 0 for a in gid}

        interactions = [Mentorship(a[0], a[1], None), Friendship(a[0], a[1], None),
                        Friendship(a[0], a[2], None), Help(a[0], a[2], None),
                        Mentorship(a[2], a[3], None), Friendship(a[2], a[3], None),
                        Friendship(a[1], a[3], None), Mentorship(a[1], a[3], None),
                        Friendship(a[2], a[1], None),

                        Friendship(a[1], a[4], None), Friendship(a[4], a[10], None),

                        Help(a[7], a[8], None), Help(a[5], a[7], None), Mentorship(a[6], a[5], None),
                        Mentorship(a[6], a[7], None), Friendship(a[6], a[8], None), Friendship(a[6], a[7], None),
                        Friendship(a[5], a[8], None),

                        Friendship(a[10], a[13], None), Friendship(a[10], a[12], None), Friendship(a[12], a[11], None),
                        Friendship(a[11], a[13], None), Help(a[10], a[11], None), Mentorship(a[13], a[12], None)]

        ite, atm, atw = TestingDetectionSoftware.generate_round(interactions, atw, list(gid.keys()), (100, 150),
                                                                (10, 20), (10, 20), (10, 20), (10, 20))
        round_to_interactions_to_earnings[1] = ite
        round_to_agent_to_mine[1] = atm
        round_to_agent_to_wealth[1] = atw.copy()


        round_to_interactions[1] = interactions

        interactions = [Mentorship(a[0], a[1], None), Friendship(a[0], a[1], None),
                        Friendship(a[0], a[2], None), Help(a[1], a[3], None),
                        Mentorship(a[0], a[2], None), Mentorship(a[0], a[3], None),

                        Theft(a[0], a[4], None), Theft(a[4], a[2], None), Friendship(a[4], a[6], None),

                        Help(a[8], a[6], None), Theft(a[8], a[7], None), Theft(a[7], a[8], None),
                        Friendship(a[6], a[8], None), Friendship(a[8], a[5], None), Mentorship(a[5], a[6], None),
                        Theft(a[5], a[10], None),

                        Theft(a[10], a[3], None), Friendship(a[10], a[13], None), Friendship(a[10], a[12], None),
                        Friendship(a[13], a[12], None), Help(a[12], a[10], None), Mentorship(a[13], a[12], None)]

        ite, atm, atw = TestingDetectionSoftware.generate_round(interactions, atw, list(gid.keys()), (100, 200),
                                                                (15, 25), (15, 25), (15, 25), (15, 25))
        round_to_interactions_to_earnings[2] = ite
        round_to_agent_to_mine[2] = atm
        round_to_agent_to_wealth[2] = atw.copy()
        round_to_interactions[2] = interactions

        interactions = [Mentorship(a[0], a[1], None), Friendship(a[0], a[1], None), Friendship(a[0], a[2], None),
                        Friendship(a[1], a[3], None), Mentorship(a[2], a[3], None), Friendship(a[2], a[3], None),
                        Mentorship(a[0], a[3], None), Theft(a[3], a[0], None), Theft(a[3], a[1], None),
                        Theft(a[3], a[4], None),

                        Mentorship(a[4], a[6], None), Mentorship(a[8], a[7], None), Mentorship(a[5], a[6], None),
                        Friendship(a[8], a[5], None), Friendship(a[5], a[7], None), Friendship(a[6], a[7], None),
                        Theft(a[5], a[10], None),

                        Theft(a[12], a[8], None), Theft(a[10], a[4], None), Friendship(a[10], a[11], None),
                        Friendship(a[10], a[12], None), Friendship(a[10], a[13], None), Friendship(a[13], a[12], None),
                        Mentorship(a[12], a[11], None), Help(a[13], a[11], None)]

        ite, atm, atw = TestingDetectionSoftware.generate_round(interactions, atw, list(gid.keys()), (100, 230),
                                                                (20, 40), (20, 40), (20, 40), (20, 40))
        round_to_interactions_to_earnings[3] = ite
        round_to_agent_to_mine[3] = atm
        round_to_agent_to_wealth[3] = atw.copy()
        round_to_interactions[3] = interactions


        interactions = [Friendship(a[3], a[2], None), Friendship(a[0], a[2], None),
                        Friendship(a[0], a[1], None), Theft(a[0], a[3], None), Theft(a[1], a[3], None),
                        Theft(a[4], a[3], None),

                        Mentorship(a[5], a[8], None), Mentorship(a[5], a[6], None), Friendship(a[4], a[6], None),
                        Friendship(a[4], a[5], None), Friendship(a[4], a[8], None), Friendship(a[8], a[5], None),
                        Friendship(a[6], a[5], None),
                        Theft(a[5], a[10], None),

                        Friendship(a[10], a[12], None), Mentorship(a[10], a[12], None), Mentorship(a[10], a[13], None),
                        Friendship(a[10], a[13], None), Friendship(a[10], a[11], None), Friendship(a[12], a[11], None),
                        Help(a[11], a[12], None), Help(a[13], a[11], None)]

        ite, atm, atw = TestingDetectionSoftware.generate_round(interactions, atw, list(gid.keys()), (100, 250),
                                                                (20, 50), (25, 60), (5, 15), (5, 15))
        round_to_interactions_to_earnings[4] = ite
        round_to_agent_to_mine[4] = atm
        round_to_agent_to_wealth[4] = atw.copy()
        round_to_interactions[4] = interactions

        interactions = [Friendship(a[0], a[1], None), Friendship(a[0], a[2], None), Friendship(a[1], a[2], None),
                        Theft(a[0], a[3], None), Theft(a[2], a[3], None),
                        Theft(a[3], a[0], None),

                        Mentorship(a[4], a[6], None), Friendship(a[4], a[8], None),
                        Friendship(a[4], a[5], None), Friendship(a[5], a[8], None),
                        Help(a[6], a[5], None), Friendship(a[4], a[6], None),

                        Friendship(a[10], a[12], None), Friendship(a[10], a[11], None), Friendship(a[12], a[11], None)]

        ite, atm, atw = TestingDetectionSoftware.generate_round(interactions, atw, list(gid.keys()), (100, 270),
                                                                (20, 55), (20, 55), (20, 55), (20, 55))
        round_to_interactions_to_earnings[5] = ite
        round_to_agent_to_mine[5] = atm
        round_to_agent_to_wealth[5] = atw.copy()
        round_to_interactions[5] = interactions

        return a, gid, round_to_interactions, round_to_interactions_to_earnings, round_to_agent_to_mine, round_to_agent_to_wealth

    @staticmethod
    def random_interaction(p, r, types=[Friendship, Mentorship, Help, Theft] ):
        t = types[random.randrange(len(types))]
        if isinstance(t, Friendship):
            interaction = Friendship(p, r, None)
        elif isinstance(t, Mentorship):
            interaction = Mentorship(p, r, None)
        elif isinstance(t, Theft):
            interaction = Theft(p, r, None)
        else:
            interaction = Help(p, r, None)
        return interaction

    @staticmethod
    def generate_test_environment(groups, number_of_agents, number_of_rounds):
        a = []
        for i in range(1, number_of_agents + 1):
            n = str(i)
            a.append(Agent.random(n, n))
        gid = {agent.generation_id:agent for agent in a}

        nwg = 20
        nbg = 2
        # Set up groups
        round_to_interactions = {}
        types = [Friendship, Mentorship, Help, Theft]
        for round in range(1, number_of_rounds + 1):
            # Get groups that are involved in this round
            rg = [g for g, s, e in groups if s <= round <= e]

            # Create interactions for these groups

            interactions = []
            for g in rg:
                for i in range(nwg):
                    lst = g.copy()
                    p = lst[random.randrange(len(lst))]
                    lst.remove(p)
                    r = lst[random.randrange(len(lst))]
                    interaction = TestingDetectionSoftware.random_interaction(gid[p], gid[r])
                    interactions.append(interaction)

            # Add interactions across groups
            for u in range(len(rg)):
                for v in range(u+1, len(rg)):
                    for i in range(nbg):
                        g, g2 = rg[u], rg[v]
                        p = g[random.randrange(len(g))]
                        n = g2.copy()
                        if p in n:
                            n.remove(p)
                        r = n[random.randrange(len(n))]
                        interaction = TestingDetectionSoftware.random_interaction(gid[p], gid[r])
                        interactions.append(interaction)
            round_to_interactions[round] = interactions
        return a, gid, round_to_interactions

    def testSocialStructures(self):
        a, gid, round_to_interactions, rte, ram, raw = self.generate_pre_built_test_environment()
        g = sorted(list(gid.keys()), key=lambda item: int(item))
        social_groups = [[g[0], g[1], g[2], g[3]], [g[5], g[6], g[7], g[8]], [g[10], g[11], g[12]]]
        found = SocialAnalysis.find_social_structures(round_to_interactions, stability_thresh=2)
        print(found)
        for group in social_groups:
            # Need to check if this group has been identified
            f = False
            for g in found:
                lst, s, e = g

                if set(group).issubset(lst):
                    f = True
            self.assertTrue(f, msg=str(group) + " wasn't found")

        # for round in sorted(round_to_interactions.keys()):
        #     interactions = round_to_interactions[round]
        #     net = DataAnalysis.get_network(interactions)
        #     DataAnalysis.draw_network(net)

    def testB(self):
        groups_test = [(['2', '1', '3', '5', '4'], 1, 5),
                        (['13', '12', '14', '11'], 1, 2),
                        (['9', '7', '8', '6'], 1, 3),
                        (['13', '12', '14', '5', '11'], 3, 5),
                        (['9', '7', '5', '6'], 4, 5)]

        a, gid, round_to_interactions = self.generate_test_environment(groups_test, 14, 5)
        found = SocialAnalysis.find_social_structures(round_to_interactions, stability_thresh=3)
        social_groups = [g for g,s,e in groups_test]

        count = 0
        for group in social_groups:
            # Need to check if this group has been identified
            f = False
            for g in found:
                lst, s, e = g

                if set(group).issubset(lst):
                    f = True
            if f == True:
                count += 1
        print("Found", count," of ", len(social_groups))
        self.assertGreaterEqual(count/len(social_groups), 0.8)

    def testConsistentSocialGroups(self):
        structures_data_sets = [[(('A', 'L', 'C', 'E'), 1,2), (('C', 'D', 'E'), 2,4), (('C', 'E', 'D'), 3,6)],
                                [(('A', 'B', 'C'), 1,2), (('C', 'D', 'E','F'), 2,4), (('A', 'L', 'E'), 2,4)],
                                [(('A', 'L', 'C', 'E'), 1,2), (('A', 'B', 'C'), 1,2), (('C', 'D', 'E','F'), 2,4), (('A', 'L'), 2,4)]]

        a, b, c = SocialAnalysisResult.social_structures_consistency(structures_data_sets)
        print(Experiment.social_group_map_csv_entry(a))

    @staticmethod
    def generate_power_test_environment(agent_to_power, number_of_interactions, total_wealth=10000):
        total = sum([agent_to_power[agent] for agent in agent_to_power])
        a_to_wealth = {agent: int(total_wealth * agent_to_power[agent]/total) for agent in agent_to_power}
        a_to_ni = {agent: int(number_of_interactions * agent_to_power[agent]/total) for agent in agent_to_power}
        gid = {agent: Agent.random(agent,agent) for agent in agent_to_power}
        i = 0
        interactions = []

        number_of_interactions = sum([a_to_ni[agent] for agent in agent_to_power])

        while i < number_of_interactions:
            interactable_agents = [agent for agent in agent_to_power if a_to_ni[agent] > 0]
            if len(interactable_agents) < 2:
                break

            tot = sum([a_to_ni[a] for a in interactable_agents])
            weights = [a_to_ni[a]/tot for a in interactable_agents]
            choices = random.choices(population=interactable_agents, weights=weights, k=2)
            p, r = choices[0], choices[1]

            a_to_ni[p] -= 1
            a_to_ni[r] -= 1

            interaction = TestingDetectionSoftware.random_interaction(gid[p], gid[r])
            interactions.append(interaction)

        return interactions, a_to_wealth

    @staticmethod
    def get_power_metric(agent_to_power, total_number_of_interactions=300):
        interactions, a_to_wealth = TestingDetectionSoftware.generate_power_test_environment(agent_to_power,
                                                                                             total_number_of_interactions)
        power = SocialAnalysis.power_distribution(interactions, agent_earnings=a_to_wealth, should_draw=True)
        metric = PowerMetrics(power)
        return metric

    @staticmethod
    def generate_power_stability_environment(p1, p2, p3, p4, p5, x1 = 4, x2 = 4, x3 = 4, x4 = 4, x5 = 5):
        x = x1
        round_to_agent_to_power = {}
        for i in range(1, x):
            round_to_agent_to_power[i] = p1
        y = x + x2
        for i in range(x, y):
            round_to_agent_to_power[i] = p2
        z = y + x3
        for i in range(y, z):
            round_to_agent_to_power[i] = p3
        e = z + x4
        for i in range(z, e):
            round_to_agent_to_power[i] = p4
        f = e + x5
        for i in range(e, f):
            round_to_agent_to_power[i] = p5

        return round_to_agent_to_power

    def test_power_stability(self):
        return None
        agents = [str(i) for i in range(1,11)]
        dic = {'1': 5, '2': 30, '3': 55, '4': 80, '5': 100, '6': 150, '7': 200, '8': 250, '9': 320, '10': 5000}
        dem = {'1': 100, '2': 100, '3': 100, '4': 100, '5': 100, '6': 100, '7': 100, '8': 100, '9': 100, '10': 100}
        rc = {'1': 10, '2': 10, '3': 10, '4': 10, '5': 10, '6': 10, '7': 10, '8': 100, '9': 100, '10': 100}
        sc = {'1': 5, '2': 5, '3': 100, '4': 100, '5': 100, '6': 100, '7': 100, '8': 100, '9': 100, '10': 100}
        sl = {'1': 5, '2': 100, '3': 100, '4': 100, '5': 100, '6': 100, '7': 100, '8': 100, '9': 100, '10': 100}
        a = ['A','B','C', 'D']
        b = ['A', 'C', 'B', 'D']

        self.assertNotEqual(SocialAnalysis.sim(a,b, True), SocialAnalysis.sim(a,b, False))

        x = PowerStability(
            TestingFindingSocialStructures.generate_power_stability_environment(
                dic, sl, rc, sc, sl), agents, 0.9)

        y = PowerStability(
            TestingFindingSocialStructures.generate_power_stability_environment(
                dic, dem, rc, sc, sl), agents, 0.9)

        z = PowerStability(
            TestingFindingSocialStructures.generate_power_stability_environment(
                rc, dem, dem, sc, sl), agents, 0.9)

        print(PowerStability.merge([x.stability_info,y.stability_info, z.stability_info]))

    def testPowerDistributions(self):
        return None
        # Testing dictatorship
        p = { '1': 5, '2': 30, '3': 55, '4': 80, '5': 100, '6': 150, '7': 200, '8': 250, '9': 320, '10': 5000}
        metric1 = self.get_power_metric(p)
        self.assertEqual(metric1.type_info[0], 'Dictatorship')
        # print(metric1)

        # Testing Democracy
        p = {'1': 100, '2': 100, '3': 100, '4': 100, '5': 100, '6': 100, '7': 100, '8': 100, '9': 100, '10': 100}
        metric2 = self.get_power_metric(p)
        self.assertEqual(metric2.type_info[0], 'Democracy')
        # print(metric2)
        # Tesing Ruiling Class
        p = {'1': 10, '2': 10, '3': 10, '4': 10, '5': 10, '6': 10, '7': 10, '8': 100, '9': 100, '10': 100}
        metric3 = self.get_power_metric(p)
        print(metric3)

        self.assertEqual(metric3.type_info[0], 'RulingClass')

        # Testing ServantClass
        p = {'1': 5, '2': 5, '3': 100, '4': 100, '5': 100, '6': 100, '7': 100, '8': 100, '9': 100, '10': 100}
        metric4 = self.get_power_metric(p)
        self.assertEqual(metric4.type_info[0], 'ServantClass')
        print(metric4)

        # Testing ServantClass
        p = {'1': 10, '2': 100, '3': 100, '4': 100, '5': 100, '6': 100, '7': 100, '8': 100, '9': 100, '10': 100}
        metric6 = self.get_power_metric(p)
        print(metric6.type_info)
        self.assertEqual(metric6.type_info[0], 'Slavery')



        p = {'1': 200, '2': 203, '3': 199, '4': 202, '5': 200, '6': 197, '7': 205, '8': 203, '9': 198, '10': 202}
        metric5 = self.get_power_metric(p)

        # print(metric2.type_info)
        # print(metric4.type_info)
        # print(metric1.type_info)
        # print(PowerMetrics.merge([metric2.type_info, metric4.type_info, metric1.type_info]))

    @staticmethod
    def generate_hierarchy(agents, number_of_rounds, stability):

        n = int((0.5 + 0.5*stability) * len(agents))

        round_to_agent_value = {}
        for round in range(1,number_of_rounds+1):
            hierarchy = agents.copy()
            l = random.randrange(len(hierarchy))

            fixed = [hierarchy[i % len(hierarchy)] for i in range(l,l+n)]
            non_fixed = [a for a in hierarchy if a not in fixed]
            random.shuffle(non_fixed)
            lst = []

            c = 0
            for i in range(len(hierarchy)):
                if hierarchy[i] not in fixed:
                    lst.append(non_fixed[c])
                    c += 1
                else:
                    lst.append(hierarchy[i])

            agent_to_value = {lst[i]: i for i in range(len(lst))}
            round_to_agent_value[round] = agent_to_value

        return round_to_agent_value

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

    def testHierarchyStability(self):
        return None
        agents = [str(i) for i in range(1,10)]
        rtv = self.generate_hierarchy(agents, 10, 0.5)

        x = SocialAnalysis.hierarchy_stablity(rtv,agents)

        rounds = rtv.keys()
        start, end = min(rounds), max(rounds)

        print(self.get_hierarchy_stability_info(start, end, x))

    @staticmethod
    def random_agents(number_of_agents):
        a = []
        for i in range(1, number_of_agents + 1):
            n = str(i)
            a.append(Agent.random(n, n))
        gid = {agent.generation_id: agent for agent in a}
        return gid, a

    def testingIntroducedWealth(self):
       return None
       gid, a = self.random_agents(3)
       ite = {}
       ite[Friendship(a[0], a[1], None)] = [10, 10]
       ite[Friendship(a[2], a[1], None)] = [20, 20]
       ite[Friendship(a[2], a[0], None)] = [10, 10]
       ite[Mentorship(a[2], a[1], None)] = [10, -10]
       ite[Mentorship(a[0], a[1], None)] = [10, -10]
       atm = {}
       atm[a[0].generation_id] = 100
       atm[a[1].generation_id] = 250
       atm[a[2].generation_id] = 150

       iw = SocialAnalysis.introduced_wealth(ite, atm, ['1', '2', '3'])
       self.assertEqual(iw,80)

       iw = SocialAnalysis.introduced_wealth(ite, atm, ['1', '2'])
       self.assertEqual(iw, 20)

       iw = SocialAnalysis.introduced_wealth(ite, atm, ['2', '3'])
       self.assertEqual(iw, 40)

       iw = SocialAnalysis.introduced_wealth(ite, atm, ['1', '3'])
       self.assertEqual(iw, 20)

    def testingGroupMetrics(self):
        return None
        gid, a = self.random_agents(3)
        ite = {}
        ite[Friendship(a[0], a[1], None)] = [10, 10]
        ite[Friendship(a[2], a[1], None)] = [20, 20]
        ite[Friendship(a[2], a[0], None)] = [10, 10]
        ite[Mentorship(a[2], a[1], None)] = [10, -10]
        ite[Mentorship(a[0], a[1], None)] = [10, -10]
        ite[Theft(a[0], a[2], None)] = [5, -5]
        ite[Help(a[2], a[1], None)] = [-9, 9]
        tf, tm, th, tt = 80, 20, 9, 5
        total = 114
        asocial, productivity, coop = SocialAnalysis.group_metrics(ite)

        self.assertEqual(asocial, tt/total)
        self.assertEqual(productivity, tm / total)
        self.assertEqual(coop, (tf + th) / total)

    @staticmethod
    def generate_group_ite():
        gid, a = TestingDetectionSoftware.random_agents(9)

        atm = {a:random.randrange(40,120) for a in gid}
        ite = {}
        ite[Friendship(a[0], a[1], None)] = [10, 10]
        ite[Friendship(a[2], a[1], None)] = [20, 20]
        ite[Friendship(a[2], a[0], None)] = [10, 10]
        ite[Mentorship(a[2], a[1], None)] = [10, -10]
        ite[Mentorship(a[0], a[1], None)] = [10, -10]
        ite[Theft(a[0], a[2], None)] = [5, -5]
        ite[Help(a[2], a[1], None)] = [-9, 9]

        ite[Friendship(a[2], a[5], None)] = [10, 10]
        ite[Theft(a[4], a[2], None)] = [5, -5]

        ite[Friendship(a[3], a[4], None)] = [10, 10]
        ite[Help(a[5], a[4], None)] = [20, 20]
        ite[Friendship(a[5], a[3], None)] = [10, 10]
        ite[Mentorship(a[5], a[3], None)] = [10, -10]
        ite[Theft(a[3], a[4], None)] = [10, -10]
        ite[Theft(a[3], a[5], None)] = [5, -5]
        ite[Friendship(a[5], a[4], None)] = [-9, 9]

        ite[Friendship(a[6], a[4], None)] = [10, 10]
        ite[Theft(a[5], a[8], None)] = [5, -5]

        ite[Friendship(a[6], a[7], None)] = [10, 10]
        ite[Help(a[8], a[7], None)] = [20, 20]
        ite[Help(a[8], a[6], None)] = [10, 10]
        ite[Mentorship(a[8], a[6], None)] = [10, -10]
        ite[Friendship(a[8], a[7], None)] = [10, -10]
        ite[Help(a[6], a[8], None)] = [5, -5]
        ite[Friendship(a[8], a[7], None)] = [-9, 9]

        ite[Friendship(a[7], a[1], None)] = [10, 10]
        ite[Theft(a[2], a[6], None)] = [5, -5]

        return ite, [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']], ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    def testingSocialGroupsAsInteractions(self):
        return None
        a, gid, round_to_interactions, rte, ram, raw = self.generate_pre_built_test_environment()
        g = sorted(list(gid.keys()), key=lambda item: int(item))
        social_groups = [[g[0], g[1], g[2], g[3]], [g[5], g[6], g[7], g[8]], [g[10], g[11], g[12]]]

        print(social_groups)

        round = 1
        interactions = round_to_interactions[round]
        ite = rte[round]

        SocialAnalysis.draw_network(SocialAnalysis.get_network(list(ite.keys())))

        itge = SocialAnalysis.social_groups_as_interactions(ite, social_groups, g)

        SocialAnalysis.draw_network(SocialAnalysis.get_network(list(itge.keys())))



        res = SocialAnalysis.limit_interactions_to_agents(list(ite.keys()), ['1', '2', '3'])

    def test_integration(self):
        a, gid, round_to_interactions, rte, ram, raw = self.generate_pre_built_test_environment()
        result = SocialAnalysisResult(rte, raw, ram, list(gid.keys()))

        print("Result", result)

    def testing_combinations(self):
        return None
        n= 32
        combinations = Experiment.spread_out_combinations(n)
        for c in combinations:
            print(c)

        print(Experiment.count(combinations))
        self.assertEqual(len(combinations), n)
        for i in range(len(combinations)):
            self.assertEqual(12, len(combinations[i]))

    def testing_consistent_social_structures(self):
        return None
        groupA = [ (['A', 'D', 'B'], 1, 2), (['F', 'B', 'C', 'E'], 1, 2),  (['A', 'M', 'C'], 1, 2),
                   (['N', 'O', 'P'], 1, 2)]
        groupB = [ (['A', 'F', 'B', 'H'], 1, 2), (['F', 'B', 'L', 'E'], 1, 2),  (['A', 'K', 'C'], 1, 2),
                   (['N', 'R', 'P', 'E'], 1, 2)]
        groupC = [(['A', 'B', 'D', 'E'], 1, 2), (['A', 'F', 'C'], 1, 2), (['A', 'C', 'K', 'E'], 1, 2)]

        print(SocialAnalysisResult.social_structures_consistency([groupA, groupC, groupA]))

    def get_combos(self):

        all_combos = []
        # RI*MH TW*ML I*HL CF*HH (X3) [Missing: (I*HH, TW*ML)
        all_combos.append([('RI', 'M', 'H'), ('TW', 'M', 'L'), ('I', 'H', 'L'), ('CF', 'H', 'H')] * 3)

        # CF*HH TW*ML I*HH I*HL (X3) [Missing: (RI*MH, TW*LL)
        all_combos.append([('CF', 'H', 'H'), ('TW', 'M', 'L'), ('I', 'H', 'H'), ('I', 'H', 'L')] * 3)

        # RI*MH TW*ML I*HL I*HH TW*LL (X2) (Found) [Missing: CF*HH, I*HL)
        all_combos.append([('RI', 'M', 'H'), ('TW', 'M', 'L'), ('I', 'H', 'H'), ('TW', 'L', 'L')] * 2)

        # RI*MH TW*ML I*HL CF*HH I*HH TW*LL (X2) (Found) [Missing:)
        all_combos.append([('RI', 'M', 'H'), ('TW', 'M', 'L'), ('I', 'H', 'L'), ('CF', 'H', 'H'), ('I', 'H', 'H'),
                           ('TW', 'L', 'L')] * 2)

        return all_combos;

    def testing_integration_with_simulation(self):
        # Testing you can acquire static agents from runs

        combinations = self.get_combos()

        complete = []
        for i in range(len(combinations)):
            current_position = i
            run = combinations[current_position]
            agentIDs = [(belbin + "*" + mine + appr) for (belbin, mine, appr) in run]
            print("AIDS", agentIDs)
            training_directory = "/Users/faithnyota1/Computer Science/3rd Year/Individual Project/Analysis/training"

            # static_agents = Experiment.get_training_agents(training_directory, [], agentIDs)[0]


            #print("Static", [(a, type(a)) for a in static_agents])

            # learning_agents = Experiment.get_training_agents(training_directory, agentIDs, [])[0]
            # print("Learning", [(a, type(a)) for a in learning_agents])

            # agents = [agent.generation_id for agent in agents]
            # Running simulation with learning agents test
            results = []

            for i in range(3):
                agents = Experiment.get_training_agents(training_directory, agentIDs, [])[0]

                print([(a.name, a.generation_id, a.id) for a in agents])

                gui_requests, hierarchy, training, social_analysis = RunningSimulation.simulate(agents,
                                                                                                should_display=False,
                                                                                                should_upload=False,
                                                                                                should_social_analyse=True)
                data = social_analysis.get_data()
                results.append(data)
                print(i, "-",  data)

            print("MERGED:", SocialAnalysisResult.merge(results))


        experiment_folder = "/Users/faithnyota1/Computer Science/3rd Year/Individual Project/Analysis/texp"
        with open(experiment_folder + "/test_results.json", 'w') as fp:
            json.dump(complete, fp)

        # testing experiment:

    @staticmethod
    def generate_full_environment(groups, number_of_agents, number_of_rounds):
        a, gid, round_to_interactions = TestingDetectionSoftware.generate_test_environment(groups, number_of_agents,
                                                                                           number_of_rounds)

    def test_experiment(self):
        f = "/Users/faithnyota1/Computer Science/3rd Year/Individual Project/Analysis/testexperiment"
        e = f + "/test_results.json"
        results = Experiment.get_json(e)

        r, agents, i = results[0]
        Experiment.process_main_experiment(results, f + "/results")































