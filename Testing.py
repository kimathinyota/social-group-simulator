import unittest
import math


class TestingInteractionSuite(unittest.TestSuite):

    def setUp(self):
        types = [self.friend_type, self.mentor_type, self.help_type, self.theft_type]
        reactive, proactive = None, None
        for interaction in self.confirmed_interactions:
            if interaction.is_success:
                interaction_type = type(interaction)
                other_types = [x for x in types if x != interaction_type]

                if interaction.proactive_agent not in self.agent_to_interactions:
                    proactive = {interaction_type: ([interaction], [])}
                    for t in other_types:
                        proactive[t] = ([],[])
                    self.agent_to_interactions[interaction.proactive_agent] = proactive
                else:
                    proactive = self.agent_to_interactions[interaction.proactive_agent]
                    proactive[interaction_type][0].append(interaction)

                if interaction.reactive_agent not in self.agent_to_interactions:
                    reactive = {interaction_type:([],[interaction])}
                    for t in other_types:
                        reactive[t] = ([],[])
                    self.agent_to_interactions[interaction.reactive_agent] = reactive
                else:
                    reactive = self.agent_to_interactions[interaction.reactive_agent]
                    reactive[interaction_type][1].append(interaction)

    def __init__(self, confirmed_interactions, agent_to_money_earnt_last_round, agent_wealth_before,
                 agent_to_wealth_after, total_number_of_possible_interactions, agent_competency_before,
                 agent_competency_after, friend_type, mentor_type, help_type, theft_type, minimum_amount, competency_percentage):

        self.agent_to_interactions = {}
        self.confirmed_interactions = confirmed_interactions
        self.friend_type, self.mentor_type = friend_type, mentor_type
        self.help_type, self.theft_type = help_type, theft_type
        self.agent_competency_before = agent_competency_before
        self.agent_competency_after = agent_competency_after
        self.agent_to_money_earnt_last_round = agent_to_money_earnt_last_round
        self.agent_wealth_before = agent_wealth_before
        self.agent_to_wealth_after = agent_to_wealth_after
        self.total_number_of_possible_interactions = total_number_of_possible_interactions
        self.minimum_amount = minimum_amount
        super(TestingInteractionSuite, self).__init__()
        self.setUp()

        self.addTest(TestingInteractionCase('test_friendship', confirmed_interactions, agent_to_money_earnt_last_round,
                                            agent_wealth_before, agent_to_wealth_after,
                                            total_number_of_possible_interactions, agent_competency_before,
                                            agent_competency_after, friend_type, mentor_type, help_type, theft_type,
                                            minimum_amount,self.agent_to_interactions, competency_percentage))
        self.addTest(TestingInteractionCase('test_mentorship', confirmed_interactions, agent_to_money_earnt_last_round,
                                            agent_wealth_before, agent_to_wealth_after,
                                            total_number_of_possible_interactions, agent_competency_before,
                                            agent_competency_after, friend_type, mentor_type, help_type, theft_type,
                                            minimum_amount, self.agent_to_interactions, competency_percentage))
        self.addTest(TestingInteractionCase('test_theft', confirmed_interactions, agent_to_money_earnt_last_round,
                                            agent_wealth_before, agent_to_wealth_after,
                                            total_number_of_possible_interactions, agent_competency_before,
                                            agent_competency_after, friend_type, mentor_type, help_type, theft_type,
                                            minimum_amount, self.agent_to_interactions, competency_percentage))
        self.addTest(TestingInteractionCase('test_theft', confirmed_interactions, agent_to_money_earnt_last_round,
                                            agent_wealth_before, agent_to_wealth_after,
                                            total_number_of_possible_interactions, agent_competency_before,
                                            agent_competency_after, friend_type, mentor_type, help_type, theft_type,
                                            minimum_amount, self.agent_to_interactions, competency_percentage))

        self.addTest(TestingInteractionCase('test_help', confirmed_interactions, agent_to_money_earnt_last_round,
                                            agent_wealth_before, agent_to_wealth_after,
                                            total_number_of_possible_interactions, agent_competency_before,
                                            agent_competency_after, friend_type, mentor_type, help_type, theft_type,
                                            minimum_amount, self.agent_to_interactions, competency_percentage))

        self.addTest(TestingInteractionCase('test_number_of_interactions', confirmed_interactions, agent_to_money_earnt_last_round,
                                            agent_wealth_before, agent_to_wealth_after,
                                            total_number_of_possible_interactions, agent_competency_before,
                                            agent_competency_after, friend_type, mentor_type, help_type, theft_type,
                                            minimum_amount, self.agent_to_interactions, competency_percentage))

        self.addTest(TestingInteractionCase('test_interaction_money_changes', confirmed_interactions,
                                            agent_to_money_earnt_last_round,
                                            agent_wealth_before, agent_to_wealth_after,
                                            total_number_of_possible_interactions, agent_competency_before,
                                            agent_competency_after, friend_type, mentor_type, help_type, theft_type,
                                            minimum_amount, self.agent_to_interactions, competency_percentage))

        self.addTest(TestingInteractionCase('test_competency_increase', confirmed_interactions,
                                            agent_to_money_earnt_last_round,
                                            agent_wealth_before, agent_to_wealth_after,
                                            total_number_of_possible_interactions, agent_competency_before,
                                            agent_competency_after, friend_type, mentor_type, help_type, theft_type,
                                            minimum_amount, self.agent_to_interactions, competency_percentage))


class TestingInteractionCase(unittest.TestCase):

    def __init__(self, method_name, confirmed_interactions, agent_to_money_earnt_last_round, agent_wealth_before,
                 agent_to_wealth_after, total_number_of_possible_interactions, agent_competency_before,
                 agent_competency_after, friend_type, mentor_type, help_type, theft_type, minimum_amount,
                 agent_to_interactions, competency_percentage):
        self.confirmed_interactions = confirmed_interactions
        self.agent_to_interactions = agent_to_interactions
        self.friend_type, self.mentor_type, self.help_type, self.theft_type = friend_type, mentor_type, help_type, theft_type
        self.agent_competency_before = agent_competency_before
        self.agent_competency_after = agent_competency_after
        self.agent_to_money_earnt_last_round = agent_to_money_earnt_last_round
        self.agent_wealth_before = agent_wealth_before
        self.agent_to_wealth_after = agent_to_wealth_after
        self.total_number_of_possible_interactions = total_number_of_possible_interactions
        self.minimum_amount = minimum_amount
        self.competency_percentage = competency_percentage
        super(TestingInteractionCase, self).__init__(methodName=method_name)

    def test_friendship(self):
        for agent in self.agent_to_interactions:
            self.agent_to_number = {}
            proactive, reactive = self.agent_to_interactions[agent][self.friend_type]
            for i in proactive:
                a = i.reactive_agent
                self.agent_to_number[a] = 1 if a not in self.agent_to_number else self.agent_to_number[a] + 1
                self.assertLessEqual(self.agent_to_number[a],1,msg="More than 1 friendship interaction: " + str(i))
            for i in reactive:
                a = i.proactive_agent
                self.agent_to_number[a] = 1 if a not in self.agent_to_number else self.agent_to_number[a] + 1
                self.assertLessEqual(self.agent_to_number[a], 1, msg="More than 1 friendship interaction: " + str(proactive) + str(reactive))

    def test_mentorship(self):
        for agent in self.agent_to_interactions:
            self.agent_to_number = {}
            proactive, reactive = self.agent_to_interactions[agent][self.mentor_type]
            for i in proactive:
                a = i.reactive_agent
                self.agent_to_number[a] = 1 if a not in self.agent_to_number else self.agent_to_number[a] + 1
                self.assertLessEqual(self.agent_to_number[a],1,msg="More than 1 of Mentorship interaction: " + str(i))
            self.agent_to_number = {}
            for i in reactive:
                a = i.proactive_agent
                self.agent_to_number[a] = 1 if a not in self.agent_to_number else self.agent_to_number[a] + 1
                self.assertLessEqual(self.agent_to_number[a], 1, msg="More than 1 Mentorship interaction: " + str(i))

    def test_theft(self):
        for agent in self.agent_to_interactions:
            self.agent_to_number = {}
            proactive, reactive = self.agent_to_interactions[agent][self.theft_type]
            for i in proactive:
                a = i.reactive_agent
                self.agent_to_number[a] = 1 if a not in self.agent_to_number else self.agent_to_number[a] + 1
                self.assertLessEqual(self.agent_to_number[a],1,msg="More than 1 of Theft interaction: " + str(i))
            self.agent_to_number = {}
            for i in reactive:
                a = i.proactive_agent
                self.agent_to_number[a] = 1 if a not in self.agent_to_number else self.agent_to_number[a] + 1
                self.assertLessEqual(self.agent_to_number[a], 1, msg="More than 1 Theft interaction: " + str(i))
        for interaction in self.confirmed_interactions:
            if isinstance(interaction,self.theft_type):
                self.assertEqual(interaction.requested_agent,interaction.proactive_agent, msg="Not requested by proactive agent")

    def test_help(self):
        for agent in self.agent_to_interactions:
            self.agent_to_number = {}
            proactive, reactive = self.agent_to_interactions[agent][self.help_type]
            for i in proactive:
                a = i.reactive_agent
                self.agent_to_number[a] = 1 if a not in self.agent_to_number else self.agent_to_number[a] + 1
                self.assertLessEqual(self.agent_to_number[a],1,msg="More than 1 of Help interaction: " + str(i))
            self.agent_to_number = {}
            for i in reactive:
                a = i.proactive_agent
                self.agent_to_number[a] = 1 if a not in self.agent_to_number else self.agent_to_number[a] + 1
                self.assertLessEqual(self.agent_to_number[a], 1, msg="More than 1 Help interaction: " + str(i))

        for interaction in self.confirmed_interactions:
            if isinstance(interaction,self.help_type):
                self.assertEqual(interaction.requested_agent,interaction.proactive_agent, msg="Not requested by proactive agent")

    def test_number_of_interactions(self):
        lim = self.total_number_of_possible_interactions
        types = [self.friend_type, self.mentor_type, self.help_type, self.theft_type]

        for agent in self.agent_to_interactions:
            tot = 0
            for type in types:
                interactions = self.agent_to_interactions[agent][type]
                tot += len(interactions[0])
                if not(type == self.help_type or type == self.theft_type):
                    tot += len(interactions[1])
            self.assertLessEqual(tot,round(lim), msg="Too many interactions for agent " + str(agent))

    def test_interaction_money_changes(self):
        cop = self.agent_wealth_before.copy()
        for interaction in self.confirmed_interactions:
            if isinstance(interaction,self.theft_type) and interaction.is_success:
                fee = 0
                if interaction.reactive_agent in self.agent_to_money_earnt_last_round:
                    fee = self.agent_to_money_earnt_last_round[interaction.reactive_agent]
                    fee = int(max(fee * 0.5, 0.5*self.minimum_amount))
                cop[interaction.proactive_agent] += fee
                cop[interaction.reactive_agent] -= fee

            if isinstance(interaction,self.help_type) and interaction.is_success:
                fee = 0
                if interaction.reactive_agent in self.agent_to_money_earnt_last_round:
                    fee = self.agent_to_money_earnt_last_round[interaction.reactive_agent]
                    fee = int(max(fee * 0.05, 0.25*self.minimum_amount))
                cop[interaction.reactive_agent] += fee
                cop[interaction.proactive_agent] -= fee

        for agent in self.agent_to_wealth_after:
            self.assertEqual(cop[agent],self.agent_to_wealth_after[agent],msg="Wealth value incorrect")

    def test_competency_increase(self):
        comp = self.agent_competency_before.copy()
        for interaction in self.confirmed_interactions:
            if isinstance(interaction,self.mentor_type):
                mdif = comp[interaction.proactive_agent].mining_skill - comp[interaction.reactive_agent].mining_skill
                adif = comp[interaction.proactive_agent].appraisal_skill - comp[interaction.reactive_agent].appraisal_skill
                if mdif > 0 and adif > 0:
                    bdiff = min(mdif,adif)
                    mdif, adif = bdiff, bdiff
                elif mdif > 0:
                    adif = 0
                elif adif > 0:
                    mdif = 0

                percent_mdif, percent_adif = mdif * self.competency_percentage, adif * self.competency_percentage

                new_mine = min(comp[interaction.reactive_agent].mining_skill + percent_mdif, 1)
                new_appr = min(comp[interaction.reactive_agent].appraisal_skill + percent_adif, 1)

                c = comp[interaction.reactive_agent].copy()
                c.update(new_mine, new_appr)
                comp[interaction.reactive_agent] = c

        for agent in self.agent_competency_after:
            self.assertAlmostEqual(comp[agent].mining_skill, self.agent_competency_after[agent].mining_skill,places=2,msg="Mentorship competency increase failed")
            self.assertAlmostEqual(comp[agent].appraisal_skill, self.agent_competency_after[agent].appraisal_skill, places=2, msg="Mentorship competency increase failed")


class TestingRoundSuite(unittest.TestSuite):

    def __init__(self, confirmed_interactions, agent_to_money_earnt_this_round, agent_wealth_before,
                 agent_to_wealth_after, agent_competency_before, agent_competency_after, friend_type, mentor_type, help_type, theft_type):
        self.confirmed_interactions = confirmed_interactions
        self.agent_to_money_earnt_this_round = agent_to_money_earnt_this_round
        self.agent_wealth_before = agent_wealth_before
        self.agent_wealth_after = agent_to_wealth_after
        self.agent_competency_after = agent_competency_after
        self.agent_competency_before = agent_competency_before
        self.friend_type = friend_type
        self.mentor_type = mentor_type

        a = TestingRoundTestCase('test_interaction_earnings', confirmed_interactions,
                                  agent_to_money_earnt_this_round, agent_wealth_before, agent_to_wealth_after,
                                  agent_competency_before, agent_competency_after, friend_type, mentor_type, help_type, theft_type)

        b = TestingRoundTestCase('test_competency_increase', confirmed_interactions,
                                  agent_to_money_earnt_this_round, agent_wealth_before, agent_to_wealth_after,
                                  agent_competency_before, agent_competency_after, friend_type, mentor_type, help_type, theft_type)

        c = TestingRoundTestCase('test_notifications', confirmed_interactions,
                                 agent_to_money_earnt_this_round, agent_wealth_before, agent_to_wealth_after,
                                 agent_competency_before, agent_competency_after, friend_type, mentor_type, help_type,
                                 theft_type)

        super(TestingRoundSuite, self).__init__([a,b,c])


class TestingRoundTestCase(unittest.TestCase):

    def __init__(self, methodName, confirmed_interactions, agent_to_money_earnt_this_round, agent_wealth_before,
                 agent_to_wealth_after, agent_competency_before, agent_competency_after, friend_type, mentor_type, help_type, theft_type):
        self.confirmed_interactions = confirmed_interactions
        self.agent_to_money_earnt_this_round = agent_to_money_earnt_this_round
        self.agent_wealth_before = agent_wealth_before
        self.agent_wealth_after = agent_to_wealth_after
        self.agent_competency_after = agent_competency_after
        self.agent_competency_before = agent_competency_before
        self.friend_type = friend_type
        self.mentor_type = mentor_type
        self.help_type = help_type
        self.theft_type = theft_type
        super(TestingRoundTestCase, self).__init__(methodName=methodName)

    def test_interaction_earnings(self):
        cop = self.agent_wealth_before.copy()
        for interaction in self.confirmed_interactions:
            if isinstance(interaction,self.friend_type) and interaction.is_success:
                proactive, reactive = interaction.proactive_agent, interaction.reactive_agent
                proactive_fee = int(0.1*self.agent_to_money_earnt_this_round[reactive])
                reactive_fee = int(0.1*self.agent_to_money_earnt_this_round[proactive])
                cop[proactive] += proactive_fee
                cop[reactive] += reactive_fee
            if isinstance(interaction,self.mentor_type) and interaction.is_success:
                proactive, reactive = interaction.proactive_agent, interaction.reactive_agent
                fee = int(0.25*self.agent_to_money_earnt_this_round[reactive])
                cop[proactive] += fee
                cop[reactive] -= fee
        for agent in self.agent_wealth_after:
            self.assertEqual(cop[agent],self.agent_wealth_after[agent], msg="Invalid interaction exchange " + str(agent))

    def test_competency_increase(self):
        for agent in self.agent_competency_before:
            a = self.agent_competency_after[agent]
            a_mining, a_appraisal = a.mining_skill, a.appraisal_skill

            b = self.agent_competency_before[agent]
            b_appraisal, b_mining = b.appraisal_skill, b.mining_skill

            self.assertGreaterEqual(a_mining,b_mining,msg="Mining didn't increase for agent: " + str(agent))
            self.assertGreaterEqual(a_appraisal, b_appraisal, msg="Appraisal didn't increase for agent: " + str(agent))

    def assertInteractionIn(self, interaction, interaction_list, msg=""):
        flag = False
        for i in interaction_list:
            if type(i) == type(interaction) and i.proactive_agent == interaction.proactive_agent and i.reactive_agent == interaction.reactive_agent:
                flag = True
        return self.assertTrue(flag,msg=msg)

    def check_notification_for_agent(self,agent, interaction):
        interactions = agent.interactions
        agent_info = agent.agent_information
        # Agent -> Wealth, Personality, Competency, Interactions, NoRounds

        interaction_type = type(interaction)
        if interaction.is_present(agent):
            p,r = interactions[interaction_type]
            mesg = str(agent) + " wasn't notified about this " + str(interaction)
            self.assertInteractionIn(interaction, p + r, msg=mesg)

        if interaction_type != self.theft_type or interaction_type == self.theft_type and interaction.is_caught:
            for a in interaction.other_agents(agent):
                interaction_info = agent_info[a]["Interactions"]
                #print("Info",interaction_info)
                p, r = interaction_info[interaction_type]
                mesg = str(agent) + " didn't store info of " + str(interaction) + " for " + str(a)
                #if type(interaction) == self.theft_type:
                    #print(interaction, interaction.is_caught)
                self.assertInteractionIn(interaction, p + r, msg=mesg)

    def test_notifications(self):
        for interaction in self.confirmed_interactions:
            if interaction.is_success:
                for agent in self.agent_wealth_before:
                    self.check_notification_for_agent(agent,interaction)



