from abc import ABC, abstractmethod
from src.Agent import Agent
from src.Helper import *
from src.GoldMiningEnvironment import *


class Interaction(ABC):

    def __init__(self, proactive_agent, reactive_agent, single_role, request_bidirectional, requires_acceptance, should_always_notify_all, environment):
        self.proactive_agent = proactive_agent
        self.reactive_agent = reactive_agent
        self.environment = environment
        self.is_success = False
        self.is_accepted_or_rejected = False
        self.is_single_role = single_role
        self.should_always_notify_all = should_always_notify_all
        self.requires_acceptance = requires_acceptance
        self.is_request_bidirectional = request_bidirectional
        self.is_proactive_busy_during_initiation = True
        self.is_reactive_busy_during_initiation = self.requires_acceptance
        self.requested_agent = None

    def reset(self):
        self.requested_agent = None
        self.is_success = False
        self.is_accepted_or_rejected = False

    def is_busy_during_initiation(self, agent):
        return (self.reactive_agent == agent and self.is_reactive_busy_during_initiation) \
               or (self.proactive_agent == agent and self.is_proactive_busy_during_initiation)

    def contains(self,agent1, agent2):
        return self.is_present(agent1) and self.is_present(agent2)

    def is_present(self,agent):
        return self.proactive_agent==agent or self.requested_agent==agent

    def get_requested_agent(self):
        return self.requested_agent

    def other_agent(self, agent):
        if not self.is_present(agent):
            return None
        return self.reactive_agent if self.proactive_agent == agent else self.proactive_agent

    def get_accepted_agent(self):
        if self.requested_agent is not None:
            return self.proactive_agent if self.reactive_agent==self.reactive_agent else self.reactive_agent
        return None

    def is_requestable_by(self, agent):
        is_proactive = agent == self.proactive_agent
        return self.is_request_bidirectional or (not self.is_request_bidirectional and is_proactive)

    def other_agents(self, agent):
        if not self.is_present(agent):
            return [self.proactive_agent, self.reactive_agent]
        return [self.other_agent(agent)]

    def request(self, agent):
        # If already accepted, do nothing
        if self.is_accepted_or_rejected:
            return None

        # Agent cant request it
        if not self.is_requestable_by(agent):
            return None

        # if doesn't need acceptance
        if not self.requires_acceptance:
            self.is_success = True
            self.is_accepted_or_rejected = True
            self.requested_agent = agent
            self.accept()
            return None

        # If a request already exists then accept
        if self.reactive_agent is not None and not self.is_accepted_or_rejected:
            self.is_success = True
            self.accept()
            return None

        # If first request, then query other agent for response
        self.requested_agent = agent
        responding_agent = self.other_agent(agent)
        self.respond(responding_agent)


    def can_agent_respond(self, agent):
        is_not_responded = not self.is_accepted_or_rejected
        is_requested = self.requested_agent is not None
        does_agent_belong = self.is_present(agent)
        is_not_requested_by_agent = (not self.requested_agent == agent)
        return  does_agent_belong and is_not_responded and is_requested and is_not_requested_by_agent


    def respond(self, agent):
        if self.can_agent_respond(agent):
            response = agent.accept_interaction(self)
            if response is not None:
                # this means interaction happened
                self.is_success = response
                self.accept()


    # NEED TO DO COPY
    def copy(self):
        pass

    def accept(self):
        self.is_accepted_or_rejected = True
        # need to tell environment that interaction happened
        self.environment.notify_interaction(self)

        # needs to tell agents involved in interaction that it happened
        self.proactive_agent.interaction_happened(self)
        self.reactive_agent.interaction_happened(self)
        self.exchange()
        if self.is_success and self.should_always_notify_all:
            self.notify_all()

    def get_is_success(self):
        return self.is_success

    def can_happen(self):
        pass

    def exchange(self):
        pass

    def notify_all(self):
        self.environment.notify_all_agents(self)

    def get_requires_acceptance(self):
        return self.get_requires_acceptance

    def get_is_bidirectional(self):
        return self.is_bidirectional

    def get_proactive_agent(self):
        return self.proactive_agent

    def get_reactive_agent(self):
        return self.reactive_agent


class Friendship(Interaction):

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self, proactive_agent, reactive_agent, True, True, True, True, environment)

    def exchange(self):
        friend_token = FriendshipToken(self.proactive_agent, self.reactive_agent)
        exchange = Exchange(self.proactive_agent, self.reactive_agent, friend_token, friend_token)
        ResourceMiningEnvironment.process_later(self.environment,self,exchange)

    def can_happen(self):
        # only condition is that other person accepts
        return True

    @classmethod
    def from_interaction(cls, friendship):
        a = cls(friendship.proactive_agent, friendship.reactive_agent, friendship.environment)
        a.is_accepted_or_rejected = friendship.is_accepted_or_rejected
        a.is_success = friendship.is_success
        a.requested_agent = friendship.requested_agent
        return a

    def copy(self):
        copy = Friendship.from_interaction(self)
        return copy


class Mentorship(Interaction):

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self,proactive_agent,reactive_agent,False,True,True, True, environment)

    def exchange(self):
        mentor_token = MentorshipToken(self.proactive_agent, self.reactive_agent)
        exchange = Exchange(self.proactive_agent, self.reactive_agent, mentor_token, mentor_token)
        ResourceMiningEnvironment.process_later(self.environment,self,exchange)

    @classmethod
    def from_interaction(cls, mentorship):
        a = cls(mentorship.proactive_agent,mentorship.reactive_agent,mentorship.environment)
        a.is_accepted_or_rejected = mentorship.is_accepted_or_rejected
        a.is_success = mentorship.is_success
        a.requested_agent = mentorship.requested_agent
        return a

    def copy(self):
        copy = Mentorship.from_interaction(self)
        return copy

    def can_happen(self):
        # only condition is that proactive_agent has a greater competency than reactive agent
        return self.proactive_agent.competency.contains_greater_competency_skill(self.reactive_agent.competency)


class Help(Interaction):

    def reset(self):
        super(Theft, self).reset()
        self.helping_funds = self.environment.helping_funds

    @classmethod
    def from_interaction(cls, help):
        a = cls(help.proactive_agent, help.reactive_agent, help.environment)
        a.is_accepted_or_rejected = help.is_accepted_or_rejected
        a.is_success = help.is_success
        a.requested_agent = help.requested_agent
        a.helping_funds = help.helping_funds
        return a

    def copy(self):
        copy = Help.from_interaction(self)
        return copy

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self, proactive_agent, reactive_agent,False,False,False, True, environment)
        self.helping_funds = self.environment.helping_funds

    def get_helping_funds(self):
        return self.helping_funds

    def exchange(self):
        help_token = HelpToken(self.proactive_agent, self.reactive_agent, self.helping_funds)
        exchange = Exchange(self.proactive_agent, self.reactive_agent,help_token, help_token)
        ResourceMiningEnvironment.process_now(self.environment,exchange)

    def can_happen(self):
        # proactive agent has enough funds to help
        return self.proactive_agent.wealth > self.helping_funds


class Theft(Interaction):

    def reset(self):
        super(Theft, self).reset()
        self.stolen_funds = self.environment.theft_amount(self.reactive_agent)
        self.is_caught = None

    @classmethod
    def from_interaction(cls, theft):
        a = cls(theft.proactive_agent, theft.reactive_agent, theft.environment)
        a.is_accepted_or_rejected = theft.is_accepted_or_rejected
        a.is_success = theft.is_success
        a.requested_agent = theft.requested_agent
        a.stolen_funds = theft.stolen_funds
        a.is_caught = theft.is_caught
        return a

    def copy(self):
        copy = Theft.from_interaction(self)
        return copy

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self,proactive_agent,reactive_agent,False,False,False, False, environment)
        self.stolen_funds = None
        self.is_caught = None
        self.reset()

    def accept(self):
        super(Theft, self).accept()
        caught = self.determine_if_caught()
        if caught:
            self.notify_all()

    def exchange(self):
        theft_token = HelpToken(self.proactive_agent, self.reactive_agent, self.stolen_funds)
        exchange = Exchange(self.proactive_agent, self.reactive_agent, theft_token, theft_token)
        ResourceMiningEnvironment.process_now(self.environment, exchange)

    def notify_all(self):
        pass

    def get_stolen_funds(self):
        return self.stolen_funds

    def can_happen(self):
        # no condition
        return True

    def determine_if_caught(self):
        if self.is_caught is not None:
            return self.is_caught
        # More you steal, less likely you are to be caught

        c = self.proactive_agent.personality_template.dimension_percentage(self.proactive_agent.personality,'C')
        n = self.proactive_agent.personality_template.dimension_percentage(self.proactive_agent.personality, 'E')
        x = self.proactive_agent.get_number_of_thefts()
        a = self.environment.get_minimum_thefts_to_maximise_theft_skill()

        # below: probability of being caught
        p = 0.8 - 0.4(1+c)*x/(a*(1+n))
        p += 0.1

        self.is_caught = random_boolean_variable(p)

        return self.is_caught


class FriendshipToken(Token):

    def remove_from(self, agent):
        return True

    def add_to(self, agent):
        if not (agent == self.friend1 or agent == self.friend2):
            return True

        friend = self.friend2 if agent == self.friend1 else self.friend1

        amount = ResourceMiningEnvironment.get_agent_earnings_this_round(friend.environment, friend)
        amount *= 0.1
        amount = int(amount)

        if amount > 0:
            return ResourceToken.change(amount, agent)

        return True

    def __init__(self, friend1, friend2):
        self.friend1 = friend1
        self.friend2 = friend2


class MentorshipToken(Token):

    def add_to(self, agent):
        # Only add competency to mentee
        if agent == self.mentee:
            CompetencyToken(self.p, self.skill_list).add_to(agent)

        if agent == self.mentor:
            self.amount = self.get_mentee_earnings()
            if self.amount > 0:
                return ResourceToken.change(self.amount, agent)

    def get_mentee_earnings(self):
        if self.amount is None:
            self.amount = ResourceMiningEnvironment.get_agent_earnings_this_round(self.mentee.environment, self.mentee)
            self.amount *= 0.25
            self.amount = int(self.amount)
        return self.amount

    def remove_from(self, agent):
        # Only remove from mentee
        if not self.mentee == agent:
            return True
        self.amount = self.get_mentee_earnings()
        if self.amount > 0:
            return ResourceToken.change(-self.amount, agent)

        return True

    def __init__(self, mentor, mentee):
        # will return list of skills better than mentee and difference for each of those skills
        self.skill_list, mp, ap = mentor.competency.better_skills(mentee.competency)
        self.mentor = mentor
        self.mentee = mentee
        self.amount = None
        if mp and ap is None:
            self.p = 0
        else:
            self.p = mp if (mp is not None and (ap is None or ap >= mp)) else ap
            # increasing by 15% of the smallest difference
            self.p /= 15


class TheftToken(Token):

    def add_to(self, agent):
        if agent == self.thief:
            if self.amount > 0:
                return ResourceToken.change(self.amount, agent)

    def remove_from(self, agent):
        if agent == self.victim:
            if self.amount > 0:
                return ResourceToken.change(-self.amount, agent)

    def __init__(self, thief, victim, amount):
        self.thief = thief
        self.victim = victim
        self.amount = amount


class HelpToken(Token):

    def add_to(self, agent):
        if agent == self.helped:
            if self.amount > 0:
                return ResourceToken.change(self.amount, agent)

    def remove_from(self, agent):
        if agent == self.helper:
            if self.amount > 0:
                return ResourceToken.change(-self.amount, agent)

    def __init__(self, helper, helped, amount):
        self.helped = helped
        self.helper = helper
        self.amount = amount