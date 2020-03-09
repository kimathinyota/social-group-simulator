import mysql.connector


class Analysis:

    def connect_to_db(self):
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="Ribbon99",
            database="interactiondb",
            auth_plugin='mysql_native_password'
        )

        self.mycursor = self.mydb.cursor()

    def __init__(self, interaction_types, number_of_rounds):
        self.mydb = None
        self.mycursor = None
        self.connect_to_db()

    def include_agents(self, agents):
        for agent in agents:
            insert_into_agent(agent, self.mycursor, self.mydb)

    def add_interaction(self, interaction, similarity_tuple, round):
        insert_into_interactions(round,interaction,similarity_tuple,self.mycursor,self.mydb)

    def add_interaction_money_earnings(self, agent, is_proactive, amount, interaction_type, round):
        insert_into_interaction_wealth_earnings(agent,round,interaction_type,is_proactive,amount,self.mycursor,self.mydb)

    def add_interaction_comp_earnings(self, agent, is_proactive, mining_amount, appraisal_amount, interaction_type, round):
        insert_into_interaction_competency_earnings(agent,round,interaction_type,is_proactive,mining_amount,appraisal_amount,self.mycursor,self.mydb)

    def add_money_earnings(self, agent, amount, round):
        insert_into_wealth_earnings(agent,round, amount,self.mycursor,self.mydb)

    def add_competency_earnings(self, agent, mining_amount, appraisal_amount, round):
        insert_into_competency_earnings(agent,round,appraisal_amount,mining_amount,self.mycursor,self.mydb)

    def print_all(self):
        print("Check DB")


def insert_into_agent(agent, mycursor, mydb):
    a = "(agentID, agentName, H1, H2, H3, H4, E1, E2, E3, E4, X1, X2, X3, X4, A1, A2, A3, A4, C1, C2, C3, C4, O1, O2, O3, O4, initialMining, initialAppraisal)"
    v = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    table = "Agent"
    sql = "INSERT INTO " + table + " " + a + " VALUES " + v
    personality = agent.personality_template.as_list(agent.personality)
    lst = []
    lst.append(str(agent.id))
    lst.append(agent.name)
    lst += personality
    lst.append(agent.competency.mining_skill)
    lst.append(agent.competency.appraisal_skill)

    val = tuple(i for i in lst)

    mycursor.execute(sql, val)
    mydb.commit()


def insert_into_competency_earnings(agent, round, appraisal, mining, mycursor, mydb):
    a = "(agentID, round, appraisal, mining)"
    v = "(%s, %s, %s, %s)"
    table = "CompetencyEarnings"
    sql = "INSERT INTO " + table + " " + a + " VALUES " + v
    val = (str(agent.id), round, appraisal, mining)
    mycursor.execute(sql, val)
    mydb.commit()


def insert_into_interaction_competency_earnings(agent, round, interaction_type, is_proactive, mining, appraisal, mycursor, mydb):
    a = "(agentID, round, interactionType, isProactive, mining, appraisal)"
    v = "(%s, %s, %s, %s, %s, %s)"
    table = "InteractionCompetencyEarnings"
    sql = "INSERT INTO " + table + " " + a + " VALUES " + v
    val = (str(agent.id), round, interaction_type.__name__, is_proactive , mining, appraisal)
    mycursor.execute(sql, val)
    mydb.commit()


def insert_into_interactions(round, interaction, similarity_tuple, mycursor, mydb):
    a = "(round, interactionType, proactiveAgentID, reactiveAgentID, personalitySimilarity, miningSimilarity, appraisalSimilarity)"
    v = "(%s, %s, %s, %s, %s, %s, %s)"
    table = "Interactions"
    sql = "INSERT INTO " + table + " " + a + " VALUES " + v
    p, m, a = similarity_tuple
    val = (round, type(interaction).__name__, str(interaction.proactive_agent.id), str(interaction.reactive_agent.id), p, m, a)
    mycursor.execute(sql, val)
    mydb.commit()


def insert_into_interaction_wealth_earnings(agent, round, interaction_type, is_proactive, amount, mycursor, mydb):
    a = "(agentID, round, interactionType, isProactive, amount)"
    v = "(%s, %s, %s, %s, %s)"
    table = "InteractionWealthEarnings"
    sql = "INSERT INTO " + table + " " + a + " VALUES " + v
    val = (str(agent.id), round, interaction_type.__name__, is_proactive , amount)
    mycursor.execute(sql, val)
    mydb.commit()


def insert_into_wealth_earnings(agent, round, amount, mycursor, mydb):
    a = "(agentID, round, amount)"
    v = "(%s, %s, %s)"
    table = "WealthEarnings"
    sql = "INSERT INTO " + table + " " + a + " VALUES " + v
    val = (str(agent.id), round, amount)
    mycursor.execute(sql, val)
    mydb.commit()


















