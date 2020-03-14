import mysql.connector
from src.GoldMiningEnvironment import *


class Analysis:

    def connect_to_db(self):
        self.is_connected = True
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="Ribbon99",
            database="interactiondb",
            auth_plugin='mysql_native_password'
        )

        self.mycursor = self.mydb.cursor()

    def close(self):
        if self.is_connected:
            self.is_connected = False
            self.mycursor.close()
            self.mydb.close()

    def __init__(self):
        self.mydb = None
        self.mycursor = None
        self.queries = []
        self.is_connected = False

    def include_agents(self, agents):
        for agent in agents:
            self.insert_into_agent(agent)

    def add_interaction(self, interaction, similarity_tuple, round):
        self.insert_into_interactions(round,interaction,similarity_tuple)

    def add_interaction_money_earnings(self, agent, is_proactive, amount, interaction_type, round):
        self.insert_into_interaction_wealth_earnings(agent,round,interaction_type,is_proactive,amount)
        self.insert_into_wealth_earnings(agent,round,amount)

    def add_interaction_comp_earnings(self, agent, is_proactive, mining_amount, appraisal_amount, interaction_type, round):
        self.insert_into_interaction_competency_earnings(agent,round,interaction_type,is_proactive,mining_amount,appraisal_amount)
        self.insert_into_competency_earnings(agent,round,appraisal_amount,mining_amount)

    def add_money_earnings(self, agent, amount, round):
        self.insert_into_wealth_earnings(agent,round, amount)

    def add_competency_earnings(self, agent, mining_amount, appraisal_amount, round):
        self.insert_into_competency_earnings(agent,round,appraisal_amount,mining_amount)

    def query_all(self):
        if len(self.queries) == 0:
            return None
        if not self.is_connected:
            self.connect_to_db()
        for query in self.queries:
            sql, val = query
            try:
                self.mycursor.execute(sql,val)
            except mysql.connector.errors.Error:
                pass

        self.mydb.commit()
        self.close()
        print("Finished executing queries")

    def remove_all_data(self):
        if not self.is_connected:
            self.connect_to_db()
        sql = "TRUNCATE `interactionDB`.`Agent`;" \
              "TRUNCATE `interactionDB`.`CompetencyEarnings`;" \
              "TRUNCATE `interactionDB`.`InteractionCompetencyEarnings`;" \
              "TRUNCATE `interactionDB`.`Interactions`;" \
              "TRUNCATE `interactionDB`.`InteractionWealthEarnings`;" \
              "TRUNCATE `interactionDB`.`WealthEarnings`;"
        self.mycursor.execute(sql)
        self.close()

    def insert_into_agent(self, agent):
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

        self.queries.append((sql,val))

    def insert_into_competency_earnings(self, agent, round, appraisal, mining):
        a = "(agentID, round, appraisal, mining)"
        v = "(%s, %s, %s, %s)"
        table = "CompetencyEarnings"
        sql = "INSERT INTO " + table + " " + a + " VALUES " + v
        val = (str(agent.id), round, appraisal, mining)
        self.queries.append((sql,val))

    def insert_into_interaction_competency_earnings(self, agent, round, interaction_type, is_proactive, mining, appraisal):
        a = "(agentID, round, interactionType, isProactive, mining, appraisal)"
        v = "(%s, %s, %s, %s, %s, %s)"
        table = "InteractionCompetencyEarnings"
        sql = "INSERT INTO " + table + " " + a + " VALUES " + v
        val = (str(agent.id), round, interaction_type.__name__, is_proactive, mining, appraisal)
        self.queries.append((sql,val))

    def insert_into_interactions(self, round, interaction, similarity_tuple):
        a = "(round, interactionType, proactiveAgentID, reactiveAgentID, personalitySimilarity, miningSimilarity, appraisalSimilarity)"
        v = "(%s, %s, %s, %s, %s, %s, %s)"
        table = "Interactions"
        sql = "INSERT INTO " + table + " " + a + " VALUES " + v
        p, m, a = similarity_tuple
        # x = "Finna " + str(p) + " " + str(m) + " " + str(a)
        val = (
        round, type(interaction).__name__, str(interaction.proactive_agent.id), str(interaction.reactive_agent.id), p,
        m, a)
        self.queries.append((sql,val))

    def insert_into_interaction_wealth_earnings(self, agent, round, interaction_type, is_proactive, amount):
        a = "(agentID, round, interactionType, isProactive, amount)"
        v = "(%s, %s, %s, %s, %s)"
        table = "InteractionWealthEarnings"
        sql = "INSERT INTO " + table + " " + a + " VALUES " + v
        val = (str(agent.id), round, interaction_type.__name__, is_proactive, amount)
        self.queries.append((sql,val))

    def insert_into_wealth_earnings(self, agent, round, amount):
        a = "(agentID, round, amount)"
        v = "(%s, %s, %s)"
        table = "WealthEarnings"
        sql = "INSERT INTO " + table + " " + a + " VALUES " + v
        val = (str(agent.id), round, amount)
        self.queries.append((sql,val))

    def get_x_vs_round_to_earn(self, agent_variable, agent_name):
        if not self.is_connected:
            self.connect_to_db()
        sql = "SELECT " + agent_variable + ", round, AVG(tot)" \
              "FROM  (SELECT K.*, SUM(amount) AS tot" \
              "       FROM (SELECT T.*, R.round, R.amount" \
              "             FROM interactionDB.WealthEarnings AS R" \
              "             LEFT JOIN (SELECT agentID, agentName, " \
              "                               (H1 + H2 + H3 + H4) AS PH," \
              "                               (E1 + E2 + E3 + E4) AS PE," \
              "                               (X1 + X2 + X3 + X4) AS PX," \
              "                               (A1 + A2 + A3 + A4) AS PA," \
              "                               (C1 + C2 + C3 + C4) AS PC," \
              "                               (O1 + O2 + O3 + O4) AS PO," \
              "                               initialMining AS CM," \
              "                               initialAppraisal AS CA" \
              "                       FROM interactionDB.Agent) AS T" \
              "             ON T.agentID = R.agentID" \
              "             WHERE T.agentName = \"" + agent_name + "\") K" \
              "             GROUP BY round, agentID) AS O" \
              "        GROUP BY round," + agent_variable

        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()
        x, y, z = [], [], []
        for r in result:
            a, b, c = r
            x.append(float(a))
            y.append(float(b))
            z.append(float(c))
        self.close()
        return x, y, z

    def get_x_vs_total_earn(self, agent_variable, agent_name):
        if not self.is_connected:
            self.connect_to_db()

        sql = " SELECT " +agent_variable + ", AVG(tot) FROM"  \
              "     (SELECT "+agent_variable+", SUM(amount) AS tot" \
              "     FROM (SELECT T.*, R.round, R.amount" \
              "     FROM interactionDB.WealthEarnings AS R" \
              "     LEFT JOIN (SELECT agentID, agentName, " \
              "                      (H1 + H2 + H3 + H4) AS PH," \
              "                      (E1 + E2 + E3 + E4) AS PE," \
              "                      (X1 + X2 + X3 + X4) AS PX," \
              "                      (A1 + A2 + A3 + A4) AS PA," \
              "                      (C1 + C2 + C3 + C4) AS PC," \
              "                      (O1 + O2 + O3 + O4) AS PO," \
              "                      initialMining AS CM," \
              "                      initialAppraisal AS CA" \
              "              FROM interactionDB.Agent) AS T" \
              "              ON T.agentID = R.agentID" \
              "     WHERE T.agentName = \""+agent_name+"\") K" \
              "     GROUP BY agentID) AS GB " \
              " GROUP BY " + agent_variable

        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()
        x, y = [], []
        for r in result:
            a, b = r
            x.append(float(a))
            y.append(float(b))
        self.close()
        return x, y

    def get_x_vs_round_to_comp(self, agent_variable, agent_name):
        if not self.is_connected:
            self.connect_to_db()
        sql = " SELECT "+agent_variable+", round, AVG(tot)" \
              " FROM  (SELECT K.*, SUM(amount) AS tot" \
              "        FROM (SELECT T.*, R.round, (R.appraisal + R.mining)  AS amount" \
              "              FROM interactionDB.CompetencyEarnings AS R" \
              "              LEFT JOIN (SELECT agentID, agentName, " \
              "                               (H1 + H2 + H3 + H4) AS PH," \
              "                               (E1 + E2 + E3 + E4) AS PE," \
              "                               (X1 + X2 + X3 + X4) AS PX," \
              "                               (A1 + A2 + A3 + A4) AS PA," \
              "                               (C1 + C2 + C3 + C4) AS PC," \
              "                               (O1 + O2 + O3 + O4) AS PO," \
              "                               initialMining AS CM," \
              "                               initialAppraisal AS CA" \
              "                         FROM interactionDB.Agent) AS T" \
              "              ON T.agentID = R.agentID" \
              "              WHERE T.agentName = \""+agent_name+"\") K" \
              "        GROUP BY round, agentID) AS O" \
              " GROUP BY round, " + agent_variable
        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()
        x, y, z = [], [], []
        for r in result:
            a, b, c = r
            x.append(float(a))
            y.append(float(b))
            z.append(float(c))
        self.close()
        return x, y, z

    def get_x_vs_total_comp(self, agent_variable, agent_name):
        if not self.is_connected:
            self.connect_to_db()
        sql = "SELECT "+agent_variable+", AVG(tot)" \
              "FROM  (SELECT "+agent_variable+", SUM(amount) AS tot " \
              "       FROM (SELECT T.*, R.round, (R.appraisal + R.mining) AS amount " \
              "             FROM interactionDB.CompetencyEarnings AS R " \
              "             LEFT JOIN (SELECT agentID, agentName, " \
              "                              (H1 + H2 + H3 + H4) AS PH, " \
              "                              (E1 + E2 + E3 + E4) AS PE, " \
              "                              (X1 + X2 + X3 + X4) AS PX, " \
              "                              (A1 + A2 + A3 + A4) AS PA, " \
              "                              (C1 + C2 + C3 + C4) AS PC, " \
              "                              (O1 + O2 + O3 + O4) AS PO, " \
              "                              initialMining AS CM, " \
              "                              initialAppraisal AS CA " \
              "             FROM interactionDB.Agent) AS T " \
              "             ON T.agentID = R.agentID " \
              "             WHERE T.agentName = \""+agent_name+"\") K " \
              "       GROUP BY agentID) AS GB " \
              "GROUP BY " + agent_variable
        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()
        x, y = [], []
        for r in result:
            a, b = r
            x.append(float(a))
            y.append(float(b))
        self.close()
        return x, y

    def get_interaction_vs_round_to_earn(self):
        if not self.is_connected:
            self.connect_to_db()
        sql = " SELECT round, interactionType, isProactive, AVG(tot) FROM" \
              " (SELECT round, interactionType, isProactive, amount, agentID, agentName, AVG(tot) AS tot FROM" \
              " (SELECT round, interactionType, isProactive, amount, k.*, SUM(amount) AS tot" \
              " FROM interactionDB.InteractionWealthEarnings AS t" \
              " LEFT JOIN (SELECT agentID, agentName," \
              " (H1 + H2 + H3 + H4) AS H," \
              " (E1 + E2 + E3 + E4) AS E," \
              " (X1 + X2 + X3 + X4) AS X," \
              " (A1 + A2 + A3 + A4) AS A," \
              " (C1 + C2 + C3 + C4) AS C," \
              " (O1 + O2 + O3 + O4) AS O," \
              " initialMining AS mining," \
              " initialAppraisal AS appraisal" \
              " FROM interactionDB.Agent) AS k" \
              " ON k.agentID = t.agentID" \
              " GROUP BY agentID, round, interactionType, isProactive) AS j" \
              " GROUP BY round, interactionType, isProactive, H, E, X, A, C, O, mining, appraisal) AS cool	" \
              " GROUP BY round, interactionType, isProactive"

        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()

        friendship_round_to_total = {}

        labels, x, y = [], [], []
        for r in result:
            rnd, iT, iP, t = r
            if iT == "Friendship":
                tot = friendship_round_to_total[rnd] if rnd in friendship_round_to_total else 0
                tot += float(t)
                friendship_round_to_total[rnd] = tot
            else:
                labels.append(str((iT, iP)))
                x.append(rnd)
                y.append(float(t))

        for rnd in friendship_round_to_total:
            labels.append("Friendship")
            x.append(rnd)
            y.append(friendship_round_to_total[rnd])
        self.close()
        return labels, x, y

    def get_interaction_vs_total_earn(self):
        if not self.is_connected:
            self.connect_to_db()
        sql = " SELECT interactionType, isProactive, SUM(tot)" \
              " FROM   (SELECT round, interactionType, isProactive, amount, agentID, agentName, AVG(tot) AS tot " \
              " FROM (SELECT round, interactionType, isProactive, amount, k.*, SUM(amount) AS tot" \
              " FROM interactionDB.InteractionWealthEarnings AS t" \
              " LEFT JOIN (SELECT agentID, agentName," \
              " (H1 + H2 + H3 + H4) AS H," \
              " (E1 + E2 + E3 + E4) AS E," \
              " (X1 + X2 + X3 + X4) AS X," \
              " (A1 + A2 + A3 + A4) AS A," \
              " (C1 + C2 + C3 + C4) AS C," \
              " (O1 + O2 + O3 + O4) AS O," \
              " initialMining AS mining," \
              " initialAppraisal AS appraisal" \
              " FROM interactionDB.Agent) AS k" \
              " ON k.agentID = t.agentID" \
              " GROUP BY agentID, round, interactionType, isProactive) AS j" \
              " GROUP BY round, interactionType, isProactive, H, E, X, A, C, O, mining, appraisal) AS cool	" \
              " GROUP BY interactionType, isProactive"

        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()
        labels = []
        values = []
        friendship_total = 0
        for r in result:
            iT, iP, t = r
            if iT == "Friendship":
                friendship_total += float(t)
            else:
                label = iT + "/" + ("P" if iP else "R")
                labels.append(label)
                values.append(float(t))
        labels.append("Friendship")
        values.append(friendship_total)
        self.close()
        return labels, values

    def get_x_vs_interaction_count(self, agent_variable, agent_name, interaction_type, is_proactive, should_close=True, should_open=True):
        if should_open and not self.is_connected:
            self.connect_to_db()
        temp = "pro" if is_proactive else "re"
        sql = " SELECT "+agent_variable+", round(AVG(tot)) " \
              " FROM	(SELECT k.*, COUNT(agentID) AS tot " \
              " FROM interactionDB.Interactions " \
              " LEFT JOIN (SELECT agentID, agentName, " \
              " (H1 + H2 + H3 + H4) AS PH, " \
              " (E1 + E2 + E3 + E4) AS PE, " \
              " (X1 + X2 + X3 + X4) AS PX, " \
              " (A1 + A2 + A3 + A4) AS PA, " \
              " (C1 + C2 + C3 + C4) AS PC, " \
              " (O1 + O2 + O3 + O4) AS PO, " \
              " initialMining AS CM, " \
              " initialAppraisal AS CA " \
              " FROM interactionDB.Agent) AS k " \
              " ON " + temp + "activeAgentID = agentID " \
              " WHERE agentName = \"" + agent_name + "\" and interactionType = \"" + interaction_type + "\" " \
              " GROUP BY agentID) AS t" \
              " GROUP BY " + agent_variable

        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()
        x, y = [], []
        for r in result:
            a, b = r
            x.append(float(a))
            y.append(float(b))

        if should_close:
            self.close()
        return x, y

    def get_x_vs_interaction_choices(self, agent_variable, agent_name):

        friendship_reactive = self.get_x_vs_interaction_count(agent_variable,agent_name, "Friendship", False, False, True)
        friendship_proactive = self.get_x_vs_interaction_count(agent_variable, agent_name, "Friendship", True, False, False)

        a, a2 = friendship_reactive[1], friendship_proactive[1]

        na = []
        for i in range(len(a)):
            f = a[i] if i < len(a) else 0
            s = a2[i] if i < len(a2) else 0
            na.append(f + s)

        friendship_total = friendship_reactive[0], na

        mentorship_proactive = self.get_x_vs_interaction_count(agent_variable, agent_name, "Mentorship", True, False,
                                                               False)
        mentorship_reactive = self.get_x_vs_interaction_count(agent_variable, agent_name, "Mentorship", False, False,
                                                               False)

        theft_proactive = self.get_x_vs_interaction_count(agent_variable, agent_name, "Theft", True, False, False)
        theft_reactive = self.get_x_vs_interaction_count(agent_variable, agent_name, "Theft", False, False, False)

        help_proactive = self.get_x_vs_interaction_count(agent_variable, agent_name, "Help", True, False, False)
        help_reactive = self.get_x_vs_interaction_count(agent_variable, agent_name, "Help", False, True, True)

        return (friendship_total, mentorship_proactive, mentorship_reactive, theft_proactive, theft_reactive,
                help_proactive, help_reactive)



























