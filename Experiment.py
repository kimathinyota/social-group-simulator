from src.Agents.LearningAgent import *
from numpy.polynomial.polynomial import polyfit
import scipy.stats
import json
import csv
from src.DataAnalysis import *
from src.SocialAnalysis.SocialAnalysis import *
from src.Display import SocialGroupGUI
from src.Helper import *
import os


class RunningSimulation:

    @staticmethod
    def run_agent(agent, environment):
        agent.set_environment(environment)
        agent.run()

    @staticmethod
    def add_agent(agent, environment):
        t1 = threading.Thread(target=RunningSimulation.run_agent, args=(agent, environment))
        t1.start()

    @staticmethod
    def setup_environment(environment, agents):
        for agent in agents:
            RunningSimulation.add_agent(agent, environment)

    @staticmethod
    def run_environment(environment, agents):
        RunningSimulation.setup_environment(environment, agents)
        return environment.run()

    @staticmethod
    def display_gui(gui_requests, display_pause = None):
        gui = SocialGroupGUI(1400, 800)
        for request in gui_requests:
            ServiceGUI.process_request(gui, request)
            if display_pause is not None:
                gui.display(display_pause, 30)

    @staticmethod
    def upload_to_database(analysis):
        analysis.query_all()

    # check = "/Users/faithnyota1/Computer Science/3rd Year/Individual Project/Analysis/training/training.json"
    @staticmethod
    def simulate(agents, should_display, number_of_rounds=40, should_upload=True, should_test=False,
                 starting_wealth=10000, minimum_mining_amount=10, should_social_analyse=False,
                 training_save_location="/Users/faithnyota1/Computer Science/3rd Year/Individual Project/Analysis/training/training.json", display_pause=None):
        environment = ResourceMiningEnvironment(starting_wealth,minimum_mining_amount,number_of_rounds,should_test)
        gui_requests, analysis, hierarchy, training, rinteracts, ram = RunningSimulation.run_environment(environment,agents)

        if should_upload:
            RunningSimulation.upload_to_database(analysis)

        if should_display:
            RunningSimulation.display_gui(gui_requests, display_pause)

        social_analysis = False
        if should_social_analyse:
            social_analysis = SocialAnalysisResult(rinteracts, hierarchy, ram,
                                                   [agent.generation_id for agent in agents])

        return gui_requests, hierarchy, training, social_analysis


class Experiment:

    # Variables:
    #     Agent
    #         Personality
    #             PH: Honesty-Humility
    #             PE: Emotionality
    #             PX: Extraversion
    #             PA: Agreeableness
    #             PC: Conscientiousness
    #             PO: Openeness
    #         Competence
    #             CM: Mining
    #             CA: Appraisal

    variable_to_name = {
        "PH": "Honesty-Humility",
        "PE": "Emotionality",
        "PX": "Extraversion",
        "PA": "Agreeableness",
        "PC": "Conscientiousness",
        "PO": "Openeness",
        "CM": "Mining",
        "CA": "Appraisal"
    }

    @staticmethod
    def is_variable(variable):
        return variable in list(Experiment.variable_to_name.keys())

    @staticmethod
    def is_agent_variable(variable):
        f = variable[0]
        return Experiment.is_variable(variable) and (f=='P' or f=='C')

    @staticmethod
    def change_agent_value(agent, agent_variable, percentage):
        if not Experiment.is_agent_variable(agent_variable):
            return agent

        f, s = agent_variable[0], agent_variable[1]
        if f == 'P':
            val = percentage * agent.personality_template.max_facet_score
            agent.personality_template.set_all_facets_in_dimension(agent.personality, s, val)
        elif agent_variable == "CM":
            agent.competency.update(percentage, agent.competency.appraisal_skill)
        elif agent_variable == "CA":
            agent.competency.update(agent.competency.mining_skill, percentage)

        return agent

    @staticmethod
    def unique_colours(n):
        colours = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "000000",
        "#800000", "#008000", "#000080", "#808000", "#800080", "#008080", "#808080",
        "#C00000", "#00C000", "#0000C0", "#C0C000", "#C000C0", "#00C0C0", "#C0C0C0",
        "#400000", "#004000", "#000040", "#404000", "#400040", "#004040", "#404040",
        "#200000", "#002000", "#000020", "#202000", "#200020", "#002020", "#202020",
        "#600000", "#006000", "#000060", "#606000", "#600060", "#006060", "#606060",
        "#A00000", "#00A000", "#0000A0", "#A0A000", "#A000A0", "#00A0A0", "#A0A0A0",
        "#E00000", "#00E000", "#0000E0", "#E0E000", "#E000E0", "#00E0E0", "#E0E0E0"]
        return colours[0:n]

    @staticmethod
    def plot_3d(X, Y, Z, xlabel="X", ylabel="Y", zlabel="Z", is_x_discrete=True, should_include_lob=True, should_include_corr=True, show=True, should_draw=True):
        title = xlabel + " vs " + ylabel + " vs " + zlabel
        x_to_y_z = {}
        for i in range(len(X)):
            x = X[i]
            res = x_to_y_z[x] if x in x_to_y_z else [[], []]
            res[0].append(Y[i])
            res[1].append(Z[i])
            x_to_y_z[x] = res

        unique = list(x_to_y_z.keys())
        colours = Experiment.unique_colours(len(unique))
        x_to_colour = {unique[i]: colours[i] for i in range(len(unique))}

        if is_x_discrete:
            x_to_position = {unique[i - 1]: i for i in range(1, len(unique) + 1)}
        else:
            x_to_position = {x:x for x in unique}

        if should_draw:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            ax.set_title(title)

        x_to_correlation = {}

        for x in x_to_y_z:
            Y, Z, = x_to_y_z[x][0], x_to_y_z[x][1]
            p, color = x_to_position[x], x_to_colour[x]
            X = [p for i in range(len(Y))]

            should_include_lob = len(Y) >= 2 and len(Z) >= 2 and should_include_lob

            if should_draw:
                if should_include_lob:
                    ax.scatter(X, Y, Z,color=color)
                else:
                    ax.scatter(X, Y, Z, color=color, label=x)

            if should_include_lob:
                X, Y, Z = tuple(np.array(i) for i in [X, Y, Z])
                Xn,Yn, Zn = X.copy(), Y.copy(), Z.copy()
                if should_draw:
                    b, m = polyfit(Y, Z, 1)
                    Z = b + m * Y
                label = str(x)
                if should_include_corr:
                    r, rs, ps, pr = Experiment.get_correlation_data(Yn, Zn)
                    label += " " + str((rs,r,ps,pr))
                    x_to_correlation[x] = r, rs, ps, pr
                if should_draw:
                    ax.plot(X, Y, Z, color=color, label=label)

        if should_draw and ((not is_x_discrete) or should_include_corr):
            plt.legend(loc='upper left', title=xlabel + " ($r_s$, r, $p_s$, $p_r$)", numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
        if should_draw and show:
            plt.show()
        return x_to_correlation

    @staticmethod
    def get_correlation_data(X, Y):
        r, pr = scipy.stats.pearsonr(X, Y)
        rs, ps = scipy.stats.spearmanr(X, Y)
        rnd = 4
        rs, r = round(rs, rnd), round(r, rnd)
        ps, pr = round(ps, rnd), round(pr, rnd)
        return r, rs, ps, pr

    @staticmethod
    def plot_2d(X, Y, xlabel="X", ylabel="Y", show=True, should_draw=True):
        title = xlabel + " vs " + ylabel
        if should_draw:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        X, Y = tuple(np.array(i) for i in [X, Y])
        Xn, Yn = X.copy(), Y.copy()
        r, rs, ps, pr = Experiment.get_correlation_data(Xn, Yn)
        if should_draw:
            ax.scatter(X, Y)
            b, m = polyfit(X,Y,1)
            Y = b + m * X
            ax.plot(X, Y)
            txt = "$r_s$ = " + str(rs) + " (p=" + str(ps) + ")" + ", r = " + str(r) + " (p=" + str(pr) + ")"
            fig.text(0.01, 0.99, txt, ha='left', va='top',fontsize="large")
            if show:
                plt.show()
        return r, rs, ps, pr

    @staticmethod
    def plot_bar_chart(X, Y, xlabel="X", ylabel="Y",show=True, should_draw=True):
        title = xlabel + " vs " + ylabel
        if should_draw:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.bar(X, Y)
        if should_draw and show:
            plt.show()
        return fig

    @staticmethod
    def create_directory(dirName):
        try:
            # Create target Directory
            os.mkdir(dirName)
        except FileExistsError:
            pass

    # Training:
    #   id -> Q
    # Agents:
    #   id -> {Name, Starting Competency, Personality}
    # Config:
    #   (discount, gamma, alpha, q_array_sizes, number_trained_per_game, long_term_reward),
    #   (number_of_rounds, starting_wealth, minimum_mining_amount, number_of_agents),
    #   Training-order
    # Progress:
    #   Run: {#N.O, Current Position,  Agents IDs involved,
    #           training-params(discount, gamma, alpha, q_array_sizes, number_trained_per_game, long_term_reward),
    #           environment-params(number_of_rounds, starting-wealth,
    #                               minimum_mining_amount, number_of_agents)}
    @staticmethod
    def set_up_training_file(training_directory, q_array_sizes=[10, 10, 8, 5, 9], discount=0.85, gamma=0.9, alpha=0.9,
                             starting_wealth=10000, minimum_mining_amount=10, number_of_rounds=40,
                             number_trained_per_game=6, long_term_reward=5, agent_number_per_game=12):
        ri = [(['X'],70), (['C'],30), (['X4', 'O2', 'X3'],80), (['C1', 'C4'],20)]
        tw = [(['A'],70), (['H4', 'H2', 'A2', 'O2'], 70), (['X2'],25), (['X3'],90)]
        co = [(['E'],30), (['H3', 'H2', 'E3'], 25), (['E2'],10), (['C4'],80)]
        p  = [(['C'],30), (['O'],90)]
        me = [(['A'],30), (['O'],75), (['X3'], 30), (['X4'], 25), (['O2', 'H2'],75)]
        s  = [(['A'],20), (['X'],80), (['X4', 'X2'],90), (['C2'],70), (['C4'],30)]
        i  = [(['C'],80), (['O'],30), (['X2'],70)]
        cf = [(['E'],20), (['C'],80), (['E2'],10), (['C3'],90)]
        pmap = {'RI': ri, 'TW': tw, 'CO': co, 'P':p, 'ME': me, 'S':s, 'I':i, 'CF': cf}
        cmap = {'RI': ('?', '?'), 'TW': ('?', '?'), 'CO': ([(0.8,'H')], '?'), 'P': ([(0.8,'H')], '?'), 'ME': ('?', '?'), 'S': ('?', '?'),
                'I': ('?', '?'), 'CF': ('?', '?')}
        amap = {}
        id_to_q = {}

        template = HexacoPersonality()
        for d in pmap:
            aq, mq = cmap[d]
            possiblems = [(0.2,'L'), (0.5,'M'), (0.8,'H')]
            if mq != '?':
                possiblems = mq
            for mp in possiblems:
                possibleas = [(0.3,'L'), (0.7,'H')]
                if aq != '?':
                    possibleas = aq
                for ap in possibleas:
                    person = pmap[d]
                    m, mc = mp
                    a, ac = ap
                    id = d + "*" + mc + ac
                    comp = (m, a)
                    amap[id] = (id, comp, template.set_values(template.average(),person))
                    q = np.array(np.zeros(q_array_sizes))
                    id_to_q[id] = q.tolist()

        ids = list(amap.keys())
        random.shuffle(ids)

        config = ((discount, gamma, alpha, q_array_sizes, number_trained_per_game, long_term_reward),
                  (number_of_rounds, starting_wealth, minimum_mining_amount, agent_number_per_game),
                  ids)

        progress = []

        t = time.time()

        with open(training_directory + "/Progress.json", 'w') as fp:
            json.dump(progress, fp)

        with open(training_directory + "/Training.json", 'w') as fp:
            json.dump(id_to_q, fp)

        with open(training_directory + "/config.json", 'w') as fp:
            json.dump(config, fp)

        with open(training_directory + "/Agents.json", 'w') as fp:
            json.dump(amap, fp)

        print("Elapsed: ", time.time()-t)

    @staticmethod
    def get_training_agents(training_directory, learningAgentIDs, staticAgentIDs):
        agent_info = Experiment.get_json(training_directory + "/Agents.json")
        training_data = Experiment.get_json(training_directory + "/Training.json")
        config = Experiment.get_json(training_directory + "/config.json")
        if agent_info is None or config is None or training_data is None:
            return None
        agents = []
        tparams, eparams, torder = config
        d, g, a, qs, npg, ltr = tparams
        id_to_count = {}
        for id in learningAgentIDs:
            id_to_count[id] = 1 + id_to_count[id] if id in id_to_count else 1
            count = id_to_count[id]
            id_str = id + "*" + str(count) if count > 1 else id
            Q = np.array(training_data[id])
            name, competency, personality = agent_info[id]
            m, a = competency
            agent = LearningAgent(name, Competency(m, a), personality, id, False, ltr, qs, Q, d, g, a)
            agents.append(agent)

        for id in staticAgentIDs:
            id_to_count[id] = 1 + id_to_count[id] if id in id_to_count else 1
            count = id_to_count[id]
            id_str = id + "*" + str(count) if count > 1 else id
            name, competency, personality = agent_info[id]
            m, a = competency
            agent = Agent(name, Competency(m,a), personality, id)
            agents.append(agent)

        return agents, eparams

    @staticmethod
    def simulate_with_training_data(training_directory, learningAgentIDs, staticAgentIDs, should_display, number_of_rounds=40, should_upload=True,
                                    should_test=False, starting_wealth=10000, minimum_mining_amount=10):
        agents, eparams = Experiment.get_training_agents(training_directory, learningAgentIDs, staticAgentIDs)
        print("Running simulation on agents trained with environment: ", eparams)
        return RunningSimulation.simulate(agents, should_display, number_of_rounds, should_upload, should_test,
                                          starting_wealth, minimum_mining_amount)

    @staticmethod
    def testing_training_environment():
        training_directory = "/Users/faithnyota1/Computer Science/3rd Year/Individual Project/Analysis/training"
        # pmap = {'RI': ri, 'TW': tw, 'CO': co, 'P':p, 'ME': me, 'S':s, 'I':i, 'CF': cf}
        # {'RI': ('?', '?'), 'TW': ('?', '?'), 'CO': ([(0.8,'H')], '?'), 'P': ([(0.8,'H')], '?'), 'ME': ('?', '?'), 'S': ('?', '?'),
        #                 'I': ('?', '?'), 'CF': ('?', '?')}
        lids = ["RI*LL"]
        sids = ["TW*LL", "P*LH", "ME*LL", "S*LL", "CF*LL"]

        gui_requests, hierarchy, training, rinteracts = Experiment.simulate_with_training_data(training_directory,
                                                                                               lids, sids, False)
        social_structures = SocialAnalysis.find_social_structures(rinteracts)
        print(social_structures)

        interactions = []
        for round in rinteracts:
            interactions += rinteracts[round]

        print(SocialAnalysis.power_distribution(interactions))

    @staticmethod
    def resume_training(training_directory):
        config = Experiment.get_json(training_directory + "/config.json")
        agent_info = Experiment.get_json(training_directory + "/Agents.json")
        progress = Experiment.get_json(training_directory + "/Progress.json")
        training_data = Experiment.get_json(training_directory + "/Training.json")
        if config is None or agent_info is None or progress is None or training_data is None:
            return None

        tparams, eparams, torder = config
        d, g, a, qs, npg, ltr = tparams
        nr, sw, mma, na = eparams

        last = progress[len(progress)-1] if len(progress) > 0 else None
        cp = last[1] if last is not None else -1

        count = last[0] if last is not None else 0

        success, failures = 0, 0

        while count < 6000:
            try:
                cp += 1
                count += 1

                num_train = min(na, npg, len(torder))
                num_static = na - num_train

                to_train = []
                aids = torder.copy()
                for i in range(cp, cp + num_train):
                    a = torder[i % len(torder)]
                    to_train.append(a)
                    aids.remove(a)

                cp = (cp + num_train) % len(torder)

                static = []
                for i in range(num_static):
                    r = aids[random.randrange(len(aids))]
                    static.append(r)
                    aids.remove(r)

                agents = []
                for id in to_train:
                    Q = np.array(training_data[id])
                    name, competency, personality = agent_info[id]
                    m, a = competency
                    agent = LearningAgent(name, Competency(m,a), personality, id, True, ltr, qs, Q, d, g, a)
                    agents.append(agent)

                for id in static:
                    name, competency, personality = agent_info[id]
                    m, a = competency
                    agent = Agent(name, Competency(m,a), personality, id)
                    agents.append(agent)

                gui_requests, hierarchy, training, socialAnalysis = RunningSimulation.simulate(agents, False, nr, False)
                data = training['training']

                for id in data:
                    training_data[id] = data[id]

                entry = (count, cp, [agent.generation_id for agent in agents], tparams, eparams)
                progress.append(entry)

                with open(training_directory + "/Progress.json", 'w') as fp:
                    json.dump(progress, fp)

                with open(training_directory + "/Training.json", 'w') as fp:
                    json.dump(training_data, fp)

                success += 1
            except Exception as e:
                print(e)
                failures += 1
                print("Something went wrong - this run failed")

            print("Success count: ", success, " Failure count: ", failures)

    @staticmethod
    def set_up_main_results_files(experiment_folder):

        social_metrics_extra = ",".join([a + " " + b for a in ["Anti-Social", "Productivity", "Cooperation",
                                                               "Friendliness", "Social-Synergy", "Wealth-Stability"]
                                                          for b in ["Mean", "Similarity"]])

        power_stability_combo = ",".join([(x + " " + y)
                                          for y in ["Democracy", "Dictatorship", "RulingClass", "ServantClass",
                                                    "Slavery", "None"]
                                          for x in ["Stability-Mean", "Stability-Similarity",
                                                    "Strength-Mean", "Strength-Similarity",
                                                    "Total-Count"]])
        power_stability_combo = "Overall Stability-Mean, Overall Stability-Similarity, " + power_stability_combo
        power_metric_combo = ",".join([(x + " " + y)
                                       for y in ["Democracy", "Dictatorship", "RulingClass", "ServantClass",
                                                 "Slavery", "None"]
                                       for x in ["Stability Percentage (AVG)", "Strength (AVG)"]])

        power_metric_agent = ",".join([(x + " " + y)
                                       for y in ["Democracy", "Dictatorship", "RulingClass", "ServantClass",
                                                 "Slavery", "None"]
                                       for x in ["PowerStability", "PowerMetric"]])

        social_structures_title = "Combination, Total Consistent, Consistency-Score, Total Found, Consistent Social-Structures, All Social Structures"
        group_power_totals = ",".join(["Total " + a for a in ["Democracy", "Dictatorship", "RulingClass",
                                                              "ServantClass", "Slavery", "None"]])
        group_power_totals = "Stability-Mean, Stability-Similarity, " + group_power_totals

        agent_file_header = "Agent, N.O. Combinations, Total Social Structures, " \
                            "Total Hierarchy Value," + power_metric_agent

        agent_file2_header = "Agent, N.O. Combinations, Total Consistent Groups, Total Consistent Score, Total Groups Found," \
                             + social_metrics_extra + ",Power Stability Mean, Power Stability Similarity "

        combination_file_names = [("Combination," + social_metrics_extra, "CombinationsSocialMetrics.csv"),
                                  ("Combination," + power_stability_combo, "CombinationsPowerStability.csv"),
                                  ("Combination," + power_metric_combo , "CombinationsPowerMetric.csv"),
                                  (social_structures_title, "CombinationsSocialStructures.csv"),
                                  ("Combination," + social_metrics_extra, "CombinationsSocialGroupMetrics.csv"),
                                  ("Combination," + group_power_totals, "CombinationsGroupPowerStability.csv"),
                                  ("Combination," + social_metrics_extra, "CombinationsCompetingGroupMetrics.csv"),
                                  (agent_file_header, "AgentsAnalysis.csv"),
                                  (agent_file2_header, "AgentsAnalysis2.csv")]

        for (info, file_name) in combination_file_names:
            path = experiment_folder + "/" + file_name
            if os.path.exists(path):
                os.remove(path)
            Experiment.update_file(path, info + "\n")

    @staticmethod
    def list_csv_entry(lst):
        return "[" + " ".join(lst) + "]"

    @staticmethod
    def social_group_csv_entry(list_of_groups):
        str_lst = [Experiment.list_csv_entry(lst) for lst in list_of_groups]
        return Experiment.list_csv_entry(str_lst)

    @staticmethod
    def social_group_map_csv_entry(group_to_score):
        lst = [ Experiment.list_csv_entry(g) + ":" + str(group_to_score[g]) for g in group_to_score]
        return Experiment.list_csv_entry(lst)

    @staticmethod
    def process_and_save_main_results(social_analysis_result_data_sets, experiment_folder):

        agent_to_social_count = {}
        agent_to_hierarchy_value = {}
        agent_to_power_type_stab = {}
        agent_to_power_type_metr = {}
        agent_to_average_social_metric = {}
        agent_to_power_stability = {}
        agent_to_wealth_stability = {}
        agent_to_social_structure_info = {}
        agent_to_count = {}
        agents = []

        for analysis_data in social_analysis_result_data_sets:
            data_set, combo, cp = analysis_data

            data = SocialAnalysisResult.merge(data_set)
            sm = data["SocialMetrics"]
            combination = " ".join(sorted(combo))
            ws = data["WealthStability"]

            sm_entry = combination + "," + ",".join([str(v) for tup in sm for v in tup]) + "," + str(ws[0]) + "," + str(ws[1])
            ps = data["PowerStability"]
            overall, ps = ps
            ps_entry = ",".join([str(a) for key in ["Democracy", "Dictatorship", "RulingClass", "ServantClass",
                                                    "Slavery", "None"]
                                 for a in [p for i in range(2) for p in ps[key][i]] + [ps[key][3]]])

            o1, o2 = overall
            ps_entry = combination + "," + str(o1) + "," + str(o2) + "," + ps_entry
            for typ in ps:
                v = ps[typ][2]
                for a in v:
                    res = {t: 0 for t in ["Democracy", "Dictatorship", "RulingClass",
                                          "ServantClass", "Slavery", "None"]}
                    if a in agent_to_power_type_stab:
                        res = agent_to_power_type_stab[a]
                    res[typ] += v[a]
                    agent_to_power_type_stab[a] = res

            pm = data["PowerMetric"]
            pm_entry = ",".join([str(a) for key in ["Democracy", "Dictatorship", "RulingClass", "ServantClass",
                                                    "Slavery", "None"]
                                 for a in [pm[key][i] for i in range(2)]])

            for typ in pm:
                v = pm[typ][2]
                for a in v:
                    res = {t: 0 for t in ["Democracy", "Dictatorship", "RulingClass",
                                          "ServantClass", "Slavery", "None"]}
                    if a in agent_to_power_type_metr:
                        res = agent_to_power_type_metr[a]
                    res[typ] += v[a]
                    agent_to_power_type_metr[a] = res

            pm_entry = combination + "," + pm_entry

            sc = data["SocialStructures"]
            cons_groups, conist_score, all_groups = sc

            str_grps = Experiment.social_group_map_csv_entry(cons_groups)
            str_af = Experiment.social_group_map_csv_entry(all_groups)

            sc_entry = combination + "," + str(len(cons_groups)) + "," + str(conist_score) + "," + str(len(all_groups)) \
                       + "," + str_grps + "," + str_af

            sgm = data["SocialGroupMetrics"]
            sgm_entry = combination + "," + ",".join([str(v) for tup in sgm for v in tup])

            gps = data["GroupPowerStability"]
            stab, ttv = gps
            m, sim = stab
            gps_entry = str(m) + "," + str(sim) + "," + ",".join([str(ttv[a]) for a in ["Democracy", "Dictatorship",
                                                                                        "RulingClass", "ServantClass",
                                                                                        "Slavery", "None"]])
            gps_entry = combination + "," + gps_entry

            cgm = data["CompetingGroupMetrics"]
            cgm_entry = combination + "," + ",".join([str(v) for tup in cgm for v in tup])

            athp = data["AgentToHierarchyPosition"]

            for a in athp:
                agent_to_hierarchy_value[a] = athp[a] if a not in agent_to_hierarchy_value else athp[a] + agent_to_hierarchy_value[a]

            combo = list(set(athp.keys()))
            atgc = data["AgentToGroupCount"]
            for a in atgc:
                agent_to_social_count[a] = atgc[a] if a not in agent_to_social_count else atgc[a] + agent_to_social_count[a]

            combo = list(set(combo + list(atgc.keys())))

            info_file_names = [(sm_entry, "CombinationsSocialMetrics.csv"),
                               (ps_entry, "CombinationsPowerStability.csv"),
                               (pm_entry, "CombinationsPowerMetric.csv"),
                               (sc_entry, "CombinationsSocialStructures.csv"),
                               (sgm_entry, "CombinationsSocialGroupMetrics.csv"),
                               (gps_entry, "CombinationsGroupPowerStability.csv"),
                               (cgm_entry, "CombinationsCompetingGroupMetrics.csv")]

            agents = list(set(agents + combo))
            for agent in combo:
                res = [ ([],[]), ([],[]), ([],[]), ([],[]), ([],[])]
                if agent in agent_to_average_social_metric:
                    res = agent_to_average_social_metric[agent]
                for i in range(5):
                    m, s = sm[i]
                    m1, s1 = res[i]
                    res[i] = (m1 + [m], s1 + [s])
                agent_to_average_social_metric[agent] = res
                aps = ([], [])
                if agent in agent_to_power_stability:
                    aps = agent_to_power_stability[agent]
                m, s = overall
                m1, s1 = aps
                aps = (m1 + [m], s1 + [s])
                agent_to_power_stability[agent] = aps
                aws = ([], [])
                if agent in agent_to_wealth_stability:
                    aws = agent_to_wealth_stability[agent]
                m, s = ws
                m1, s1 = aws
                aws = (m1 + [m], s1 + [s])
                agent_to_wealth_stability[agent] = aws

                entry = (0, 0, 0)
                if agent in agent_to_social_structure_info:
                    entry = agent_to_social_structure_info[agent]
                entry = (entry[0] + len(cons_groups), entry[1] + conist_score, entry[2] + len(all_groups))
                agent_to_social_structure_info[agent] = entry

                agent_to_count[agent] = agent_to_count[agent] + 1 if agent in agent_to_count else 1

            for (info, file_name) in info_file_names:
                path = experiment_folder + "/" + file_name
                Experiment.update_file(path, info + "\n")

        for agent in agents:
            ttm = {t: 0 for t in ["Democracy", "Dictatorship", "RulingClass", "ServantClass", "Slavery", "None"]}
            if agent in agent_to_power_type_metr:
                ttm = agent_to_power_type_metr[agent]
            tts = {t: 0 for t in ["Democracy", "Dictatorship", "RulingClass", "ServantClass", "Slavery", "None"]}
            if agent in agent_to_power_type_stab:
                tts = agent_to_power_type_stab[agent]
            hp = agent_to_hierarchy_value[agent] if agent in agent_to_hierarchy_value else 0
            ss = agent_to_social_count[agent] if agent in agent_to_social_count else 0

            powstab, wealthstab = (0, 1), (0, 1)
            if agent in agent_to_power_stability:
                a, b = agent_to_power_stability[agent]
                if len(a) >= 2 and len(b) >= 2:
                    powstab = (SocialAnalysis.get_mean_similarity(a)[0], SocialAnalysis.get_mean_similarity(b)[0])

            if agent in agent_to_wealth_stability:
                a, b = agent_to_wealth_stability[agent]
                if len(a) >= 2 and len(b) >= 2:
                    wealthstab = (SocialAnalysis.get_mean_similarity(a)[0], SocialAnalysis.get_mean_similarity(b)[0])

            avsm = [(0, 1), (0, 1), (0, 1), (0, 1)]
            if agent in agent_to_average_social_metric:
                vs = agent_to_average_social_metric[agent]
                avsm = []

                for a, b in vs:
                    if len(a) >= 2 and len(b) >= 2:
                        a, b = (SocialAnalysis.get_mean_similarity(a)[0], SocialAnalysis.get_mean_similarity(b)[0])
                    else:
                        a, b = 0, 1
                    avsm.append((a,b))
            count = agent_to_count[agent]
            ss_info = agent_to_social_structure_info[agent] if agent in agent_to_social_structure_info else (0, 0, 0)

            ss_entry = ",".join([str(t/count) for t in ss_info])

            info2 = agent + "," + str(count) + "," + ss_entry + "," + ",".join([str(v) for tup in avsm + [wealthstab, powstab] for v in tup])

            power_metric_agent = ",".join([str(x/count)
                                           for key in ["Democracy", "Dictatorship", "RulingClass", "ServantClass",
                                                       "Slavery", "None"]
                                           for x in [tts[key], ttm[key]]])

            a_entry = agent + "," + str(count) + "," + str(ss/count) + "," + str(hp/count) + "," + power_metric_agent
            path = experiment_folder + "/" + "AgentsAnalysis.csv"
            Experiment.update_file(path, a_entry + "\n")
            path2 = experiment_folder + "/" + "AgentsAnalysis2.csv"
            Experiment.update_file(path2, info2 + "\n")

    @staticmethod
    def process_main_experiment(results, to_store_data_folder):
        if len(results) == 0:
            return None
        Experiment.set_up_main_results_files(to_store_data_folder)
        Experiment.process_and_save_main_results(results,
                                                 to_store_data_folder)

    @staticmethod
    def process_main_experiment_results(experiment_directory, training_results_folder, static_results_folder,
                                        should_static=True, should_train=True):
        config = Experiment.get_json(experiment_directory + "/config.json")
        static_results = Experiment.get_json(experiment_directory + "/static_experiment_results.json")
        training_results = Experiment.get_json(experiment_directory + "/training_experiment_results.json")

        if config is None or static_results is None or training_results is None:
            return None

        combinations, eparams = config

        # Processing Training Experiment
        if should_train:
            Experiment.process_main_experiment(training_results, training_results_folder)

        # Processing Main Experiment
        if should_static:
            Experiment.process_main_experiment(static_results, static_results_folder)

    @staticmethod
    def spread_out_combinations(number):
        pmap = ['RI', 'TW', 'CO', 'P', 'ME', 'S', 'I', 'CF']

        ah_roles = ['CO', 'P']
        mt, at = ['L', 'M', 'H'], ['L', 'H']
        mine = {a: [0, 0, 0] for a in pmap}
        appr = {a: [0, 0] for a in pmap}

        extra_agents = {a: 0 for a in pmap}

        runs = []
        for i in range(number):
            extra = list(sorted(extra_agents.keys(), key=lambda key: extra_agents[key]))[0:4]
            for e in extra:
                extra_agents[e] += 1
            roles = pmap + extra

            random.shuffle(roles)

            order = [0, 1, 2]
            random.shuffle(order)

            new_roles = roles.copy()
            r_to_m = []
            for o in order:
                m = {a: mine[a] for a in new_roles}
                mp = {}
                repeats = {}
                for i in range(len(new_roles)):
                    a = new_roles[i]
                    scr = repeats[a] if a in repeats else 0
                    repeats[a] = scr + 1
                    mp[(a, i)] = mine[a][o] + scr

                chosen = list(sorted(mp.keys(), key=lambda key: mp[key]))[0:4]
                chosen = [a for (a, b) in chosen]

                for c in chosen:
                    mine[c][o] += 1
                    r_to_m.append((c, mt[o]))
                    new_roles.remove(c)

            hs = ah_roles + [e for e in extra if e in ah_roles]

            new_roles = roles.copy()
            order = [0, 1]
            random.shuffle(order)
            r_to_a = []

            for c in hs:
                appr[c][1] += 1
                r_to_a.append((c, 'H'))
                new_roles.remove(c)
            ap = {}
            repeats = {}
            for i in range(len(new_roles)):
                a = new_roles[i]
                scr = repeats[a] if a in repeats else 0
                repeats[a] = scr + 1
                ap[(a,i)] = appr[a][1] + scr

            # print("HS", hs)
            chosen = list(sorted(ap.keys(), key=lambda key: ap[key]))[0:(6 - len(hs))]

            #print(chosen)
            chosen = [a for (a,b) in chosen]
            # print("AHs", chosen, hs)

            for c in chosen:
                appr[c][1] += 1
                r_to_a.append((c, 'H'))
                new_roles.remove(c)

            for c in new_roles:
                appr[c][0] += 1
                r_to_a.append((c, 'L'))

            agents = []

            # print("M", r_to_m)
            # print("A", r_to_a)

            for a in r_to_m:
                n, m = a
                l = list([u for (u, v) in r_to_a])
                p = r_to_a[l.index(n)]
                agents.append((n, m, p[1]))
                r_to_a.remove(p)

            #print("Agents", agents)


            if not contains(runs, agents):
                runs.append(agents)

        return runs

    @staticmethod
    def count(runs):
        mdict = {'L': 0, 'M': 0, 'H': 0}
        adict = {'L': 0, 'H': 0}
        agdict = {}
        for run in runs:
            for entry in run:
                agent, m, a = entry
                t = agdict[agent] if agent in agdict else 0
                agdict[agent] = t + 1
                mdict[m] += 1
                adict[a] += 1
        return agdict, mdict, adict

    @staticmethod
    def set_up_main_experiment_combos(experiment_directory, combinations, starting_wealth=10000,
                                      minimum_mining_amount=10, number_of_rounds=40, agent_number_per_game=12):
        # Config: ([Runs], eparams)
        eparams = (number_of_rounds, starting_wealth, minimum_mining_amount, agent_number_per_game)
        a1, a2, a3 = Experiment.count(combinations)
        print("Agents", a1)
        print("MDict", a2)
        print("ADict", a3)
        config = (combinations, eparams)

        with open(experiment_directory + "/config.json", 'w') as fp:
            json.dump(config, fp)

        # static_results = [ (SocialAnalysisResults(x3), Run, Position) ]

        static_experiment_results, training_experiment_results = [], []
        with open(experiment_directory + "/static_experiment_results.json", 'w') as fp:
            json.dump(static_experiment_results, fp)
        with open(experiment_directory + "/training_experiment_results.json", 'w') as fp:
            json.dump(training_experiment_results, fp)

    @staticmethod
    def set_up_main_experiment(experiment_directory, number_of_different_combinations, starting_wealth=10000,
                               minimum_mining_amount=10, number_of_rounds=40, agent_number_per_game=12):
        combinations = Experiment.spread_out_combinations(number_of_different_combinations)
        Experiment.set_up_main_experiment_combos(experiment_directory, combinations, starting_wealth,
                                                 minimum_mining_amount, number_of_rounds, agent_number_per_game)

    @staticmethod
    def resume_main_experiment(experiment_directory, training_directory, should_static=True, should_training=True):
        if not (should_static or should_training):
            return None
        config = Experiment.get_json(experiment_directory + "/config.json")
        static_results = Experiment.get_json(experiment_directory + "/static_experiment_results.json")
        training_results = Experiment.get_json(experiment_directory + "/training_experiment_results.json")

        if config is None or static_results is None or training_results is None:
            return None

        combinations, eparams = config
        number_of_rounds, starting_wealth, minimum_mining_amount, agent_number_per_game = eparams

        max_number_of_runs = len(combinations)

        # For Experiment with trained agents:

        bools = [True, False]
        if not should_training:
            bools.remove(True)
        if not should_static:
            bools.remove(False)

        for is_training_experiment in bools:
            if is_training_experiment:
                current_position = 0 if len(training_results) == 0 else training_results[len(training_results) - 1][2] + 1
            else:
                current_position = 0 if len(static_results) == 0 else static_results[len(static_results) - 1][2] + 1

            print("Starting", "Training" if is_training_experiment else "Static", "Experiment")
            while current_position < max_number_of_runs:
                try:
                    run = combinations[current_position]
                    agentIDs = [(belbin + "*" + mine + appr) for (belbin, mine, appr) in run]

                    results = []

                    agent_gids = []
                    for i in range(3):
                        if is_training_experiment:
                            agents = Experiment.get_training_agents(training_directory, agentIDs, [])[0]

                        else:
                            agents = Experiment.get_training_agents(training_directory, [], agentIDs)[0]

                        agent_gids = list(set(agent_gids + [a.generation_id for a in agents]))
                        print("Running simulations ", str(i + 1) + "/3")
                        gui_requests, hierarchy, training, socialAnalysis = RunningSimulation.simulate(agents, False,
                                                                                                       number_of_rounds,
                                                                                                       False, False,
                                                                                                       starting_wealth,
                                                                                                       minimum_mining_amount,
                                                                                                       True)
                        results.append(socialAnalysis.get_data())

                    entry = (results, agent_gids, current_position)

                    if is_training_experiment:
                        training_results.append(entry)
                        with open(experiment_directory + "/training_experiment_results.json", 'w') as fp:
                            json.dump(training_results, fp)

                    else:
                        static_results.append(entry)
                        with open(experiment_directory + "/static_experiment_results.json", 'w') as fp:
                            json.dump(static_results, fp)

                    print("Training" if is_training_experiment else "Static", "Experiment", "Run", current_position,
                          "Success", str(max_number_of_runs - current_position) + " left")
                    current_position += 1
                except Exception as e:
                    print(e)
                    print("Training" if is_training_experiment else "Static", "Experiment", "Run", current_position,
                          "Something went wrong - this run failed")
            print("Finished", "Training" if is_training_experiment else "Static", "Experiment")

    @staticmethod
    def run_x_vs_earned_experiment(agent_var,total_number_of_agents, directory, starting=1, ending=9, number_of_repeats=3, number_of_rounds=40, starting_wealth=10000,
                                   minimum_mining_amount=10):
        if not Experiment.is_agent_variable(agent_var):
            print("Experiment couldn't start - input variable isn't an agent variable")
            return None
        analysis = Analysis()
        analysis.remove_all_data()
        other_agents = []
        for i in range(total_number_of_agents - 1):
            name = "AG" + str(i)
            other_agents.append(Agent.random(name))
        n = "testAgent"
        agent = Agent.random(n)
        xl = Experiment.variable_to_name[agent_var]
        total_runs = (ending - starting + 1) * number_of_repeats
        print("Running experiment for " + xl + " Total simulation runs: " + str(total_runs))
        count = 0

        for p in range(starting, ending+1):
            p /= 10
            experimental_agent = Experiment.change_agent_value(agent.copy(), agent_var, p)
            for i in range(number_of_repeats):
                experimental_agents = [agent.copy() for agent in ([experimental_agent] + other_agents)]
                RunningSimulation.simulate(experimental_agents, False, number_of_rounds,
                                           starting_wealth=starting_wealth, minimum_mining_amount=minimum_mining_amount)
                count += 1
                print("Finished " + xl + ": " + str(count) + "/" + str(total_runs) )
        # Data analysis
        # Get table: X | Round | Average-Wealth-Earnings

        title1 = xl + "vsGold"
        data1 = Experiment.construct_x_vs_gold_earned_json(agent_var,n,analysis)
        title2 = xl + "vsComp"
        data2 = Experiment.construct_x_vs_comp_earned_json(agent_var,n,analysis)
        title3 = xl + "vsInteractionChoices"
        data3 = Experiment.construct_x_vs_interaction_choice(agent_var,n,analysis)
        data = [(title1,data1), (title2, data2), (title3, data3)]

        variable_directory = directory + "/" + xl
        Experiment.create_directory(variable_directory)

        for d in data:
            t, d = d
            with open(variable_directory + "/" + t + '.json', 'w') as fp:
                json.dump(d, fp)

        print("Finished experiment for " + xl)

    @staticmethod
    def run_interaction_vs_earn(total_number_of_agents, directory, starting=1, ending=9, number_of_repeats=3, number_of_rounds=40, starting_wealth=10000,
                                   minimum_mining_amount=10):
        analysis = Analysis()
        analysis.remove_all_data()

        n = "testAgent"
        count = 0

        total_runs = (ending - starting + 1) * number_of_repeats
        print("Running Experiment for Interaction vs Earn: Total Simulations: " + str(total_runs))

        for p in range(starting, ending+1):
            other_agents = []
            for i in range(total_number_of_agents):
                name = "AG" + str(i)
                other_agents.append(Agent.random(name))

            for i in range(number_of_repeats):
                experimental_agents = [agent.copy() for agent in other_agents]
                RunningSimulation.simulate(experimental_agents, False, number_of_rounds,
                                           starting_wealth=starting_wealth, minimum_mining_amount=minimum_mining_amount)
                count += 1
                print(str(count) + "/" + str(total_runs))

        # Data analysis
        title1 = "InteractionVsGold"
        data1 = Experiment.construct_interaction_vs_gold_earned_json(analysis)

        with open(directory + "/" + title1 + '.json', 'w') as fp:
            json.dump(data1, fp)

        print("Finished Experiment for Interaction vs Earn")

    @staticmethod
    def construct_x_vs_gold_earned_json(agent_var, agent_name, analysis=Analysis()):
        xl = Experiment.variable_to_name[agent_var]
        x, y, z = analysis.get_x_vs_round_to_earn(agent_var, agent_name)
        yl, zl = "Round", "Gold Earned"
        d1 = {"data": (x, y, z), "labels": (xl, yl, zl)}
        x, y = analysis.get_x_vs_total_earn(agent_var, agent_name)
        yl = "Total Gold Earned"
        d2 = {"data": (x, y), "labels": (xl, yl)}
        data = {
            '3D': [d1],
            '2D': [d2]
        }
        return data

    @staticmethod
    def construct_x_vs_interaction_choice(agent_var, agent_name, analysis=Analysis()):
        xl = Experiment.variable_to_name[agent_var]
        f, mp, mr, tp, tr, hp, hr = analysis.get_x_vs_interaction_choices(agent_var, agent_name)

        # x --> {f, mp, mr, tp, tr, hp, hr}
        var_to_values = {}

        X, Y, Z = [], [], []

        x, y = f
        for i in range(len(x)):
            X.append("Friendship")
            Y.append(x[i])
            Z.append(y[i])

        d1 = {"data": (x, y), "labels": (xl, "Friendship Total")}
        x, y = mp
        for i in range(len(x)):
            X.append("Mentorship\P")
            Y.append(x[i])
            Z.append(y[i])
        d2 = {"data": (x, y), "labels": (xl, "# Mentorship Proactive")}
        x, y = mr
        for i in range(len(x)):
            X.append("Mentorship\R")
            Y.append(x[i])
            Z.append(y[i])
        d3 = {"data": (x, y), "labels": (xl, "# Mentorship Reactive")}
        x, y = tp
        for i in range(len(x)):
            X.append("Theft\P")
            Y.append(x[i])
            Z.append(y[i])
        d4 = {"data": (x, y), "labels": (xl, "# Theft Proactive")}
        x, y = tr
        for i in range(len(x)):
            X.append("Theft\R")
            Y.append(x[i])
            Z.append(y[i])
        d5 = {"data": (x, y), "labels": (xl, "# Theft Reactive")}
        x, y = hp
        for i in range(len(x)):
            X.append("Help\P")
            Y.append(x[i])
            Z.append(y[i])
        d6 = {"data": (x, y), "labels": (xl, "# Help Proactive")}
        x, y = hr
        for i in range(len(x)):
            X.append("Help\R")
            Y.append(x[i])
            Z.append(y[i])
        d7 = {"data": (x, y), "labels": (xl, "# Help Reactive")}

        lst = [d1, d2, d3, d4, d5, d6, d7]

        data = {
            '3DB': [{"data": (X,Y,Z), "labels":("Interaction Type",Experiment.variable_to_name[agent_var], "Number of interactions")}]
        }

        return data

    @staticmethod
    def construct_x_vs_comp_earned_json(agent_var, agent_name, analysis=Analysis()):
        xl = Experiment.variable_to_name[agent_var]
        x, y, z = analysis.get_x_vs_round_to_comp(agent_var, agent_name)
        yl, zl = "Round", "Competency Improved"
        d1 = {"data": (x, y, z), "labels": (xl, yl, zl)}
        x, y = analysis.get_x_vs_total_comp(agent_var, agent_name)
        yl = "Total Competency Increase"
        d2 = {"data": (x, y), "labels": (xl, yl)}
        data = {
            '3D': [d1],
            '2D': [d2]
        }
        return data

    @staticmethod
    def construct_interaction_vs_gold_earned_json(analysis=Analysis()):
        labels, x, y = analysis.get_interaction_vs_round_to_earn()
        d1 = {"data":(labels,x,y), "labels":("Interaction Type","Round","Gold Earned")}

        x, y = analysis.get_interaction_vs_total_earn()
        d2 = {"data": (x, y), "labels": ("Interaction Type", "Total Gold Earned")}

        x, y = analysis.get_interaction_to_total()
        d3 = {"data": (x, y), "labels": ("Interaction Type", "Total Number of Rounds")}

        data = {
            '3DB': [d1],
            '2DB': [d2, d3]
        }
        return data

    @staticmethod
    def estimated_time_seconds(number_of_rounds):
        s = 32
        return s * number_of_rounds/40

    @staticmethod
    def run_all_experiments(total_number_of_agents, agent_variable_experiment_directory,
                            interaction_experiment_directory, start=1, end=10, repeats=3,
                            agent_variables=["CM", "PC", "PX", "CA", "PE", "PO", "PH", "PA"], number_of_rounds=40,
                            starting_wealth=10000, minimum_mining_amount=10):

        experimental_variables = agent_variables

        secs_per_sim = Experiment.estimated_time_seconds(number_of_rounds)

        number_sims = (end - start + 1) * repeats

        total_sims = (len(experimental_variables) + 1) * number_sims

        total_time = round((total_sims * secs_per_sim) / 60, 2)

        print("Running all experiments ... Estimated time: " + str(total_time) + " mins")
        print("...")

        for variable in experimental_variables:
            Experiment.run_x_vs_earned_experiment(variable,total_number_of_agents, agent_variable_experiment_directory,
                                                  start,end,repeats, number_of_rounds,starting_wealth,
                                                  minimum_mining_amount)

        Experiment.run_interaction_vs_earn(total_number_of_agents, interaction_experiment_directory, start, end, repeats)

    @staticmethod
    def plot(agent_var, directory):
        n = "testAgent"
        analysis = Analysis()
        # Data analysis
        # Get table: X | Round | Average-Wealth-Earnings
        x, y, z = analysis.get_x_vs_round_to_earn(agent_var, n)
        xl, yl, zl = Experiment.variable_to_name[agent_var], "Round", "Gold Earned"
        d1 = {"data": (x, y, z), "labels": (xl, yl, zl)}
        x, y = analysis.get_x_vs_total_earn(agent_var, n)
        yl = "Total Gold Earned"
        d2 = {"data": (x, y), "labels": (xl, yl)}
        data = {
            '3D': [d1],
            '2D': [d2]
        }
        with open(directory + "/" + xl + '.json', 'w') as fp:
            json.dump(data, fp)

    @staticmethod
    def load_json(data,show=True, csv_string="", should_draw=True):

        d1 = data['3D'] if '3D' in data else []
        d2 = data['2D'] if '2D' in data else []
        d3 = data['2DB'] if '2DB' in data else []
        d4 = data['3DB'] if '3DB' in data else []

        if not isinstance(d1,list):
            d1 = [d1]
        if not isinstance(d2,list):
            d2 = [d2]
        if not isinstance(d3,list):
            d3 = [d3]
        if not isinstance(d4,list):
            d4 = [d4]

        for d in d1:
            x, y, z = d['data']
            xl, yl, zl = d['labels']
            Experiment.plot_3d(x, y, z, xl, yl, zl,False, True, True, False,should_draw)

        for d in d2:
            x, y = d['data']
            xl, yl = d['labels']
            r, rs, ps, pr = Experiment.plot_2d(x, y, xl, yl,False,should_draw)
            line = Experiment.construct_line(xl,yl, r, pr, rs, ps)
            csv_string += line

        for d in d3:
            x, y = d['data']
            xl, yl = d['labels']
            Experiment.plot_bar_chart(x,y, xl, yl,False, should_draw)

        for d in d4:
            x, y, z = d['data']
            xl, yl, zl = d['labels']
            x_to_correlation = Experiment.plot_3d(x, y, z, xl, yl, zl,True, True, True, False,should_draw)
            for x in x_to_correlation:
                r, rs, ps, pr = x_to_correlation[x]
                line = Experiment.construct_line(yl, x, r, pr, rs, ps)
                csv_string += line

        if show:
            plt.show()
        return csv_string

    @staticmethod
    def construct_line(variable1, variable2, r, pr, rs, ps):
        line = variable1 + "," + variable2 + "," + str(rs) + "," + str(ps) + "," + str(r) + "," + str(pr)
        line += "\n"
        return line

    @staticmethod
    def load_graph(path_to_json, show=True, csv_string="", should_draw=True):
        data = Experiment.get_json(path_to_json)
        if data is not None:
            csv_string += Experiment.load_json(data, False, csv_string, should_draw)
        if show:
            plt.show()
        return csv_string

    @staticmethod
    def get_json(path_to_json):
        if ".json" in path_to_json:
            with open(path_to_json, 'r') as fp:
                data = json.load(fp)
                return data
        return None

    @staticmethod
    def load_graphs(directory, show=True, csv_string="", should_draw=True):
        entries = os.listdir(directory+"/")
        for entry in entries:
            e = directory+"/" + str(entry)
            csv_string += Experiment.load_graph(e, False, should_draw=should_draw)
        if show:
            plt.show()
        return csv_string

    @staticmethod
    def update_file(file_path, information):
        if not os.path.exists(file_path):
            with open(file_path, 'w'): pass
        f = open(file_path, "a")
        f.write(information)
        f.close()

    @staticmethod
    def display():
        plt.show()

    @staticmethod
    def show_experiment_results(agent_variable_directory, interaction_directory, agent_variables_to_display, should_display_interaction_graphs, file_path, show=True):
        csv = "Variable1,Variable2,rs,ps,r,pr\n"
        for var in agent_variables_to_display:
            direc = agent_variable_directory + "/" + Experiment.variable_to_name[var]
            print(direc)
            csv += Experiment.load_graphs(direc, False, should_draw=show)
        Experiment.update_file(file_path,csv)
        if should_display_interaction_graphs:
            Experiment.load_graphs(interaction_directory,False, should_draw=show)

        if show:
            plt.show()

    @staticmethod
    def get_expected_values(expected_csv_file_location):
        data = {}
        with open(expected_csv_file_location) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    var1, var2, expected = row
                    data[(var1, var2)] = expected
                line_count += 1
            print(f'Processed {line_count} lines.')
        return data

    @staticmethod
    def get_actual_values(actual_csv_file_location):
        data = {}
        with open(actual_csv_file_location) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    var1, var2, rs, ps, r, p = row
                    data[(var1, var2)] = (rs, ps, r, p)
                line_count += 1
            print(f'Processed {line_count} lines.')
        return data

    @staticmethod
    def tofloat(x):
        f = float(x)
        f = 0 if math.isnan(f) else f
        return f

    @staticmethod
    def check_values(expected_csv_file_location, actual_csv_file_location, save_location, alpha=0.09, start_positive=0.2, start_negative=-0.2):
        expected_data = Experiment.get_expected_values(expected_csv_file_location)
        actual_data = Experiment.get_actual_values(actual_csv_file_location)

        Experiment.update_file(save_location, "Independent Variable, Dependent Variable, Spearman (rs), p, Pearson(r), "
                                              "p, Expected? \n")

        for variables in actual_data:
            var1, var2 = variables
            rsl, psl, rl, pl = actual_data[variables]
            expected = expected_data[variables] if variables in expected_data else '?'
            line = var1 + "," + var2 + "," + rsl + "," + psl + "," + rl + "," + pl + ","
            if expected != '?':
                rs, ps, r, p = Experiment.tofloat(rsl), Experiment.tofloat(psl), Experiment.tofloat(rl), Experiment.tofloat(pl)
                is_statistically_significant = ps <= alpha or p <= alpha
                extra = "SI" if not is_statistically_significant else ""

                matches = (expected == '1' and (rs >= start_positive or r >= start_positive)) \
                          or (expected == '-1' and (rs <= start_negative or r <= start_negative))

                result = "Y" if matches else "N"
                result += extra
                line += result
            line += "\n"
            Experiment.update_file(save_location, line)







































