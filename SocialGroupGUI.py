import operator
import random
from src.graphics import *
import math
import src.Interaction
import enum





class SocialGroupGUI:

    def __init__(self, width, height):
        self.win = GraphWin('Social Group', width, height, autoflush=False)
        x = 0.8 * width
        y = 0.7 * height

        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.agent_canvas = {}
        self.hierarchy_canvas = {}
        self.personality_canvas = []

        agent_view = Rectangle(Point(0, 0), Point(x, y))
        agent_view.setFill(color_rgb(143, 170, 220))
        agent_view.setWidth(2)
        agent_view.setOutline(color_rgb(32, 56, 100))
        agent_view.draw(self.win)


        self.resource_width = (x*2)/11


        #Resource view
        resource_view = Rectangle(Point(x - self.resource_width, 0), Point(x, y))
        resource_view.setFill(color_rgb(191, 144, 0))
        resource_view.setWidth(2)

        resource_view.draw(self.win)

        label = Text(Point(int(x - self.resource_width/2), int(y/2)), "Gold")
        label.setSize(23)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("bold")
        label.draw(self.win)




        hierarchy_view = Rectangle(Point(x, 0), Point(width, y))
        hierarchy_view.setFill(color_rgb(0, 176, 240))
        hierarchy_view.setWidth(2)
        hierarchy_view.setOutline(color_rgb(32, 56, 100))
        hierarchy_view.draw(self.win)

        label = Text(Point(int((x + width) / 2), int(0.02 * height + 10)), "  Agent Wealth  ")

        label.setSize(23)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("bold")
        label.draw(self.win)
        self.hierarchy_starting_height = 50
        self.hierarchy_separation = 15
        self.hierarchy_box_height = 30

        personality_view = Rectangle(Point(0, y), Point(width, height))
        personality_view.setFill(color_rgb(189, 215, 238))
        personality_view.setWidth(2)
        personality_view.setOutline(color_rgb(32, 56, 100))
        personality_view.draw(self.win)

        labelWidth = 95
        labelHeight = 30
        label = Text(Point(int(0.02 * width + labelWidth / 2), int(0.025 * height + y)), "Personality")
        label.setSize(23)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("bold")
        label.draw(self.win)

        desc = 'Agent | (N1,N2,N3,N4,N5,N6) | (C1,C2,C3,C4,C5,C6) | (A1,A2,A3,A4,A5,A6) | (E1,E2,E3,E4,E5,E6) | (O1,O2,O3,O4,O5,O6)'
        labelWidth = 10 + self.estimate_length(desc)
        label = Text(Point(int(10 + labelWidth / 2), int(0.025 * height + y + labelHeight)), desc)
        label.setSize(17)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("normal")
        label.draw(self.win)

        # Interaction -> [components]
        self.interactions = {}

        green = color_rgb(112, 173, 71)
        blue = color_rgb(47, 85, 151)
        red = color_rgb(255, 0, 0)
        length = 20

        friendship_template = ArrowLineTemplate(2.5, green,
                                                          ArrowHeadTemplate(ArrowType.line, green, 2.5, 15),
                                                          ArrowHeadTemplate(ArrowType.line, green, 2.5, 15))
        mentorship_template = ArrowLineTemplate(2.5, blue,
                                                ArrowHeadTemplate(ArrowType.line, blue, 2.5, 15),
                                                ArrowHeadTemplate(ArrowType.line, blue, 2.5, 15))

        help_template = ArrowLineTemplate(2.5, green,
                                                ArrowHeadTemplate(ArrowType.none, green, 2.5, 15),
                                                ArrowHeadTemplate(ArrowType.line, green, 2.5, 15))

        theft_template = ArrowLineTemplate(2.5, red,
                                                ArrowHeadTemplate(ArrowType.none, red, 2.5, 15),
                                                ArrowHeadTemplate(ArrowType.line, red, 2.5, 15))

        # Type -> (colour, length, width, angle,  (on_first_point, is_line_type), (on_second_point, is_line_type) )
        self.interaction_type_display = {src.Interaction.Friendship: friendship_template,
                                         src.Interaction.Mentorship: mentorship_template,
                                         src.Interaction.Help: help_template,
                                         src.Interaction.Theft: theft_template}

        self.mine_arrow_line_template = ArrowLineTemplate(2.5, color_rgb(0, 0, 0),
                                                          ArrowHeadTemplate(ArrowType.none, color_rgb(0, 0, 0), 2.5, 15),
                                                          ArrowHeadTemplate(ArrowType.line, color_rgb(0, 0, 0), 2.5, 15))






        # Agent -> { components: [componentlist], radius: radius}
        self.agent_to_components_info = {}

        self.personality_starting_height = int(0.025 * height + y + labelHeight + 25)
        self.personality_current_height = self.personality_starting_height





    def estimate_length(self, text):
        return len(text) * 8

    def draw_agent_personality(self, name, personality):
        order = ['N', 'C', 'A', 'E', 'O']
        text = name + ' | '
        for j in range(0, 5):
            c = order[j]
            dimension = '('
            for i in range(1, 7):
                p = personality[c + str(i)]
                t = '0' + str(p) if p < 10 else str(p)
                dimension += t
                if i < 6:
                    dimension += ","
            dimension += ')'
            text += dimension
            if j < 4:
                text += ' | '

        labelWidth = self.estimate_length(text)
        label = Text(Point(int(labelWidth / 2), self.personality_current_height), text)
        label.setSize(17)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("normal")
        label.draw(self.win)
        self.personality_canvas.append(label)
        self.personality_current_height += 25

    def close(self):
        self.win.close()

    def clear_agent_canvas(self):
        for key in self.agent_canvas:
            self.remove_components(self.agent_canvas[key])
        self.agent_canvas.clear()

    def clear_personality_canvas(self):
        for item in self.personality_canvas:
            item.undraw()
        self.personality_canvas.clear()
        self.personality_current_height = self.personality_starting_height

    def display(self, secs, rate):
        for i in range(1, int(secs * rate)):
            update(rate)



    def rotate_point(self, pivot_point, angle, point):
        xr, yr = point.x, point.y

        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        # translate point back to origin
        px = xr - pivot_point.x
        py = yr - pivot_point.y

        #rotate point
        xnew = px * c - py * s
        ynew = px * s + py * c

        # translate point back from origin
        px = xnew + pivot_point.x
        py = ynew + pivot_point.y

        return Point(px,py)

    def point_on_line(self, p1, p2, length):
        a = length / math.sqrt(math.pow(p2.y - p1.y, 2) + math.pow(p2.x - p1.x, 2))
        point = Point(p1.x + a * (p2.x - p1.x), p1.y + a * (p2.y - p1.y))
        return point




    def draw_arrow_head(self, p1, p2, template):
        point = self.point_on_line(p1, p2, template.length)
        p3 = self.rotate_point(p1, 45, point)
        p4 = self.rotate_point(p1, -45, point)

        if template.arrow_type.name == 'line':
            return self.draw_lined_arrow_head(p1,p3,p4,template.colour,template.width)
        elif template.arrow_type.name == 'triangle':
            return self.draw_filled_arrow_head(p1,p3,p4,template.colour,template.width)

        return []


    def draw_lined_arrow_head(self,p1,p3,p4, colour, width):
        components = []
        line = Line(p1, p3)
        line.setFill(colour)
        line.setWidth(width)
        line.draw(self.win)
        components.append(line)

        line = Line(p1, p4)
        line.setFill(colour)
        line.setWidth(width)
        line.draw(self.win)
        components.append(line)

        p1.setFill(colour)
        p1.draw(self.win)
        components.append(p1)

        return components

    def draw_filled_arrow_head(self,p1,p3,p4, colour, width):

        components = []

        vertices = [p1,p3,p4]

        poly = Polygon(vertices)

        poly.setFill(colour)

        poly.setWidth(width)
        poly.setOutline(color_rgb(0,200,0))

        poly.draw(self.win)

        components.append(poly)

        return components

    def draw_arrow_line(self, p1, p2, template):
        components = []
        line = Line(p1, p2)
        line.setFill(template.colour)
        line.setWidth(template.width)
        line.draw(self.win)
        components.append(line)

        components += self.draw_arrow_head(p1,p2,template.first_arrow_head)

        components += self.draw_arrow_head(p2,p1,template.second_arrow_head)

        return components


    def draw_wealth_arrow(self, p1, p2, text):
        x = (p1.x + p2.x)/2
        y = p1.y - 25


        arrow = self.draw_arrow_line(p1,p2,self.mine_arrow_line_template)
        label = Text(Point(x, y), text)
        label.setSize(18)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("bold")
        label.draw(self.win)
        arrow.append(label)

        return arrow


    def move_independent_agent(self, agent, radius, x, y):
        # Agent -> { components: [componentlist], radius: radius, interactions: [InteractionList] }

        i = []
        if agent in self.agent_to_components_info:
            agent_components = self.agent_to_components_info[agent]
            components = agent_components["components"]
            i = agent_components["interactions"]
            # undraw current agent
            self.remove_components(components)

        # (re)draw current agent
        components = self.draw_adv_agent(agent.name,x,y,radius)
        self.agent_to_components_info[agent] = { "components": components, "radius": radius, "interactions": i}

        return components

    def get_interaction_window_boundaries(self, independent_agents, radius=20, padding=10):
        sep = radius*2 + padding
        x = self.x - self.resource_width
        xt = x - 1.5*sep
        nxt, nx, ny = int(xt / sep), int(x / sep), int(self.y / sep)
        z = nxt + nx + ny
        n = int(len(independent_agents) / z)
        rem = len(independent_agents) - n*z

        print(nxt,nx,ny)
        wx, wy1 = n*sep, n*sep
        wy2 = self.y - n*sep

        if rem > 0:
            wy1 += sep

        if rem > nxt:
            wx += sep

        if rem > (nxt + ny):
            wy2 -= sep

        return wx, wy1, wy2






    def draw_independent_agents(self, independent_agents, radius=20, padding=10):
        x = self.x - self.resource_width
        ideal_diameter = radius*2
        padding = padding
        sep = ideal_diameter + padding

        xt = x - 1.5*sep
        # number of agents that can fit at the top edge
        nxt = int(xt / sep)

        # number of agents that can fit at the bottom edge
        nx = int(x / sep)

        # number of agents that can fit at the left edge
        ny = int((self.y - sep) / sep)

        # try to fit as many agents around the edge

        index = 0

        n = len(independent_agents)

        count = 0

        while index < n:
            # fit agents on top
            end = min(index + nxt, n)
            sublist = independent_agents[index:end]
            # for each agent in sublist - move agent to correct position
            ty = sep/2 + sep*count
            tx = xt - sep/2
            for i in range(len(sublist)):
                agent = sublist[i]
                self.move_independent_agent(agent,ideal_diameter/2,tx,ty)
                tx -= sep

            index = end

            # fit agents on left
            end = min(index + ny, n)
            sublist = independent_agents[index:end]
            # for each agent in sublist - move agent to correct position
            tx = sep / 2 + sep * count
            ty = sep + sep / 2
            for i in range(len(sublist)):
                agent = sublist[i]
                self.move_independent_agent(agent, ideal_diameter / 2,tx,ty)
                ty += sep

            index = end

            # fit agents on bottom
            end = min(index + nx, n)
            sublist = independent_agents[index:end]
            # for each agent in sublist - move agent to correct position
            ty = (self.y - sep / 2) - sep * count
            tx = sep / 2
            for i in range(len(sublist)):
                agent = sublist[i]
                self.move_independent_agent(agent, ideal_diameter / 2, tx, ty)
                tx += sep

            index = end

            count += 1


    def display_mine_animations(self, r, agentsWealth, sY, seperationY, rate, speed):
        x = self.x - self.resource_width
        x1 = x - 2*r
        x2 = x - r
        x3 = x + r

        # animation of agent moving from x1 to x2
        for x in range(int(x1),int(x2), int(speed*3)):
            ax = x + r
            ax2 = ax + 1.5*r
            components = []
            for i in range(0, len(agentsWealth)):
                y = sY + seperationY * i
                ax = x + r
                ax2 = ax + 1.5 * r
                agent = agentsWealth[i][0]

                #components += self.draw_adv_agent(agent.name, x, y, r) +  self.draw_mine_arrow(Point(ax, y),Point(ax2, y),False, True)

                components += self.move_independent_agent(agent,r,x,y) + self.draw_arrow_line(Point(ax, y), Point(ax2, y), self.mine_arrow_line_template)

            update(rate)
            self.remove_components(components)

        # animation of agent moving from x2 to x3 (no arrow)
        for x in range(int(x2), int(x3), int(speed*4)):
            ax = x + r
            ax2 = ax + 1.5 * r
            components = []
            for i in range(0, len(agentsWealth)):
                y = sY + seperationY * i
                ax = x + r
                ax2 = ax + 1.5 * r
                agent = agentsWealth[i][0]
                components += self.move_independent_agent(agent, r, x, y)
                #components += self.draw_adv_agent(agent.name, x, y, r)
            update(rate)
            self.remove_components(components)

        # animation of agent moving from x3 to x2 (arrow)
        for x in range(int(x3), int(x2), int(speed*-3)):
            ax = x - r
            ax2 = ax - 1.5 * r
            components = []
            for i in range(0, len(agentsWealth)):
                y = sY + seperationY * i
                ax = x - r
                ax2 = ax - 1.5 * r
                agent = agentsWealth[i][0]
                wealth = agentsWealth[i][1]

                #components += self.draw_adv_agent(agent.name,x,y,r) + self.draw_wealth_arrow( Point(ax,y),Point(ax2,y),("+" + str(wealth)), False,True)
                components += self.move_independent_agent(agent,r,x,y) + self.draw_wealth_arrow(Point(ax, y), Point(ax2,y), ("+" + str(wealth)))
            update(rate)
            self.remove_components(components)

        # animation of agent moving from x2 to x1 (no arrow)
        for x in range(int(x2), int(x1), int(speed*-4)):
            ax = x - r
            ax2 = ax - 1.5 * r
            components = []
            for i in range(0, len(agentsWealth)):
                y = sY + seperationY * i
                ax = x - r
                ax2 = ax - 1.5 * r
                agent = agentsWealth[i][0]
                components += self.move_independent_agent(agent, r, x, y)
                #components += self.draw_adv_agent(agent.name, x, y, r)
            update(rate)
            self.remove_components(components)

    def test_display(self,agentWealth, interactions):
        sY = 40
        rate = 30
        r = 20
        seperationY = 80
        #self.display_mine_animations(r,agentWealth,sY,seperationY,rate, 1.6)
        #self.display_mine_animation(agentWealth,rate, 2)

        keys = [u[0] for u in agentWealth]
        b = {u: [] for u in keys}
        self.draw_independent_agents(keys)
        self.display(3, rate)
        components = self.display_interaction(interactions,keys)
        self.display(3, rate)
        self.clear_interactions()
        self.draw_independent_agents(keys)
        self.display(1, rate)

        #self.set_social_hierarchy()


        self.display_mine_animation(agentWealth, rate, 1.6)








    def display_mine_animation(self, agentsToWealth, rate, speed, radius = 20):
        agentHeight = 70
        paddingHeight = 10
        seperationY = agentHeight + paddingHeight
        n = int(self.y / seperationY)

        index = 0

        aw = [u[0] for u in agentsToWealth]

        while index < len(agentsToWealth):
            end = min(index+n,len(agentsToWealth))
            sublist = agentsToWealth[index:end]


            index = end
            self.display_mine_animations(radius,sublist,seperationY/2,seperationY,rate,speed)


            self.draw_independent_agents(aw[0:end])


            # pause between animations
            #if index < len(agentsToWealth):
                #self.display(0.2*1/speed, rate)

    def draw_interction(self, key, first, second):
        components = []

        w = 0.6 * self.x
        s = 0.2 * self.x
        x1 = random.randrange(s, int((s + w / 2)))

        h = 0.6 * self.y
        s2 = 0.2 * self.y
        y1 = random.randrange(s2, int(s2 + h / 2))

        w2 = 0.8 * self.x
        x2 = random.randrange(int(s + w / 2), w2)

        h2 = 0.8 * self.y
        y2 = random.randrange(int(s2 + h / 2), h2)

        line = Line(Point(x1, y1), Point(x2, y2))
        line.setFill(color_rgb(237, 125, 49))
        line.setWidth(5)

        line.draw(self.win)
        components.append(line)

        ag1 = self.draw_agent(first, x1, y1)
        ag2 = self.draw_agent(second, x2, y2)

        components = components + ag1 + ag2

        self.agent_canvas[key] = components

    def draw_adv_agent(self,name,x,y,r):
        components = []

        circle = Circle(Point(x, y), r)
        circle.setWidth(2)
        circle.setFill(color_rgb(255, 192, 0))
        circle.setOutline(color_rgb(155, 140, 0))
        circle.draw(self.win)
        components.append(circle)

        label = Text(Point(x, y), name)
        label.setSize(int(20*r/35))
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("bold")
        label.draw(self.win)
        components.append(label)

        return components


    def draw_agent(self, name, x, y):
        return self.draw_adv_agent(name,x,y,35)

    def draw_hierarchy_box(self, name, status, position):

        sx = 10 + self.x
        ex = self.width - 10

        sy = self.hierarchy_starting_height + position * (self.hierarchy_box_height + self.hierarchy_separation)
        ey = sy + self.hierarchy_box_height

        number_width = 30
        x1 = sx + number_width
        x2 = x1 + ((ex - sx) - number_width) / 2

        components = []
        if position in self.hierarchy_canvas:
            self.remove_components(self.hierarchy_canvas[position])
            self.hierarchy_canvas.pop(position, None)

        left = Rectangle(Point(sx, sy), Point(x1, ey))
        left.setFill('white')
        left.setWidth(2)
        left.setOutline('black')
        left.draw(self.win)
        components.append(left)

        rank = Text(Point(sx + int((x1 - sx) / 2), sy + int((ey - sy) / 2)), str(position + 1))
        rank.setSize(18)
        rank.setTextColor(color_rgb(237, 125, 49))
        rank.setFace("arial")
        rank.setStyle("bold")
        rank.draw(self.win)
        components.append(rank)

        middle = Rectangle(Point(x1, sy), Point(x2, ey))
        middle.setFill('white')
        middle.setWidth(2)
        middle.setOutline('black')
        middle.draw(self.win)
        components.append(middle)

        nameLabel = Text(Point(x1 + int((x2 - x1) / 2), sy + int((ey - sy) / 2)), name)
        nameLabel.setSize(18)
        nameLabel.setTextColor("black")
        nameLabel.setFace("arial")
        nameLabel.setStyle("bold")
        nameLabel.draw(self.win)
        components.append(nameLabel)

        right = Rectangle(Point(x2, sy), Point(ex, ey))
        right.setFill('white')
        right.setWidth(2)
        right.setOutline('black')
        right.draw(self.win)
        components.append(right)

        statusLabel = Text(Point(x2 + int((ex - x2) / 2), sy + int((ey - sy) / 2)), status)
        statusLabel.setSize(18)
        statusLabel.setTextColor("black")
        statusLabel.setFace("arial")
        statusLabel.setStyle("bold")
        statusLabel.draw(self.win)
        components.append(statusLabel)

        self.hierarchy_canvas[position] = components

    def clear_hierarchy_canvas(self):
        for p in self.hierarchy_canvas:
            self.remove_components(self.hierarchy_canvas[p])
            self.hierarchy_canvas[p].clear()


    def remove_interaction(self, interaction):
        if interaction in self.interactions:
            self.remove_components(self.interactions[interaction])

    def clear_interactions(self):
        for interaction in self.interactions:
            self.remove_components(self.interactions[interaction])

    def get_angle(self,p1,p2):
        if (p2.y - p1.y) == 0:
            a = math.radians(0) if p2.x > p1.x else math.radians(180)
        else:
            a = math.atan((p2.x - p1.x) / (p2.y - p1.y))
        return a





    def draw_single_interaction(self, interaction, p1, p2, radius=20):
        a1 = self.get_angle(p1,p2)
        a2 = self.get_angle(p2,p1)
        print("angle",a1,a2)

        np1 = self.point_on_line(p1,p2,radius)
        np2 = self.point_on_line(p2,p1,radius)


        #np1 = Point(p1.x + radius*math.cos(a1), p1.y + radius*math.sin(a1))
        #np2 = Point(p2.x + radius * math.cos(a2), p2.y + radius * math.sin(a2))

        components = []

        components += self.move_independent_agent(interaction.get_reactive_agent(),radius, p1.x,p1.y)
        components += self.move_independent_agent(interaction.get_proactive_agent(), radius, p2.x, p2.y)

        template = self.interaction_type_display[type(interaction)]

        l1 = template.first_arrow_head.length
        l2 = template.second_arrow_head.length
        tp1 = self.point_on_line(np1, np2, 2 * l1)
        tp2 = self.point_on_line(np2, np1, 2 * l2)


        components += self.draw_arrow_line(np1,np2,template)


        size = 15

        if interaction.requires_acceptance:
            accepted_agent = interaction.get_accepted_agent()
            if accepted_agent is not None:
                if interaction.is_success:
                    text = "ACCEPTOR"
                    colour = color_rgb(0,255,0)
                else:
                    text = "REJECTOR"
                    colour = color_rgb(255, 0, 0)

                respond_point = tp2
                if interaction.get_reactive_agent() == interaction.get_requested_agent():
                    respond_point, request_point = tp1

                components.append(self.get_text(respond_point, text, size, colour, "arial", "bold"))

        else:
            if isinstance(interaction,src.Interaction.Help):
                funds = str(interaction.helping_funds)
                components.append(self.get_text(tp2,"+" + funds,size,"black","arial","bold"))


            elif isinstance(interaction,src.Interaction.Theft):
                funds = str(interaction.stolen_funds)
                if interaction.is_caught:
                    components.append(self.get_text(tp2, "0", size, "black", "arial", "bold"))
                    components.append(self.get_text(tp1, "caught", size, "red", "arial", "bold"))

                components.append(self.get_text(tp2, "-" + funds, size, "black", "arial", "bold"))

        return components


    def get_text(self, point, text, size, colour, font, style):
        label = Text(point, text)
        label.setSize(size)
        label.setTextColor(colour)
        label.setFace(font)
        label.setStyle(style)
        label.draw(self.win)
        return label

    def display_interaction(self, interactions, independent_agents, radius=20, padding=10):
        #independent_agents = [u[0] for u in independent_agents]
        self.draw_independent_agents(independent_agents,radius,padding)

        bx, by1, by2 = self.get_interaction_window_boundaries(independent_agents, radius, padding)
        bx2 = self.x - self.resource_width

        wx, wx2 = bx, bx2
        wy1,wy2 = by1, by2

        components = []
        asep, bsep, isep = 2*radius, 2*radius, 110
        ws = asep + isep + bsep

        nwx = int( (wx2 - wx) / ws)
        nwy = int((wy2 - wy1) / ws)

        factor = min(((wx2 - wx) - ws*nwx)/nwx, ((wy2 - wy1) - ws*nwy)/nwy)

        asep += factor

        ws = asep + isep + bsep

        remaining_blocks = [(x, y) for x in range(nwx) for y in range(nwy)]

        interactions_not_displayed = interactions.copy()

        for i in range(len(interactions)):
            interaction = interactions[i]

            if len(remaining_blocks) == 0:
                break

            interactions_not_displayed.remove(interaction)

            index = random.randrange(len(remaining_blocks))
            x,y = remaining_blocks[index]
            remaining_blocks.pop(index)

            sx, sy = wx + x*ws, wy1 + y*ws
            sx2, sy2 = sx + ws, sy + ws

            # rec = Rectangle(Point(sx, sy), Point(sx2, sy2))
            # rec.setFill(color_rgb(200, 0, 0))
            # rec.draw(self.win)

            x1 = (sx+bsep/2, sx+bsep/2 + asep/2)
            x2 = (sx+bsep/2 + asep/2 + isep, sx2-bsep/2)
            x3 = (sx+bsep/2, sx2-bsep/2)

            y1 = (sy+bsep/2, sy + bsep/2 + asep/2)
            y2 = (sy + bsep/2 + isep + asep/2, sy2 - bsep/2)
            y3 = (sy + bsep/2, sy2 - bsep/2)

            strategies = [((x1, x2), (y3, y3)), ((x2, x1), (y3, y3)),
                          ((x3, x3), (y1, y2)), ((x3, x3), (y2, y1))]

            strat = strategies[random.randrange(len(strategies))]

            # rec = Rectangle(Point(strat[0][0][0],strat[1][0][0]), Point(strat[0][0][1],strat[1][0][1]))
            # rec.setFill("blue")
            # rec.draw(self.win)
            # rec = Rectangle(Point(strat[0][1][0], strat[1][1][0]), Point(strat[0][1][1], strat[1][1][1]))
            # rec.setFill("blue")
            # rec.draw(self.win)

            # draw interaction at random position in this window
            p1 = Point(random.randrange(int(strat[0][0][0]), int(strat[0][0][1])), random.randrange(int(strat[1][0][0]), int(strat[1][0][1])))

            # draw interaction at random position in this window
            p2 = Point(random.randrange(int(strat[0][1][0]), int(strat[0][1][1])), random.randrange(int(strat[1][1][0]), int(strat[1][1][1])))


            self.interactions[interaction] = self.draw_single_interaction(interaction,p1,p2,radius)

            #Interaction -> [components]
        return interactions_not_displayed

    def remove_components(self, components):
        for x in components:
            x.undraw()

    def set_social_hierarchy(self, agentToStatus):
        sorted_x = sorted(agentToStatus.items(), key=operator.itemgetter(1))
        position = 0
        self.clear_hierarchy_canvas()
        sorted_x.reverse()

        for x in sorted_x:
            agent = x[0]
            status = x[1]
            self.draw_hierarchy_box(agent.name, status, position)
            position += 1

    def set_agent_personality_list(self, agents):
        self.clear_personality_canvas()
        for a in agents:
            self.draw_agent_personality(a.name, a.personality)




class ArrowType(enum.Enum):
    line = 1
    triangle = 2
    none = 3


class ArrowLineTemplate:
    def __init__(self, width, colour, first_arrow_head, second_arrow_head):
        self.width = width
        self.colour = colour
        self.first_arrow_head = first_arrow_head
        self.second_arrow_head = second_arrow_head


class ArrowHeadTemplate:
    def __init__(self, arrow_type, colour, width, length):
        self.arrow_type = arrow_type
        self.colour = colour
        self.width = width
        self.length = length