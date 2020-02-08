import operator
import random
from src.graphics import *
import math





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

        # Interaction -> [ Agent
        self.interactions = {}

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



    def draw_arrow_head(self, p1, p2,deg,length, colour, width, is_line_type ):
        deg = math.radians(deg)

        if (p2.y - p1.y) == 0:
            a = math.radians(0) if p2.x > p1.x else math.radians(180)
        else:
            a = math.atan((p2.x - p1.x) / (p2.y - p1.y))

        p3 = Point(p1.x + length * math.cos(a + deg), p2.y + length * math.sin(a + deg))
        p4 = Point(p1.x + length * math.cos(a - deg), p2.y + length * math.sin(a - deg))

        if is_line_type:
            return self.draw_lined_arrow_head(p1,p3,p4,colour,width)

        return self.draw_filled_arrow_head(p1,p3,p4,colour,width)




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

        poly.draw(self.win)

        components.append(poly)

        return components



    def draw_arrow_line(self, p1, p2, length, on_first_point, on_second_point, deg, colour, width,is_line_type):

        components = []
        line = Line(p1, p2)
        line.setFill(colour)
        line.setWidth(width)
        line.draw(self.win)
        components.append(line)

        if on_first_point:
            components += self.draw_arrow_head(p1,p2,deg,length,colour,width,is_line_type)

        if on_second_point:
            components += self.draw_arrow_head(p2,p1,deg,length,colour,width,is_line_type)

        return components



    def draw_mine_arrow(self, p1, p2, on_first_point, on_second_point):
        #l = math.sqrt( int(abs(x2 - x))^2 + int(abs(y2 - y))^2)

        l = math.sqrt(int(abs(p2.x - p1.x)) ^ 2 + int(abs(p2.y - p1.y)) ^ 2)

        print(l)
        r = 10*l * 0.25
        deg = 45
        colour = color_rgb(0, 0, 0)
        width = 2.5
        is_line_type = True
        return self.draw_arrow_line(p1,p2,r,on_first_point,on_second_point,deg,colour,width,is_line_type)


    def draw_wealth_arrow(self, p1, p2, text, on_first_point, on_second_point):
        x = (p1.x + p2.x)/2
        y = p1.y - 25
        arrow = self.draw_mine_arrow(p1,p2,on_first_point,on_second_point)
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

    def get_interaction_window_boundaries(self, independent_agents, radius=35, padding=10):
        sep = radius*2 + padding
        x = self.x - self.resource_width
        xt = x = 1.5*sep
        nxt, nx, ny = int(xt / sep), int(x / sep), int(self.y / sep)
        z = nxt + nx + ny
        n = int(len(independent_agents) / z)
        rem = len(independent_agents) - n*z

        wx, wy1 = n*sep
        wy2 = self.y - n*sep

        if rem > 0:
            wy1 += sep

        if rem > nxt:
            wx += sep

        if rem > (nxt + ny):
            wy2 -= sep

        return wx, wy1, wy2






    def draw_independent_agents(self, independent_agents, radius=35, padding=10):
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
        ny = int(self.y / sep)

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
            ty = sep / 2
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

                components += self.move_independent_agent(agent,r,x,y) + self.draw_mine_arrow(Point(ax, y), Point(ax2, y), False, True)

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
                components += self.move_independent_agent(agent,r,x,y) + self.draw_wealth_arrow(Point(ax, y), Point(ax2, y), ("+" + str(wealth)), False, True)
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

    def test_display(self,agentWealth):
        sY = 40
        rate = 30
        r = 35
        seperationY = 80
        #self.display_mine_animations(r,agentWealth,sY,seperationY,rate, 1.6)
        #self.display_mine_animation(agentWealth,rate, 2)

        keys = [u[0] for u in agentWealth]
        b = {u: [] for u in keys}
        self.draw_independent_agents(keys)
        self.display(3, rate)
        self.display_mine_animation(agentWealth, rate, 2)








    def display_mine_animation(self, agentsToWealth, rate, speed):
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
            self.display_mine_animations(35,sublist,seperationY/2,seperationY,rate,speed)


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
        label.setSize(18)
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

    def remove_interaction(self, key):
        if key not in self.agent_canvas:
            pass
        self.remove_components(self.agent_canvas[key])


    def draw_single_interaction(self, interaction, p1, p2, radius=35):
        if (p2.y - p1.y) == 0:
            a = math.radians(0) if p2.x > p1.x else math.radians(180)
        else:
            a = math.atan((p2.x - p1.x) / (p2.y - p1.y))
        a2 = math.radians(90) - a
        np1 = Point(p1.x + radius*math.cos(a), p1.y + radius*math.sin(a))
        np2 = Point(p2.x + radius * math.cos(a2), p2.y + radius * math.sin(a2))

        components = []
        components += self.draw_adv_agent(interaction.reactive_agent(),p1.x,p1.y,radius)
        components += self.draw_adv_agent(interaction.proactive_agent(), p2.x, p2.y, radius)
        components += self.draw_arrow_line(np1,np2,15,False,True)





    def display_interaction(self, interactions, independent_agents, radius=35, padding=10):
        wx, wy1, wy2 = self.get_interaction_window_boundaries(independent_agents,radius,padding)

        sep = 5*(radius*2)
        winx = (self.x - self.resource_width) - wx
        winy = wy2 - wy1

        x = int(winx / sep)
        y = int(winy / sep)

        partx = sep + (winx - x*sep)/x if x > 0 else winx
        party = sep + (winy - y * sep) / y if y > 0 else winy

        nx, ny = int(winx / partx), int(winy / party)

        for i in range(1,len(interactions)+1):
            interaction = interactions[i-1]

            by = int(i / nx)
            bx = i % nx

            # window coordinates
            x1, y1 = wx + bx*partx, wy1 + by*party
            x2, y2 = x1 + partx, y1 + partx

            # draw interaction at random position in this window
            p1 = (random.randrange(x1, x2), random.randrange(y1, y2))

            potentialx = [x for x in range(int(x1, x2)) if radius < x < x2 - radius and
                          (x < (p1[0]-2*radius) or x > (p1[0]+2*radius))]
            potentialy = [y for y in range(int(y1, y2)) if radius < y < y2 - radius and
                          (y < (p1[1]-2*radius) or y > (p1[1]+2*radius))]

            p2 = (potentialx[random.randrange(len(potentialx))], potentialy[random.randrange(len(potentialy))])









































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




