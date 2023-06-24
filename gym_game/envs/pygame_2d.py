import pygame
import math
import numpy as np
import random

screen_width = 1500
screen_height = 800
pen_start_point = [120, 590]
goal_point = [1260, 600]

draw_points = []

class Pen:
    def __init__(self, pen_file, map_file, pos):
        print(":::---------------- Initializing Pen ---------------- ::::")
        self.surface = pygame.image.load(pen_file)
        self.map = pygame.image.load(map_file)
        # self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.pen_rect = self.surface.get_rect()
        self.pos = pos
        self.prev_position = pen_start_point
        self.pen_start_point = pos

        self.distance_to_left_boundary = 10
        self.distance_to_right_boundary = 10
        self.distance_to_up_boundary = 10
        self.distance_to_down_boundary = 10

        self.action = 0
        self.distance = 0
        self.y_distance_from_start = pen_start_point[1] - self.pos[1]
        self.x_distance_to_goal = int(goal_point[0] - self.pos[0])
        self.out_or_hitting_snake_boundary = False


        self.is_alive = True
        self.time_spent = 0
        self.draw_path = []


    def draw(self, screen):
        screen.blit(self.surface, self.pos)
        self.map.set_at(self.pos, "red") # draw a pixel on map
    
        x = self.pos[0] 
        y = self.pos[1] 

        # when coming to this function, the action is already done. so we have to draw back that much of distance to the opposite direction.
        if self.action == 0:
            pygame.draw.line(self.map, (255, 0, 0), (x, y), (x + self.distance, y), 2)
        elif self.action == 1:
            pygame.draw.line(self.map, (255, 0, 0), (x, y), (x - self.distance, y), 2)
        elif self.action == 2:
            pygame.draw.line(self.map, (255, 0, 0), (x, y), (x, y + self.distance), 2)
        elif self.action == 3:
            pygame.draw.line(self.map, (255, 0, 0), (x, y), (x, y - self.distance), 2)


    def is_hitting_snake_boundary(self):

        print("this position color: ")
        print(self.map.get_at((self.pos[0] + 1, self.pos[1])))

        if((self.map.get_at((self.pos[0], self.pos[1])) == (255, 255, 255, 255)) or 
           (self.map.get_at((self.pos[0], self.pos[1])) == (0, 0, 0, 255))) :
            
            self.out_or_hitting_snake_boundary = False
            self.is_alive = True
            return False

        elif (self.map.get_at((self.pos[0], self.pos[1])) == (255, 0, 0, 255)):
            # check sourounding pixels to see if it is white. if not white, then it is out of snake boundary
            if(self.map.get_at((self.pos[0] + 1, self.pos[1])) == (255, 255, 255, 255) 
                or self.map.get_at((self.pos[0] - 1, self.pos[1])) == (255, 255, 255, 255) 
                or self.map.get_at((self.pos[0], self.pos[1] + 1)) == (255, 255, 255, 255) 
                or self.map.get_at((self.pos[0], self.pos[1] - 1)) == (255, 255, 255, 255)
                or self.map.get_at((self.pos[0] + 2, self.pos[1])) == (255, 255, 255, 255) 
                or self.map.get_at((self.pos[0] - 2, self.pos[1])) == (255, 255, 255, 255) 
                or self.map.get_at((self.pos[0], self.pos[1] + 2)) == (255, 255, 255, 255) 
                or self.map.get_at((self.pos[0], self.pos[1] - 2)) == (255, 255, 255, 255)):

                print("was on red strokes, but still within the snake")
                self.out_or_hitting_snake_boundary = False
                return False
            else:
                print("was on red strokes, but seems outside of snake.. so Im killing the pen as a penalty")
                self.out_or_hitting_snake_boundary = True
                self.is_alive = False
                return True
        else:
            print("outside of snake boundary for sure... killing the pen as a penalty")
            self.out_or_hitting_snake_boundary = True
            self.is_alive = False
            return True
        
    def is_going_to_go_out_of_boundary(self, p, distance):
        if(p[0] - distance < 0 or p[0] + distance > 1500 or p[1] - distance < 0 or p[1] + distance > 800):
            print(":::: This change is going to go out of boundary, so don't do this change. and Im killing the pen as a penalty ::::")
            self.is_alive = False
            return True
        else:
            return False



    def update(self):
        # random normal distribution
        # self.distance = 50
        # self.distance = int(np.random.normal(loc=2, scale = 1, size=1))
        self.distance = int(np.random.normal(loc=10, scale = 4, size=1))

        print("distance to draw: ", self.distance, " in direction: ", self.action, " from pos: ", self.pos)

        if not self.is_going_to_go_out_of_boundary(self.pos, self.distance):

            if self.action == 0: # move left
                self.pos[0] -= self.distance
            if self.action == 1: # move right
                self.pos[0] += self.distance
            if self.action == 2: # move up
                self.pos[1] -= self.distance
            if self.action == 3: # move down
                self.pos[1] += self.distance

            print("now position:                          self.x ", self.pos[0], ", self.y ", self.pos[1])

            # updating observation data after action done.
            self.y_distance_from_start = int(pen_start_point[1] - self.pos[1])
            self.x_distance_to_goal = int(goal_point[0] - self.pos[0])    
        
            self.is_hitting_snake_boundary()

            if(self.out_or_hitting_snake_boundary): # if out of boundary or hit the snake boundary change the position to previous position.

                if self.action == 0: # move left
                    self.pos[0] += self.distance
                if self.action == 1: # move right
                    self.pos[0] -= self.distance
                if self.action == 2: # move up
                    self.pos[1] += self.distance
                if self.action == 3: # move down
                    self.pos[1] -= self.distance
                
                self.distance = 0
                print("--------------------- hit the boundary or drew outside the boundary. So change the position to previous position. ---------------------")
                print("changed back, now position:                          self.x ", self.pos[0], ", self.y ", self.pos[1])
                return
            
            print("=============== changed position color: ", self.map.get_at((self.pos[0], self.pos[1])))

class PyGame2D:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 60)

        self.pen = Pen('pen.png', 'map_cobra_3.png', [120,590])
        print("=====>>>>> pen pos: ", self.pen.pos)

        self.game_speed = 30
        self.mode = 0

    def action(self, action):
        # actions =
            # 0. move left
            # 1. move right
            # 2. move up
            # 3. move down

        if action == 0:  # move left
            self.pen.action = 0
        if action == 1:  # move right
            self.pen.action = 1
        elif action == 2: # move up
            self.pen.action = 2
        elif action == 3:   # move down
            self.pen.action = 3

        self.pen.update()
        


    def evaluate(self):
        reward = 0

        if not self.pen.is_alive:
            if self.pen.y_distance_from_start == 0:
                reward = -200
            else:
                reward = -100 - int(1000/self.pen.y_distance_from_start)
            print(":::: penalty because pen is dead, reward: ", reward, "::::")
        
        if self.pen.is_alive:
            if self.pen.distance == 0:
                print("::: penalty because distance is 0 which means moved out or hit the snake boundry :::")
                reward += -50
            else:
                print("pen is alive :::")
                print("::: y_distance_from_start: ", self.pen.y_distance_from_start, "<------------------", 
                      "\n::: x_distance_to_goal: ", self.pen.x_distance_to_goal)

                reward += 2 * self.pen.y_distance_from_start
                reward += 10000 / self.pen.x_distance_to_goal  # to make when decreesing x_distance_to_goal, the reward will increase

                print(":::: reward from pen y distance form start: ", self.pen.y_distance_from_start)
                print(":::: reward from pen x distance to goal: ", 25000 / self.pen.x_distance_to_goal)

                if self.pen.x_distance_to_goal < 20:
                    reward += 100
                
                if self.pen.x_distance_to_goal < 5:
                    reward += 100
        
        return reward

    def is_done(self):
        if not self.pen.is_alive or self.pen.x_distance_to_goal < 10:
            return True
        return False

    def observe(self):
        # return state
        in_or_out = 0
        if self.pen.out_or_hitting_snake_boundary:
            in_or_out = 1
        else:
            in_or_out = 0
            
        ret = [in_or_out, int(self.pen.y_distance_from_start), int(self.pen.x_distance_to_goal)]

        # convert to numpy array
        ret = np.array(ret)
        return ret

    def view(self):
        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode += 1
                    self.mode = self.mode % 3

        self.screen.blit(self.pen.map, (0, 0))

        self.pen.draw(self.screen)


        pygame.display.flip()
        self.clock.tick(self.game_speed)


def get_distance(p1, p2):
	return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))

