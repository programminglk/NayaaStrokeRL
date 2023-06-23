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
        self.surface = pygame.image.load(pen_file)
        self.map = pygame.image.load(map_file)
        # self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.pen_rect = self.surface.get_rect()
        self.pos = pos
        self.prev_position = pen_start_point

        self.distance_to_left_boundary = 10
        self.distance_to_right_boundary = 10
        self.distance_to_up_boundary = 10
        self.distance_to_down_boundary = 10

        self.action = 0
        self.distance = 0
        self.y_distance_from_start = int(-(self.pos[1] - pen_start_point[1]))
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

        if self.action == 0:
            pygame.draw.line(self.map, (255, 0, 0), (x, y), (x - self.distance, y), 2)
        elif self.action == 1:
            pygame.draw.line(self.map, (255, 0, 0), (x, y), (x + self.distance, y), 2)
        elif self.action == 2:
            pygame.draw.line(self.map, (255, 0, 0), (x, y), (x, y - self.distance), 2)
        elif self.action == 3:
            pygame.draw.line(self.map, (255, 0, 0), (x, y), (x, y + self.distance), 2)


    def is_hitting_snake_boundary(self):

        if(self.map.get_at((self.pos[0], self.pos[1])) != (255, 255, 255, 255)):
            if(self.map.get_at((self.pos[0], self.pos[1])) != (255, 0, 0, 255)): # if pos color is not red
                print("hit the boundary")
                self.out_or_hitting_snake_boundary = True
                return True
            else:
                # check sourounding pixels to see if it is white. if not white, then it is out of snake boundary
                if(self.map.get_at((self.pos[0] + 1, self.pos[1])) == (255, 255, 255, 255) 
                   or self.map.get_at((self.pos[0] - 1, self.pos[1])) == (255, 255, 255, 255) 
                   or self.map.get_at((self.pos[0], self.pos[1] + 1)) == (255, 255, 255, 255) 
                   or self.map.get_at((self.pos[0], self.pos[1] - 1)) == (255, 255, 255, 255)):
                    print("was on red strokes, but still within the snake")
                    self.out_or_hitting_snake_boundary = False
                    return False
                else:
                    print("was on red strokes, but seems outside of snake..")
                    self.out_or_hitting_snake_boundary = True
                    return True
        else:
            self.out_or_hitting_snake_boundary = False
            return False
        
    def is_going_to_go_out_of_boundary(self, p, distance):
        if(p[0] - distance < 0 or p[0] + distance > 1500 or p[1] - distance < 0 or p[1] + distance > 800):
            print("This change is going to go out of boundary, so don't do this change.")
            return True
        else:
            return False



    def update(self):
        self.distance = random.randint(20, 30)
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
                print("--------------------- hit the boundary or drew outside the boundary. So change the position to previous position. ---------------------")
                # self.is_alive = False
                return
            
            print("now position:                          self.x ", self.pos[0], ", self.y ", self.pos[1])
            print("=============== changed position color: ", self.map.get_at((self.pos[0], self.pos[1])))

class PyGame2D:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)

        self.pen = Pen('pen.png', 'map_cobra_2.png', pen_start_point)
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
            reward = -10000 + self.pen.y_distance_from_start
        
        if self.pen.is_alive:
            print("pen is alive :::")

            reward += self.pen.y_distance_from_start
            reward += 10000 / self.pen.x_distance_to_goal  # to make when decreesing x_distance_to_goal, the reward will increase

            if(self.pen.distance_to_left_boundary > 350 or self.pen.distance_to_right_boundary > 350 or 
               self.pen.distance_to_up_boundary > 350 or self.pen.distance_to_down_boundary > 350):
                reward += -10000

            if((self.pen.pos[1] < 770 and self.pen.pos[1] > 40 and self.pen.pos[0] < 1470 and self.pen.pos[0] > 40) and
                (self.pen.map.get_at((self.pen.pos[0], self.pen.pos[1])) != (255, 255, 255, 255)) and 
               (self.pen.map.get_at((self.pen.pos[0], self.pen.pos[1])) != (255, 0, 0, 255))):
                reward += -10000

            if self.pen.y_distance_from_start < 0:
                reward += -9000

            if self.pen.x_distance_to_goal < 20:
                reward = 10000

        return reward

    def is_done(self):
        if not self.pen.is_alive or self.pen.x_distance_to_goal < 10:
            return True
        return False

    def observe(self):
        # return state
        within_snake_body = 0
        if(self.pen.distance_to_left_boundary > 350 or self.pen.distance_to_right_boundary > 350 or 
               self.pen.distance_to_up_boundary > 350 or self.pen.distance_to_down_boundary > 350):
            within_snake_body = 1

        ret = [int(within_snake_body), int(self.pen.y_distance_from_start), int(self.pen.x_distance_to_goal)]

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

