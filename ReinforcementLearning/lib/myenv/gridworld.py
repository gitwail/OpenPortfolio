import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import pygame

class gridworld(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,dim=10):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # gridworld dimension 
        self.dim=dim
        # 
        self.observation_space = spaces.Tuple((
                    spaces.Discrete(dim),  # Agent x-coordinate
                    spaces.Discrete(dim),  # Agent y-coordinate
                    spaces.Discrete(dim),  # Target x-coordinate
                    spaces.Discrete(dim)   # Target y-coordinate
                ))
        # init observation space
        self.observation_space=(0,0,dim-1,dim-1)
        # agent position
        self.agent,self.target=self.observation_space[:2],self.observation_space[2:]
       
        self.terminated=False
     
        


    def step(self, action):
      
        x_agent,y_agent=self.agent
        
        if action==0:
            # going right ! make sure not go outside the gridworld
            if x_agent+1<self.dim:
                x_agent+=1

        if action==1:
            #going up 
            if  x_agent-1>=0:
                x_agent-=1

        if action==2:
            #going left
            if y_agent-1>=0:
                y_agent-=1

        if action==3:
            #going down 
            if y_agent+1<self.dim:
                y_agent+=1
        

        self.observation_space=(x_agent,y_agent,self.target[0],self.target[1])
        self.agent=x_agent,y_agent

        # negative reward when agent is still wondering
        reward=-1
        # give positive reward if agent reach its goal
        if self.agent==self.target:
            reward=0
            self.terminated=True
            

        # info and truncated
        info="wandering the maze"
        truncated=""

        return self.observation_space, reward, self.terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.observation_space=(0,0,self.dim-1,self.dim-1)

        self.agent=[0,0]
        info='initialisation'
        self.terminated=False

        

        return self.observation_space, info
    
    def render(self, V,mode='human'):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.dim*50, self.dim*50))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.screen.fill((255, 255, 255))  # Fill the screen with white

        # Draw the cells
        for i in range(self.dim):
            for j in range(self.dim):
                pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(i*50, j*50, 50, 50), 1)
                text_surface = self.font.render(str(V[i,j]), True, (0, 0, 0))
                self.screen.blit(text_surface, (i*50 + 25 - text_surface.get_width() / 2, j*50 + 25 - text_surface.get_height() / 2))

        # Draw the agent
        pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(self.agent[0]*50, self.agent[1]*50, 50, 50))

        # Draw the goal
        pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(self.target[0]*50, self.target[1]*50, 50, 50))

        pygame.display.flip()

        # Cap the frame rate
        self.clock.tick(2)



