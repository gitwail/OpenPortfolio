import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def genArrivals():

        # Define arrival rate (e.g., 10 arrivals per hour)
        lambda_per_hour = 10

        # Define number of intervals in the day (48 intervals for 30-minute spacing over 24 hours)
        num_intervals = 48

        # Generate arrivals for each interval
        arrivals = np.random.poisson(lambda_per_hour / 2, num_intervals)

        # Generate time intervals
        time_intervals = pd.date_range(start="00:00:00", periods=num_intervals, freq="30T").time

        df=pd.DataFrame({'time':time_intervals,'arrivals':arrivals})

        df.to_csv("data/arrivals.csv")

def get_agents_schedule(shifts, shift_duration):
    # Convert shift duration to equivalent number of 30-min slots
    slots_for_shift = 2 * shift_duration

    # Initialize a dictionary
    agents_per_30mins_dict = {}

    # Function to convert index to HH:MM time format
    def index_to_time(index):
        hh = index // 2
        mm = (index % 2) * 30
        return f"{hh:02d}:{mm:02d}"

    # Function to convert HH:MM time to an index
    def time_to_index(time_str):
        hh, mm = map(int, time_str.split(":"))
        return 2 * hh + mm // 30

    # Ensure all time slots are present in the dict
    for i in range(48):
        time_key = index_to_time(i)
        agents_per_30mins_dict[time_key] = 0

    for count, time_str in shifts:
        start_index = time_to_index(time_str)
        for i in range(slots_for_shift):
            if start_index + i < 48:
                time_key = index_to_time(start_index + i)
                agents_per_30mins_dict[time_key] += count

    # Sort the dictionary by keys
    sorted_agents_dict = dict(sorted(agents_per_30mins_dict.items()))

    return sorted_agents_dict


class scheduling(gym.Env):
    """Custom Environment that follows gym interface."""

    

    def time_slots(self):
       # Define the start and end times for the day
        start_time = datetime.strptime("00:00", "%H:%M")
        end_time = datetime.strptime("23:59", "%H:%M")

        # Initialize a dictionary to store the index and time slot pairs
        time_slots_dict = {}

        # Generate time slots for every 30 minutes and store them with an index
        current_time = start_time
        index = 1
        while current_time <= end_time:
            time_slots_dict[index] = current_time.strftime("%H:%M")
            current_time += timedelta(minutes=30)
            index += 1

        # Example usage:
        # Print all index-time slot pairs
        # for index, time_slot in time_slots_dict.items():
        #     print(f"Index {index}: {time_slot}")

        return time_slots_dict


    def __init__(self,n_agents=11,n_services=1,n_shifts=4):
        super().__init__()

        # number of start time
        self.n_st=48
        # number of agents
        self.n_agents=n_agents
        #agent ids
        self.agents_ids=[i for i in range(1,self.n_agents)]

        # time slots
        self.ts=self.time_slots()
        # number of services
        self.n_services=n_services
        # number of shifts
        self.n_shifts=n_shifts

        # observation space
        self.observation_space = spaces.Discrete(n_services*n_shifts)
        self.observation_space=1
       
        # Action space 
        self.action_space = spaces.Tuple((
                    spaces.Discrete(self.n_agents),  # number of agents
                    spaces.Discrete(self.n_st),  # start time 
                
                ))
       
        # termination of episode condition
        self.terminated=False

        # list of action
        self.gk=[]


    def step(self, action):
       if self.terminated==False:
        # substract the dispatched agents from the total number of available agents   
        self.n_agents-=action[0]
       
            

        print("number of available agents",self.n_agents)

        if self.observation_space!=self.n_shifts:
            reward=0
            self.gk.append(action)
            print("observation_space is:",self.observation_space)
            print("reward is:",reward)
            self.observation_space+=1

            return self.observation_space, reward, self.terminated
        
        else:
            self.gk.append(action)
            reward=30
            print("observation_space is:",self.observation_space)
            print("reward is:",reward)
            print(self.gk)
            self.terminated=True
            return self.observation_space, reward, self.terminated

           
    
    def reset(self, seed=None, options=None):
         self.observation_space=1

         self.terminated=False
    
    def render(self, V,mode='human'):
       pass


s=scheduling(n_shifts=10)


# Loop over an episode of 5 shift 
for _ in range(11):

    # pick random number of agent
    if s.n_agents!=0:
         n_ag = random.randint(1, s.n_agents)

    else:
        n_ag=0

    # pick a random time slot
    values = list(s.ts.values())
    random_ts = random.choice(values)


    # step using random action 
    s.step(action=(n_ag,random_ts))


shifts=s.gk

shift_duration = 3 # 2 hours

result = get_agents_schedule(shifts, shift_duration)
print(result)
