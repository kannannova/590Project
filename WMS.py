#!/usr/bin/env python
# coding: utf-8

# ## 590 Topic 8 Assignment
# ## Milestone 3
# ## Kannan Nova

# ### Project Details : 
# The warehouse is a part of supply chain logistics industry where store the goods and bring back it from storage for shipment.  Warehouse has two major core processes such as 
# Put away where download the goods from truck and place it into final location inside warehouse (McCrea, 2016). 
# Picking where take out the goods from final location and stuff into truck for shipping (Lopienski, 2020).
# These two warehouse operations are overly complex and more labor-oriented tasks. The workers manually go there and pick the goods if it is a small and light weight otherwise, they operate forklifts for picking. Sometimes robots are used for picking (Gillman, 2016). If the system is not providing the proper optimal route/shortest path details for picking a particular order, then workers spend more time on walking instead of performing the actual picking job.
# Based on literature reviews, the picking process is optimized by linearly or deterministic or deep learning, but it is dynamic and stochastic because goods can be placed anywhere in warehouse during put away/storing (Janse, 2019).  Thus, these enable to use reinforcement learning (RL) and if system is not providing the shortest optimal path for picking operation, then the most of times, workers spend more time on walking on warehouse instead performing the actual task. 
# This project is based on dynamic environment and if picking process failed, system/agent will learn it from failure for future improvement and it needs the data for environment, state, Q-table, and locations. 
# 
# This project is developed by using reinformcement Q-learning algorithm through python language.
# The warehouse layout location details will be used as states for q-learning. It has three files such as wms.py/580_topic5_milestone3.py, app.py and index.html

# ### Import all necessary libraries

# In[65]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle
import joblib


# ### Read Data From File for creating states for reinforcement Q-learning

# In[67]:


df = pd.read_csv("D:/Nova/Personal/GCU-20210416T232635Z-001/GCU/Admission and course/projthesis/wmscode/wmsdata.csv", header=None)
df.head()


# ### Data Cleaning

# In[68]:


df.isnull().sum()


# In[69]:


print(df.isnull().values.any())


# There is no null and missing value so there is no need of data cleaning.

# ### Data Exploration

# In[70]:


df.describe()


# In[71]:


print(df.shape)


# There is no need of checking corelation and multivarinace because it is not supervised or unsupervised learning.
# It has three types of daata such states data, rewards, and actions 

# ### Data Preprocessing
# * The panda dataframe df data for states has to be converted into numpy array
# * QTable has to created and initialized with zeroes
# * Action data has to be created
# 

# In[72]:


envRows = 30
envColumns = 30
rewards = df.to_numpy() 
actions = ['up', 'right', 'down', 'left']
qValues = np.zeros((envRows, envColumns, 4)) # 4 represents number of actions


# In[73]:


df.plot.box(grid='True')


# ### Build Model

# In[74]:


class RLWMS:
    rewards = df.to_numpy() 
    num_rows, num_cols = rewards.shape
    envRows = num_cols
    envColumns = num_cols
    qValues = np.zeros((envRows, envColumns, 4))
    actions = ['up', 'right', 'down', 'left']    
    cumulative_rewards=[]
    #This method returns current row and column, initiallly it returns a random row and column
    def findStartLocation(self):
        current_row_index = np.random.randint(self.envRows)
        current_column_index = np.random.randint(self.envColumns)
        while self.isTerminalOrWall(current_row_index, current_column_index):
            current_row_index = np.random.randint(self.envRows)
            current_column_index = np.random.randint(self.envColumns)
        return current_row_index, current_column_index
    
    #This method returns whether state is terminal/wall or not
    def isTerminalOrWall(self,current_row_index, current_column_index):
        if self.rewards[current_row_index, current_column_index] == -5.:
            return False
        else:
            return True
        
    #This method returns the next action based on greedy algorithm
    def findNextAction(self,current_row_index, current_column_index, epsilon):
        if np.random.random() < epsilon:
            return np.argmax(self.qValues[current_row_index, current_column_index])
        else: #choose a random action
            return np.random.randint(4)
    
     #This method returns the next location based on action taken
    def findNextLocation(self,current_row_index, current_column_index, action_index):
        new_row_index = current_row_index
        new_column_index = current_column_index
        if self.actions[action_index] == 'up' and current_row_index > 0:
            new_row_index -= 1
        elif self.actions[action_index] == 'right' and current_column_index < self.envColumns - 1:
            new_column_index += 1
        elif self.actions[action_index] == 'down' and current_row_index < self.envRows - 1:
            new_row_index += 1
        elif self.actions[action_index] == 'left' and current_column_index > 0:
            new_column_index -= 1
        return new_row_index, new_column_index
    
    #This method returns the best shortest optimal path from staging area to picking location. 
    def findShortestPath(self,start_row_index, start_column_index):
        if self.isTerminalOrWall(start_row_index, start_column_index):
            return []
        else: 
            current_row_index, current_column_index = start_row_index, start_column_index
            shortestPath = []
            shortestPath.append([current_row_index, current_column_index])
            while not self.isTerminalOrWall(current_row_index, current_column_index):
                self.action_index = self.findNextAction(current_row_index, current_column_index, 1.)
                current_row_index, current_column_index = self.findNextLocation(current_row_index, current_column_index, self.action_index)
                shortestPath.append([current_row_index, current_column_index])
            return shortestPath


# ### Train the model

# In[75]:


discount_factor = 0.9
learning_rate = 0.9
epsilon = 0.9 
num_episodes = 5000
episode_lengths = np.zeros(num_episodes)
episode_rewards = np.zeros(num_episodes)
total = 0
deltaStates = []

wms = RLWMS()
for episode in range(num_episodes): #20000 episodes
    row_index, column_index = wms.findStartLocation()
    # if it is terminal location then it returns immediately
    while not wms.isTerminalOrWall(row_index, column_index):
        wms.action_index = wms.findNextAction(row_index, column_index, epsilon)  
        old_row_index, old_column_index = row_index, column_index 
        row_index, column_index = wms.findNextLocation(row_index, column_index, wms.action_index) 
        wms.reward = wms.rewards[row_index, column_index]
        
        
        total += wms.reward
        wms.cumulative_rewards.append(total)        
        # Update statistics
        episode_rewards[episode] += wms.reward
        episode_lengths[episode] += total  
        
        old_q_value = wms.qValues[old_row_index, old_column_index, wms.action_index]
        temporal_difference = wms.reward + (discount_factor * np.max(wms.qValues[row_index, column_index])) - old_q_value    
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        wms.qValues[old_row_index, old_column_index, wms.action_index] = new_q_value
        
        deltaStates.append(new_q_value)

print('Training complete!')


# ### Evaluation and Validating the model

# In[76]:


path = wms.findShortestPath(28,6)
print(path)


# In[77]:


def color_specific_cell1(x):
    color = 'background-color : green'
    df1 = pd.DataFrame('',index=x.index, columns=x.columns)
    #path = wms.get_shortest_path(6,2)
    for each in path:        
        df1.loc[each[0],each[1]]= color
    return df1
df = pd.DataFrame(wms.rewards)
df.style.apply(color_specific_cell1, axis=None)


# ### Q-Values

# In[78]:


#wms.qValues


# ### Rewards

# In[79]:


#wms.rewards


# ### Packing for Deployment

# In[81]:


picklefile = open("D:/Nova/Personal/GCU-20210416T232635Z-001/GCU/Admission and course/projthesis/wmscode/rlwms.pkl", 'wb')
#pickle the dictionary and write it to file
pickle.dump(wms, picklefile)
#close the file
picklefile.close()


# #### Validating the deployment package

# In[82]:


wmsUnPicle = joblib.load("D:/Nova/Personal/GCU-20210416T232635Z-001/GCU/Admission and course/projthesis/wmscode/rlwms.pkl")
pathUnPicke = wmsUnPicle.findShortestPath(25,17)
pathUnPicke


# In[83]:


def color_specific_cell(x):
    color = 'background-color : green'
    df1 = pd.DataFrame('',index=x.index, columns=x.columns)    
    for each in pathUnPicke:        
        df1.loc[each[0],each[1]]= color
    return df1


# In[84]:


df = pd.DataFrame(wmsUnPicle.rewards)
df.style.apply(color_specific_cell, axis=None)


# ## Conclusion
# * Warehouse layout location data is used for states.
# * This model uses reinforcement Q-learning algorithm and finds out the shortest optimal path.

# In[ ]:




