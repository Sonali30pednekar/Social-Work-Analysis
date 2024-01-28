#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wbpy
import json
from pprint import pprint


# In[2]:


api = wbpy.IndicatorAPI()

#iso_country_codes = ["GB", "FR", "JP"]
poverty = "SI.DST.FRST.10"

dataset = api.get_dataset(poverty, country_codes=None, date="1980:2012")
dataset


# In[3]:


dataset.as_dict()


# In[4]:


poverty_data = dataset.api_response


# In[5]:


poverty_data


# In[7]:


out_file = open("poverty.json", "w")
json.dump(poverty_data, out_file, indent = 6) 
    
out_file.close()

