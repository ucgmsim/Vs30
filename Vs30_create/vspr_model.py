#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
vspr_data = pd.read_csv('vspr.csv', header=0)
print(vspr_data)


# In[62]:


class Model:
    def __init__(self, vspr_data, name, classifiers, vs30, stDv):
        self.name=name
        self.groupID_name='groupID_'+name
        self.classifiers = classifiers
        self.vs30=vs30
        self.stDv=stDv
        self.data = vspr_data[[self.groupID_name]]
    
    def get_Vs30(self):
        swap_dict=dict(zip(self.classifiers,self.vs30))
        return data.replace(swap_dict)[[self.groupID_name]]
    
    def get_stDv(self):
        swap_dict=dict(zip(self.classifiers,self.stDv))
        return data.replace(swap_dict)[[self.groupID_name]]
    


# In[66]:


# YongCA parameters


classifiers_YongCA = [                                  "01 - Well dissected alpine summits, mountains, etc.",
                                              "02 - Large volcano, high block plateaus, etc.",
                                              "03 - Well dissected, low mountains, etc.",
                                              "04 - Volcanic fan, foot slope of high block plateaus, etc.",
                                              "05 - Dissected plateaus, etc.",
                                              "06 - Basalt lava plain, glaciated plateau, etc.",
                                              "07 - Moderately eroded mountains, lava flow, etc.",
                                              "08 - Desert alluvial slope, volcanic fan, etc.",
                                              "09 - Well eroded plain of weak rocks, etc.",
                                              "10 - Valley, till plain, etc.",
                                              "11 - Eroded plain of weak rocks, etc.",
                                              "12 - Desert plain, delta plain, etc.",
                                              "13 - Incised terrace, etc.",
                                              "14 - Eroded alluvial fan, till plain, etc.",
                                              "15 - Dune, incised terrace, etc.",
                                              "16 - Fluvial plain, alluvial fan, low-lying flat plains, etc."]
Vs30_YongCA = [519,
                   393,
                   547,
                   459,
                   402,
                   345,
                   388,
                   374,
                   497,
                   349,
                   328,
                   297,
                   500, # No data in Yong et al. This is my guess for incised terraces.
                   209,
                   363,
                   246]
stDv_YongCA =[0.3521,  # All sigmas in this vector were chosen
                   0.4161,  # based on Yong et al. (2012) figure 9, the
                   0.4695,  # "bandplot" showing scatter for each category.
                   0.3540,  # The work done to estimate these is contained
                   0.3136,  # in the folder "Yong---digitizing".
                   0.2800,
                   0.4161,
                   0.3249,
                   0.3516,
                   0.2800,
                   0.2736,
                   0.2931,
                   0.5,    # guess
                   0.1749,
                   0.2800,
                   0.2206]


# In[68]:


# AhdiAK
classifiers_AhdiAK = [
    "01_peat",
    "04_fill",
    "05_fluvialEstuarine",
    "06_alluvium",
    "08_lacustrine",
    "09_beachBarDune",
    "10_fan",
    "11_loess",
    "12_outwash",
    "13_floodplain",
    "14_moraineTill",
    "15_undifSed",
    "16_terrace",
    "17_volcanic",
    "18_crystalline" ]

Vs30_AhdiAK = [
    161,
    198,
    239,
    323,
    326,
    339,
    360,
    376,
    399,
    448,
    453,
    455,
    458,
    635,
    750]
  
stDv_AhdiAK = [
    0.522,
    0.314,
    0.867,
    0.365,
    0.135,
    0.647,
    0.338,
    0.380,
    0.305,
    0.432,
    0.512,
    0.545,
    0.761,
    0.995,
    0.641]


# In[67]:


YongCA=Model(vspr_data, 'YongCA',classifiers_YongCA, Vs30_YongCA,stDv_YongCA)
print(YongCA.get_Vs30())
print(YongCA.get_stDv())


# In[ ]:






