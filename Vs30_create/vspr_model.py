#!/usr/bin/env python
# coding: utf-8

# In[638]:


import pandas as pd
vspr_data = pd.read_csv('vspr.csv', header=0)


# In[639]:


class Model:
    def __init__(self, vspr_data, lookup, groupID_name):
        
        self.groupID_name=groupID_name
        self.lookup=lookup
        self.orig_data = vspr_data
        self.data = vspr_data[[self.groupID_name]]
    
    def get_Vs30(self):
        swap_dict=dict(zip(self.lookup['groupID'],self.lookup['vs30']))
        tmp = self.data
        return tmp.replace(swap_dict)[[self.groupID_name]]
    
    def get_stDv(self):
        swap_dict=dict(zip(self.lookup['groupID'],self.lookup['stDv']))
        tmp = self.data
        return tmp.replace(swap_dict)[[self.groupID_name]]
    


# In[640]:


# YongCA parameters
YongCA_lookup ={
    "groupID":[
        "01 - Well dissected alpine summits, mountains, etc.",
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
        "16 - Fluvial plain, alluvial fan, low-lying flat plains, etc."
    ],
    "vs30":[
        519,
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
        246
    ],
    "stDv":[
        0.3521,  # All sigmas in this vector were chosen
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
        0.2206
    ]
}



#YongCA_noQ3 conversion

'''
subset(upd_YongCA_noQ3$summary ,select=c("groupID","posteriorVs30","posteriorStDv"))
                                                         groupID posteriorVs30 posteriorStDv
1            01 - Well dissected alpine summits, mountains, etc.      519.0000     0.5000000
2                  02 - Large volcano, high block plateaus, etc.      393.0000     0.5000000
3                       03 - Well dissected, low mountains, etc.      547.0000     0.5000000
4     04 - Volcanic fan, foot slope of high block plateaus, etc.      459.0000     0.5000000
5                                  05 - Dissected plateaus, etc.      323.5040     0.4074310
6                06 - Basalt lava plain, glaciated plateau, etc.      300.6395     0.3065942
7              07 - Moderately eroded mountains, lava flow, etc.      535.7864     0.3807887
8                 08 - Desert alluvial slope, volcanic fan, etc.      514.9706     0.3807887
9                     09 - Well eroded plain of weak rocks, etc.      284.4710     0.3605551
10                                 10 - Valley, till plain, etc.      317.3684     0.3316625
11                         11 - Eroded plain of weak rocks, etc.      266.8702     0.4000000
12                          12 - Desert plain, delta plain, etc.      297.0000     0.5000000
13                                    13 - Incised terrace, etc.      216.6237     0.2549510
14                    14 - Eroded alluvial fan, till plain, etc.      241.8087     0.3074824
15                              15 - Dune, incised terrace, etc.      198.6448     0.2052564
16 16 - Fluvial plain, alluvial fan, low-lying flat plains, etc.      202.1259     0.2068201
'''
YongCA_noQ3_lookup = {
    "groupID":[
        "01 - Well dissected alpine summits, mountains, etc.",
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
        "16 - Fluvial plain, alluvial fan, low-lying flat plains, etc."
    ],
    "vs30":[
        519.0000,
        393.0000,
        547.0000,
        459.0000,
        323.5040,
        300.6395,
        535.7864,
        514.9706,
        284.4710,
        317.3684,
        266.8702,
        297.0000,
        216.6237,
        241.8087,
        198.6448,
        202.1259
    ],
    "stDv":[
        0.5000000,
        0.5000000,
        0.5000000,
        0.5000000,
        0.4074310,
        0.3065942,
        0.3807887,
        0.3807887,
        0.3605551,
        0.3316625,
        0.4000000,
        0.5000000,
        0.2549510,
        0.3074824,
        0.2052564,
        0.2068201
    ]
}


# In[641]:


# AhdiAK
AhdiAK_lookup = {
    "groupID" : [
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
        "18_crystalline"
    ],
    "vs":[
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
        750
    ],
    "stDv":[
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
        0.641
    ]
}

# AhdiAK_noQ3_conversion
'''
subset(upd_AhdiAK_noQ3$summary, select=c("groupID","posteriorVs30","posteriorStDv"))
    groupID posteriorVs30 posteriorStDv
1              01_peat      162.8921     0.3010332
2              04_fill      272.5127     0.2803060
3  05_fluvialEstuarine      199.5024     0.4387537
4          06_alluvium      271.0506     0.2434866
5        08_lacustrine      326.0000     0.5000000
6      09_beachBarDune      204.3740     0.2321970
7               10_fan      246.6138     0.3446012
8             11_loess      472.7494     0.3545621
9           12_outwash      399.0000     0.5000000
10       13_floodplain      197.4728     0.2026298
11      14_moraineTill      453.0000     0.5120000
12         15_undifSed      455.0000     0.5450000
13          16_terrace      335.2795     0.6028869
14         17_volcanic      635.0000     0.9950000
15      18_crystalline      690.9743     0.4460370
'''

AhdiAK_noQ3_lookup = {
    "groupID":[
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
        "18_crystalline",
    ],
    "vs30":[
        162.8921,
        272.5127,
        199.5024,
        271.0506,
        326.0000,
        204.3740,
        246.6138,
        472.7494,
        399.0000,
        197.4728,
        453.0000,
        455.0000,
        335.2795,
        635.0000,
        690.9743
    ],
    "stDv":[
        0.3010332,
        0.2803060,
        0.4387537,
        0.2434866,
        0.5000000,
        0.2321970,
        0.3446012,
        0.3545621,
        0.5000000,
        0.2026298,
        0.5120000,
        0.5450000,
        0.6028869,
        0.9950000,
        0.4460370
    ]
}

# AhdiAK_noQ3_hyb09c conversion


# In[642]:


model_YongCA=Model(vspr_data,YongCA_lookup,'groupID_YongCA')
print(model_YongCA.get_Vs30())
print(model_YongCA.get_stDv())


# In[643]:


model_YongCA_noQ3=Model(vspr_data,YongCA_noQ3_lookup,'groupID_YongCA_noQ3')
print(model_YongCA_noQ3.get_Vs30())
print(model_YongCA_noQ3.get_stDv())


# In[660]:


class Model_HYB09(Model):
    def __init__(self, vspr_data, lookup, groupID_name, adf):
        #lookup is the same as noQ3, but will be altered
        self.adf = adf
        super().__init__(vspr_data, lookup, groupID_name)
        
        for i in range(len(self.adf['slopeUnit'])):
            idx = self.lookup['groupID'].index(self.adf['slopeUnit'][i])
            self.lookup['stDv'][idx]*=self.adf['sigmaReducFac'][i]
            
        
    def adjust_hyb(self, vs30):
        all_lslp = np.log10(self.orig_data[['slp09c']])
        
        for i in range(len(self.adf['slopeUnit'])):
            gID = self.adf['slopeUnit'][i]
      
            l10s_0 = self.adf['log10slope_0'][i]
            l10s_1 = self.adf['log10slope_1'][i]
            l10Vs30_0 = np.log10(self.adf['Vs30_0'][i])
            l10Vs30_1 = np.log10(self.adf['Vs30_1'][i])
  
            #find rows in the data that match the groupID
            locs = self.data.loc[self.data[self.groupID_name] == gID].index.values
            
            lslp = all_lslp.iloc[locs,].to_numpy()
            lslp = np.reshape(lslp, locs.size) #make 1D array
            
            #We have slope vs Vs30 values evenly spread
            x=np.linspace(l10s_0,l10s_1,lslp.size)
            y=np.linspace(l10Vs30_0,l10Vs30_1,lslp.size)
           
            #Fitting lslp to x, and find corresponding y value
            expnt = np.interp(lslp,x,y) #in R, approx(x=x,y=y,xout=lslp,rule=2)
            vs30_to_update = 10**expnt 
            for i, idx in enumerate(locs):
                #updated subset of locs is now copied back to vs30
                vs30.at[idx,self.groupID_name] = vs30_to_update[i]
            
        return vs30
        
    def get_Vs30(self):
        swap_dict=dict(zip(self.lookup['groupID'],self.lookup['vs30']))
        vs30 = super().get_Vs30() #this vs30 is just geo (usual way)
        vs30 = self.adjust_hyb(vs30) #adjusting for HYB09 variation
        return vs30


# In[ ]:





# In[661]:


AhdiAK_noQ3_hyb09c_hybDF = {
    "slopeUnit": [
        "04_fill",
        "05_fluvialEstuarine",
        "06_alluvium",
        "09_beachBarDune"
    ],
    "log10slope_0": [
        -1.85,
        -2.70,
        -3.44,
        -3.56
    ],
    "log10slope_1": [
        -1.22,
        -1.35,
        -0.88,
        -0.93
    ],
    "Vs30_0" : [
        242,
        171,
        252,
        183
    ],
    "Vs30_1" : [
        418,
        228,
        275,
        239
    ],
    "sigmaReducFac" : [
        0.4888,
        0.7103,
        0.9988,
        0.9348
    ]  
}

model_AhdiAK_noQ3_hyb09c = Model_HYB09(vspr_data,AhdiAK_noQ3_lookup,"groupID_AhdiAK", AhdiAK_noQ3_hyb09c_hybDF)

print(model_AhdiAK_noQ3_hyb09c.get_stDv())
print(model_AhdiAK_noQ3_hyb09c.get_Vs30())


# In[ ]:





# In[ ]:




