# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 18:15:54 2020

@author: nhmeyer
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import math as mth
from scipy import stats
from scipy.stats import spearmanr
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.graphics.factorplots as f
import statsmodels.stats.anova as hh
import statsmodels.formula.api as smf
import statsmodels.api as smm
from statsmodels.formula.api import ols

# from__future__import print_function
WristAccuracy_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Wrist accuracy CondS')
WristAccuracy_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Wrist accuracy CondL')
PinchDistance_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='PinchDistance CondS')
PinchDistance_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='PinchDistance CondL')
Embodiment_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Embodiment CondS')
Embodiment_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Embodiment CondL')
SelfConf_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='SelfConf CondS')
SelfConf_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='SelfConf CondL')
MaxPeakVelocity_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocity_r CondS')
MaxPeakVelocity_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocity_r CondL')
MaxLatencyVelocity_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxLatencyvelocityv_r CondS')
MaxLatencyVelocity_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxLatencyvelocityv_r CondL')
MovementDuration_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Movement duration CondS')
MovementDuration_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Movement duration CondL')
MaxDistance_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Max Distance CondS')
MaxDistance_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Max Distance CondL')
DistanceTarget_ = pd.read_excel('BKL_results_260420.xlsx', sheet_name='DistTarget')
MaxDistance_ElbowS = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Max Distance Elbow CondS')
MaxDistance_ElbowL = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Max Distance Elbow CondL')
MaxDistance_ShoulderS = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Max Distance Shoulder CondS')
MaxDistance_ShoulderL = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Max Distance Shoulder CondL')
MaxDistance_TrunkS = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Max Distance Trunk CondS')
MaxDistance_TrunkL = pd.read_excel('BKL_results_260420.xlsx', sheet_name='Max Distance Trunk CondL')
MaxPeakVelocity_ElbowS = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocityrElbow CondS')
MaxPeakVelocity_ElbowL = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocityrElbow CondL')
MaxPeakVelocity_ShoulderS = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakShouldervelocity_r CondS')
MaxPeakVelocity_ShoulderL = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakShouldervelocity_r CondL')
MaxPeakVelocity_TrunkS = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvel_r_Trunk CondS')
MaxPeakVelocity_TrunkL = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvel_r_Trunk CondL')
MaxPeakVelocity_zS = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocity_z CondS')
MaxPeakVelocity_zL = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocity_z CondL')
MaxPeakVelocity_zElbowS = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocityzElbow CondS')
MaxPeakVelocity_zElbowL = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocityzElbow CondL')
MaxPeakVelocity_zShoulderS = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocityzShoulder CondS')
MaxPeakVelocity_zShoulderL = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocityzShoulder CondL')
MaxPeakVelocity_zTrunkS = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocityzTrunk CondS')
MaxPeakVelocity_zTrunkL = pd.read_excel('BKL_results_260420.xlsx', sheet_name='MaxPeakvelocityzTrunk CondL')
StartWrist_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StartWrist CondS')
StartWrist_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StartWrist CondL')
StopWrist_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StopWrist CondS')
StopWrist_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StopWrist CondL')
StartElbow_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StartElbow CondS')
StartElbow_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StartElbow CondL')
StopElbow_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StopElbow CondS')
StopElbow_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StopElbow CondL')
StartShoulder_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StartShoulder CondS')
StartShoulder_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StartShoulder CondL')
StopShoulder_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StopShoulder CondS')
StopShoulder_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StopShoulder CondL')
StartTrunk_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StartTrunk CondS')
StartTrunk_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StartTrunk CondL')
StopTrunk_S = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StopTrunk CondS')
StopTrunk_L = pd.read_excel('BKL_results_260420.xlsx', sheet_name='StopTrunk CondL')
DistTarget = DistanceTarget_.Dist
numberPatientSoFar = 24
# remove bad participant for trunk and shoulder(Y09_BKL-R- bad quality, outliers result)
MaxDistance_TrunkS = MaxDistance_TrunkS.drop(8,axis = 0)
MaxDistance_TrunkL = MaxDistance_TrunkL.drop(8,axis = 0)
MaxPeakVelocity_TrunkS = MaxPeakVelocity_TrunkS.drop(8,axis = 0)
MaxPeakVelocity_TrunkL = MaxPeakVelocity_TrunkL.drop(8,axis = 0)
MaxPeakVelocity_zTrunkS = MaxPeakVelocity_zTrunkS.drop(8,axis = 0)
MaxPeakVelocity_zTrunkL = MaxPeakVelocity_zTrunkL.drop(8,axis = 0)
MaxDistance_ShoulderS = MaxDistance_ShoulderS.drop(8,axis = 0)
MaxDistance_ShoulderL = MaxDistance_ShoulderL.drop(8,axis = 0)
MaxPeakVelocity_ShoulderS = MaxPeakVelocity_ShoulderS.drop(8,axis = 0)
MaxPeakVelocity_ShoulderL = MaxPeakVelocity_ShoulderL.drop(8,axis = 0)
MaxPeakVelocity_zShoulderS = MaxPeakVelocity_zShoulderS.drop(8,axis = 0)
MaxPeakVelocity_zShoulderL = MaxPeakVelocity_zShoulderL.drop(8,axis = 0)
StartShoulder_S = StartShoulder_S.drop(8,axis = 0)
StartShoulder_L = StartShoulder_L.drop(8,axis = 0)
StopShoulder_S = StopShoulder_S.drop(8,axis = 0)
StopShoulder_L = StopShoulder_L.drop(8,axis = 0)
StartTrunk_S = StartTrunk_S.drop(8,axis = 0)
StartTrunk_L = StartTrunk_L.drop(8,axis = 0)
StopTrunk_S = StopTrunk_S.drop(8,axis = 0)
StopTrunk_L = StopTrunk_L.drop(8,axis = 0)

numberPatientSoFarTrunk = numberPatientSoFar-1 # we removed one partticipant
# with angle discrimination (A0,A25 and AM25)

#Order in which the participant did the conditions. In principle it was supposed to be alternated but i did a mistake and did twicestarting with the long condition
#So it is a bit more complicated, but everything can be checked on the experiment sheet of each participant
Order_S_L = np.append(np.arange(0,13,2),np.arange(15,24,2))
Order_L_S = np.append(np.arange(1,13,2),[13,14])
Order_L_S = np.append(Order_L_S,np.arange(16,24,2))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#data visualization
#1. Embodiment
AllEmb = np.empty(0)
AllEmb = np.append(Embodiment_S.Ownership,Embodiment_L.Ownership)
AllEmb2 = np.append(Embodiment_S.Agency,Embodiment_L.Agency)
AllEmb3 = np.append(Embodiment_S.Size,Embodiment_L.Size)
AllEmb4 = np.append(Embodiment_S.Control,Embodiment_L.Control)
AllEmb = np.append(AllEmb,AllEmb2)
AllEmb = np.append(AllEmb,AllEmb3)
AllEmb = np.append(AllEmb,AllEmb4)



plt.figure(dpi = 1200)
data = {'Bias':  AllEmb/7,
        'Conditions': np.tile(np.repeat(['Standard','Long'],numberPatientSoFar),4),
        'Groups': np.repeat(['Ownership','Agency','Size','Control'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Long': "darkgrey",'Standard': "r"}
my_pal2 = {'Long': "grey",'Standard': "r"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')

for i in range(0,len(Embodiment_S.Agency)):
  # plt.plot([-0.20,0.20],[Embodiment_S.Ownership[i]/7,Embodiment_L.Ownership[i]/7],color='lightgrey',linestyle='solid')
  # plt.plot([0.80,1.20],[Embodiment_S.Agency[i]/7,Embodiment_L.Agency[i]/7],color='lightgrey',linestyle='solid')
  # plt.plot([1.80,2.20],[Embodiment_S.Size[i]/7,Embodiment_L.Size[i]/7],color='lightgrey',linestyle='solid')
  # plt.plot([2.80,3.20],[Embodiment_S.Control[i]/7,Embodiment_L.Control[i]/7],color='lightgrey',linestyle='solid')
  plt.scatter(x=[-0.20,0.20,-0.20,0.20,-0.20,0.20], y=[Embodiment_S.Ownership[2]/7,Embodiment_L.Ownership[2]/7,Embodiment_S.Ownership[20]/7,Embodiment_L.Ownership[20]/7,Embodiment_S.Ownership[23]/7,Embodiment_L.Ownership[23]/7],c = 'k',marker = '$*$',s = 60)
  plt.scatter(x=[0.80,1.20,0.80,1.20,0.80,1.20,0.80,1.20], y=[Embodiment_S.Agency[2]/7,Embodiment_L.Agency[2]/7,Embodiment_S.Agency[20]/7,Embodiment_L.Agency[20]/7,Embodiment_S.Agency[11]/7,Embodiment_L.Agency[11]/7,Embodiment_S.Agency[18]/7,Embodiment_L.Agency[18]/7],c = 'k',marker = '$*$',s = 60)
  plt.scatter(x=[1.80,2.20], y=[Embodiment_S.Size[14]/7,Embodiment_L.Size[14]/7],c = 'k',marker = '$*$',s = 60)
  plt.scatter(x=[2.80,3.20,2.80,3.20,2.80,3.20], y=[Embodiment_S.Control[1]/7,Embodiment_L.Control[1]/7,Embodiment_S.Control[6]/7,Embodiment_L.Control[6]/7,Embodiment_S.Control[9]/7,Embodiment_L.Control[9]/7],c = 'k',marker = '$*$',s = 60)

#plt.hlines(1, -1, 5, colors='k', linestyles='dashed')
#plt.plot(np.repeat(['Ownership','Ownership'],numberPatientSofar),[Embodiment_S.Ownership,Embodiment_L.Ownership],color='lightgrey', linestyle='dashed')
plt.title('Bodily self-consciousness') 
plt.ylabel('Normalized Ratings')   

lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('Figure_Paper/BSC_noline',bbox_extra_artists=(lgd,), bbox_inches='tight')



EMB_O = stats.ttest_rel(Embodiment_S.Ownership,Embodiment_L.Ownership,nan_policy='omit')#missing data for participant 24
EMB_A = stats.ttest_rel(Embodiment_S.Agency,Embodiment_L.Agency)
EMB_S = stats.ttest_rel(Embodiment_S.Size,Embodiment_L.Size)
EMB_C = stats.ttest_rel(Embodiment_S.Control,Embodiment_L.Control,nan_policy='omit') #missing data for one participant 19
EMB_ObjDist = stats.ttest_rel(Embodiment_S.ObjectDistance,Embodiment_L.ObjectDistance)
#Check if outliers ( 2*sd)
for i in range(0,len(Embodiment_L.Ownership)):
    if(Embodiment_L.Ownership[i]> 2*np.std(Embodiment_L.Ownership)+np.mean(Embodiment_L.Ownership) or Embodiment_L.Ownership[i]< np.mean(Embodiment_L.Ownership)-2*np.std(Embodiment_L.Ownership)):
         print(i) #2 and 23 are above or below 2*std
    if(Embodiment_S.Ownership[i]> 2*np.std(Embodiment_S.Ownership)+np.mean(Embodiment_S.Ownership) or Embodiment_S.Ownership[i]< np.mean(Embodiment_S.Ownership)-2*np.std(Embodiment_S.Ownership)):
         print(i) #20 is above or below 2*std
 stats.ttest_rel(Embodiment_S.Ownership[np.array([0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22])],Embodiment_L.Ownership[np.array([0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22])],nan_policy='omit')


for i in range(0,len(Embodiment_L.Agency)):
    if(Embodiment_L.Agency[i]> 2*np.std(Embodiment_L.Agency)+np.mean(Embodiment_L.Agency) or Embodiment_L.Agency[i]< np.mean(Embodiment_L.Agency)-2*np.std(Embodiment_L.Agency)):
         print(i) #2 and 20 are above or below 2*std
    if(Embodiment_S.Agency[i]> 2*np.std(Embodiment_S.Agency)+np.mean(Embodiment_S.Agency) or Embodiment_S.Agency[i]< np.mean(Embodiment_S.Agency)-2*np.std(Embodiment_S.Agency)):
         print(i) # 11 and 18 are above or elow 2* std
stats.ttest_rel(Embodiment_S.Agency[np.array([0,1,3,4,5,6,7,8,9,10,12,13,14,15,16,17,19,21,22,23])],Embodiment_L.Agency[np.array([0,1,3,4,5,6,7,8,9,10,12,13,14,15,16,17,19,21,22,23])],nan_policy='omit')
        
for i in range(0,len(Embodiment_L.Size)):
    if(Embodiment_L.Size[i]> 2*np.std(Embodiment_L.Size)+np.mean(Embodiment_L.Size) or Embodiment_L.Size[i]< np.mean(Embodiment_L.Size)-2*np.std(Embodiment_L.Size)):
         print(i)
    if(Embodiment_S.Size[i]> 2*np.std(Embodiment_S.Size)+np.mean(Embodiment_S.Size) or Embodiment_S.Size[i]< np.mean(Embodiment_S.Size)-2*np.std(Embodiment_S.Size)):
         print(i) #14 is above or below 2"std
stats.ttest_rel(Embodiment_S.Size[np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23])],Embodiment_L.Size[np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23])],nan_policy='omit')

for i in range(0,len(Embodiment_L.Control)):
    if(Embodiment_L.Control[i]> 2*np.std(Embodiment_L.Control)+np.mean(Embodiment_L.Control) or Embodiment_L.Control[i]< np.mean(Embodiment_L.Control)-2*np.std(Embodiment_L.Control)):
         print(i) #6 and 9 are above or below 2*std
    if(Embodiment_S.Control[i]> 2*np.std(Embodiment_S.Control)+np.mean(Embodiment_S.Control) or Embodiment_S.Control[i]< np.mean(Embodiment_S.Control)-2*np.std(Embodiment_S.Control)):
         print(i) #1 6 and 9 is above or below 2*std

stats.ttest_rel(Embodiment_S.Control[np.array([0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23])],Embodiment_L.Control[np.array([0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23])],nan_policy='omit')





#divide size perception according to their order 

AllEmb3 = np.append(Embodiment_S.Size,Embodiment_L.Size)


import statsmodels.formula.api as smf
import statsmodels.api as smm
from statsmodels.formula.api import ols
MyRegression = {'Perception': np.append(Embodiment_S.Size,Embodiment_L.Size), 
                'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),1),
                'Order': np.tile(np.append(np.append(np.append(np.tile([1,2],7),[2]),np.tile([1,2],4)),1),2),
        }
df = pd.DataFrame(MyRegression,columns=['Perception','Conditions','Order'])

model = ols('Perception ~ C(Conditions)+C(Order)+C(Conditions)*C(Order)', data = df,missing='drop')
modeL = model.fit()
modeL.summary()
res = smm.stats.anova_lm(modeL, typ= 2)
mc = MultiComparison(df['Bias'], df['Conditions'],df['Order'])
result = mc.tukeyhsd()
print(result)
print(mc.groupsunique)
f.interaction_plot(df['Conditions'], df['Order'],df['Bias'],colors=['red','blue'], markers=['D','^'], ms=10)
stats.ttest_ind(Embodiment_S.Size[Order_S_L],Embodiment_S.Size[Order_L_S],nan_policy='omit')
stats.ttest_ind(Embodiment_L.Size[Order_S_L],Embodiment_L.Size[Order_L_S],nan_policy='omit')
stats.ttest_ind(Embodiment_S.Size[Order_S_L],Embodiment_L.Size[Order_S_L],nan_policy='omit')
stats.ttest_ind(Embodiment_S.Size[Order_L_S],Embodiment_L.Size[Order_L_S],nan_policy='omit')
plt.figure(dpi = 1200)

data = {'Bias':  AllEmb3,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),1),
         'Order': np.tile(np.append(np.append(np.append(np.tile(['Standard1','Standard2'],7),['Standard2']),np.tile(['Standard1','Standard2'],4)),'Standard1'),2)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Order'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Order", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
'''
for i in range(0,len(Embodiment_S.Agency)):
  plt.plot([-0.20,0.20],[Embodiment_S.Ownership[i],Embodiment_L.Ownership[i]],color='lightgrey')
  plt.plot([0.80,1.20],[Embodiment_S.Agency[i],Embodiment_L.Agency[i]],color='lightgrey')
  plt.plot([1.80,2.20],[Embodiment_S.Size[i],Embodiment_L.Size[i]],color='lightgrey')

'''
pp = sns.boxplot(y='Bias', x='Order', data=df, palette=my_pal,hue='Conditions')
#plt.hlines(1, -1, 5, colors='k', linestyles='dashed')
#plt.plot(np.repeat(['Ownership','Ownership'],numberPatientSofar),[Embodiment_S.Ownership,Embodiment_L.Ownership],color='lightgrey', linestyle='dashed')
plt.title('Size perception depending on the order') 
plt.ylabel('Ratings')   
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('SizePerc_Order',bbox_extra_artists=(lgd,), bbox_inches='tight')



  #stat_compare_means(method = "t.test")
#stat_compare_means(method = "anova")+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#2. Object Distance
ObjDist = np.append(Embodiment_S.ObjectDistance/(100*DistTarget),Embodiment_L.ObjectDistance/(100*DistTarget))
plt.figure(dpi = 1200)
data = {'Bias':  ObjDist,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
        'Groups': np.repeat(['Distance'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "darkgrey",'Standard': "r"}
my_pal2 = {'Elongated': "grey",'Standard': "r"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(Embodiment_S.Agency)):
  plt.plot([-0.20,0.20],[Embodiment_S.ObjectDistance[i]/(100*DistTarget[i]),Embodiment_L.ObjectDistance[i]/(100*DistTarget[i])],color='lightgrey')
plt.scatter(x=[-0.20,0.20,-0.20,0.20], y=[Embodiment_S.ObjectDistance[2]/(100*DistTarget[i]),Embodiment_L.ObjectDistance[2]/(100*DistTarget[i]),Embodiment_S.ObjectDistance[19]/(100*DistTarget[i]),Embodiment_L.ObjectDistance[19]/(100*DistTarget[i])],c = 'k',marker = '$*$',s = 60)

plt.title('Perceived target distance') 
plt.ylabel('Perceived target distance [cm]')   
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('Figure_Paper/Object Distance',bbox_extra_artists=(lgd,), bbox_inches='tight')

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#2. Object Size
ind_objSize_S = np.empty
ind_objSize_L = np.empty
ind_objDist_S = np.empty
ind_objDist_L = np.empty
ObjSize = np.append(Embodiment_S.ObjectSize,Embodiment_L.ObjectSize)
plt.figure(dpi = 1200)
data = {'Bias':  ObjSize,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
        'Groups': np.repeat(['Distance'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "darkgrey",'Standard': "r"}
my_pal2 = {'Elongated': "grey",'Standard': "r"}
plt.scatter(x=[-0.20,0.20], y=[Embodiment_S.ObjectSize[1],Embodiment_L.ObjectSize[1]],c = 'k',marker = '$*$',s = 60)

ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')

for i in range(0,len(Embodiment_S.ObjectSize)):
  plt.plot([-0.20,0.20],[Embodiment_S.ObjectSize[i],Embodiment_L.ObjectSize[i]],color='lightgrey')

plt.scatter(x=[-0.20,0.20], y=[Embodiment_S.ObjectSize[1],Embodiment_L.ObjectSize[1]],c = 'k',marker = '$*$',s = 60)
plt.title('Perceived target size') 
plt.ylabel('Perceived dimension [cm]')   
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('Figure_Paper/Object Size',bbox_extra_artists=(lgd,), bbox_inches='tight')
for i in range(0,len(Embodiment_S.ObjectSize)):
    if(Embodiment_S.ObjectSize[i]> 2*np.std(Embodiment_S.ObjectSize)+np.mean(Embodiment_S.ObjectSize) or Embodiment_S.ObjectSize[i]< np.mean(Embodiment_S.ObjectSize)-2*np.std(Embodiment_S.ObjectSize)):
        print(i)# index 1 is above/below
        ObjSize_S_stat = np.delete(Embodiment_S.ObjectSize, i)
        ObjSize_L_stat = np.delete(Embodiment_L.ObjectSize, i, axis=0)
        ind_objSize_S = np.append(ind_objSize_S,i)
       #since it is paired ttest we need to remove the pair  
for i in range(0,len(Embodiment_L.ObjectSize)):
    if(Embodiment_L.ObjectSize[i]> 2*np.std(Embodiment_L.ObjectSize)+np.mean(Embodiment_L.ObjectSize) or Embodiment_L.ObjectSize[i]< np.mean(Embodiment_L.ObjectSize)-2*np.std(Embodiment_L.ObjectSize)):
        print(i)
        ind_objSize_L = np.append(ind_objSize_L,i)
ObjSize_S_stat = np.delete(Embodiment_S.ObjectSize, ind_objSize_L, axis=0)
ObjSize_L_stat = np.delete(Embodiment_L.ObjectSize, ind_objSize_L, axis=0)       #since it is paired ttest we need to remove the pair  
stats.ttest_rel(Embodiment_S.ObjectSize[np.append(0,np.arange(2,numberPatientSoFar,1))],Embodiment_L.ObjectSize[np.append(0,np.arange(2,numberPatientSoFar,1))])
for i in range(0,len(Embodiment_S.ObjectSize)):
    if(Embodiment_S.ObjectDistance[i]> 2*np.std(Embodiment_S.ObjectDistance)+np.mean(Embodiment_S.ObjectDistance) or Embodiment_S.ObjectDistance[i]< np.mean(Embodiment_S.ObjectDistance)-2*np.std(Embodiment_S.ObjectDistance)):
         print(i)# index 2 and 19 is above/below
for i in range(0,len(Embodiment_L.ObjectSize)):
    if(Embodiment_L.ObjectDistance[i]> 2*np.std(Embodiment_L.ObjectDistance)+np.mean(Embodiment_L.ObjectDistance) or Embodiment_L.ObjectDistance[i]< np.mean(Embodiment_L.ObjectDistance)-2*np.std(Embodiment_L.ObjectDistance)):
         print(i)
stats.ttest_rel(Embodiment_S.ObjectDistance[np.append(np.append(np.array([0,1]),np.arange(3,19,1)),np.arange(20,numberPatientSoFar,1))],Embodiment_L.ObjectDistance[np.append(np.append(np.array([0,1]),np.arange(3,19,1)),np.arange(20,numberPatientSoFar,1))])

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#3. Max Distance
AllDist = np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget)
plt.figure(dpi = 1200)
data = {'Bias':  AllDist,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
        'Groups': np.repeat(['MaxDistance'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "darkgrey",'Standard': "r"}
my_pal2 = {'Elongated': "grey",'Standard': "r"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxDistance_S.MeanMaxDistancev_r)):
  plt.plot([-0.20,0.20],[MaxDistance_S.MeanMaxDistancev_r[i]/DistTarget[i],MaxDistance_L.MeanMaxDistancev_r[i]/DistTarget[i]],color='lightgrey')

plt.title('MaxDistance') 
plt.ylabel('Normalized Reaching Distance')   
plt.legend(loc = 'upper right')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('Figure_Paper/MaxDistance',bbox_extra_artists=(lgd,), bbox_inches='tight')
#MaxDist_Stat = stats.ttest_rel(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget)

#3.1 With Angle

AllDist = np.append(MaxDistance_S.MeanMaxDistanceA0v_r/DistTarget,MaxDistance_L.MeanMaxDistanceA0v_r/DistTarget)
AllDist2 = np.append(MaxDistance_S.MeanMaxDistanceA25v_r/DistTarget,MaxDistance_L.MeanMaxDistanceA25v_r/DistTarget)
AllDist3 = np.append(MaxDistance_S.MeanMAxDistanceAM25v_r/DistTarget,MaxDistance_L.MeanMAxDistanceAM25v_r/DistTarget)
AllDist = np.append(AllDist,AllDist2)
AllDist = np.append(AllDist,AllDist3)
plt.figure(dpi = 1200)
data = {'Bias':  AllDist,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "darkgrey",'Standard': "r"}
my_pal2 = {'Elongated': "grey",'Standard': "r"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxDistance_S.MeanMaxDistancev_r)):
  plt.plot([-0.20,0.20],[MaxDistance_S.MeanMaxDistanceA0v_r[i]/DistTarget[i],MaxDistance_L.MeanMaxDistanceA0v_r[i]/DistTarget[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxDistance_S.MeanMaxDistanceA25v_r[i]/DistTarget[i],MaxDistance_L.MeanMaxDistanceA25v_r[i]/DistTarget[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxDistance_S.MeanMAxDistanceAM25v_r[i]/DistTarget[i],MaxDistance_L.MeanMAxDistanceAM25v_r[i]/DistTarget[i]],color='lightgrey')

plt.title('MaxDistance Angle') 
plt.ylabel('Normalized Reaching Distance')   
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('Figure_Paper/MaxDistanceAngle',bbox_extra_artists=(lgd,), bbox_inches='tight')

MaxDist_Stat_A0 = stats.ttest_rel(MaxDistance_S.MeanMaxDistanceA0v_r/DistTarget,MaxDistance_L.MeanMaxDistanceA0v_r/DistTarget)
MaxDist_Stat_A25 = stats.ttest_rel(MaxDistance_S.MeanMaxDistanceA25v_r/DistTarget,MaxDistance_L.MeanMaxDistanceA25v_r/DistTarget)
MaxDist_Stat_AM25 = stats.ttest_rel(MaxDistance_S.MeanMAxDistanceAM25v_r/DistTarget,MaxDistance_L.MeanMAxDistanceAM25v_r/DistTarget)
# ELBOW
#3. Max Distance
AllDist = np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget)
plt.figure(dpi = 1200)
data = {'Bias':  AllDist,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
        'Groups': np.repeat(['MaxDistanceElbow'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r)):
  plt.plot([-0.20,0.20],[MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r[i]/DistTarget[i],MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r[i]/DistTarget[i]],color='lightgrey')

plt.title('MaxDistanceElbow') 
plt.ylabel('Normalized Reaching Distance')   
plt.legend(loc = 'upper right')
plt.savefig('MaxDistanceElbow')
#MaxDist_Stat = stats.ttest_rel(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget)

#3.1 With Angle

AllDist = np.append(MaxDistance_ElbowS.MeanMaxDistanceA0_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistanceA0_Elbowv_r/DistTarget)
AllDist2 = np.append(MaxDistance_ElbowS.MeanMaxDistanceA25_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistanceA25_Elbowv_r/DistTarget)
AllDist3 = np.append(MaxDistance_ElbowS.MeanMAxDistanceAM25_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMAxDistanceAM25_Elbowv_r/DistTarget)
AllDist = np.append(AllDist,AllDist2)
AllDist = np.append(AllDist,AllDist3)
plt.figure(dpi = 1200)
data = {'Bias':  AllDist,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r)):
  plt.plot([-0.20,0.20],[MaxDistance_ElbowS.MeanMaxDistanceA0_Elbowv_r[i]/DistTarget[i],MaxDistance_ElbowL.MeanMaxDistanceA0_Elbowv_r[i]/DistTarget[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxDistance_ElbowS.MeanMaxDistanceA25_Elbowv_r[i]/DistTarget[i],MaxDistance_ElbowL.MeanMaxDistanceA25_Elbowv_r[i]/DistTarget[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxDistance_ElbowS.MeanMAxDistanceAM25_Elbowv_r[i]/DistTarget[i],MaxDistance_ElbowL.MeanMAxDistanceAM25_Elbowv_r[i]/DistTarget[i]],color='lightgrey')

plt.title('MaxDistanceElbow Angle') 
plt.ylabel('Normalized Reaching Distance Elbow')   
plt.legend(loc = 'upper right')
plt.savefig('MaxDistanceElbow Angle')
#MaxDist_Stat_A0 = stats.ttest_rel(MaxDistance_ElbowS.MeanMaxDistanceA0_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistanceA0_Elbowv_r/DistTarget)
#MaxDist_Stat_A25 = stats.ttest_rel(MaxDistance_ElbowS.MeanMaxDistanceA25_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistanceA25_Elbowv_r/DistTarget)
#MaxDist_Stat_AM25 = stats.ttest_rel(MaxDistance_ElbowS.MeanMAxDistanceAM25_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMAxDistanceAM25_Elbowv_r/DistTarget)
#Since there is a difference of velocity at the angle M25, we would like to check if there is a difference in the distance as well
MyElbow = {'ElbowAngle': np.append(np.append(np.append(MaxDistance_ElbowS.MeanMaxDistanceA0_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistanceA0_Elbowv_r/DistTarget),np.append(MaxDistance_ElbowS.MeanMaxDistanceA25_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistanceA25_Elbowv_r/DistTarget)),np.append(MaxDistance_ElbowS.MeanMAxDistanceAM25_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMAxDistanceAM25_Elbowv_r/DistTarget)), 
           'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
           'Angle':np.repeat(['A0','A25','AM25'],numberPatientSoFar*2)
                        }
df = pd.DataFrame(MyElbow,columns=['ElbowAngle','Conditions','Angle'])

model = ols('ElbowAngle ~ C(Conditions)+C(Angle)', data = df,missing='drop')
modeL = model.fit()
modeL.summary()
res = smm.stats.anova_lm(modeL, typ= 2)
mc = MultiComparison(df['ElbowAngle'], df['Angle'])
result = mc.tukeyhsd()
print(result)
print(mc.groupsunique)
f.interaction_plot(df['Angle'],df['Conditions'],df['ElbowAngle'],colors=['red','blue'], markers=['D','^'], ms=10)

# SHOULDER
#3. Max Distance
AllDist = np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget)
plt.figure(dpi = 1200)
data = {'Bias':  AllDist,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
        'Groups': np.repeat(['MaxDistanceShoulder'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar)) #here we remove the participant 9 from the stats because of bad quality data

for i in iteration:
  plt.plot([-0.20,0.20],[MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r[i]/DistTarget[i],MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r[i]/DistTarget[i]],color='lightgrey')

plt.title('MaxDistanceShoulder') 
plt.ylabel('Normalized Reaching Distance')   
plt.legend(loc = 'upper right')
plt.savefig('MaxDistanceShoulder')
#MaxDist_Stat = stats.ttest_rel(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget)

#3.1 With Angle

AllDist = np.append(MaxDistance_ShoulderS.MeanMaxDistanceA0_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistanceA0_Shoulderv_r/DistTarget)
AllDist2 = np.append(MaxDistance_ShoulderS.MeanMaxDistanceA25_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistanceA25_Shoulderv_r/DistTarget)
AllDist3 = np.append(MaxDistance_ShoulderS.MeanMAxDistanceAM25_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMAxDistanceAM25_Shoulderv_r/DistTarget)
AllDist = np.append(AllDist,AllDist2)
AllDist = np.append(AllDist,AllDist3)
plt.figure(dpi = 1200)
data = {'Bias':  AllDist,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar)) #here we remove the participant 9 from the stats because of bad quality data

for i in iteration:
  plt.plot([-0.20,0.20],[MaxDistance_ShoulderS.MeanMaxDistanceA0_Shoulderv_r[i]/DistTarget[i],MaxDistance_ShoulderL.MeanMaxDistanceA0_Shoulderv_r[i]/DistTarget[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxDistance_ShoulderS.MeanMaxDistanceA25_Shoulderv_r[i]/DistTarget[i],MaxDistance_ShoulderL.MeanMaxDistanceA25_Shoulderv_r[i]/DistTarget[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxDistance_ShoulderS.MeanMAxDistanceAM25_Shoulderv_r[i]/DistTarget[i],MaxDistance_ShoulderL.MeanMAxDistanceAM25_Shoulderv_r[i]/DistTarget[i]],color='lightgrey')

plt.title('MaxDistanceShoulder Angle') 
plt.ylabel('Normalized Reaching Distance Shoulder')   
plt.legend(loc = 'upper right')
plt.savefig('MaxDistanceShoulder Angle')
#MaxDist_Stat_A0 = stats.ttest_rel(MaxDistance_ShoulderS.MeanMaxDistanceA0_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistanceA0_Shoulderv_r/DistTarget,nan_policy = 'omit')
#MaxDist_Stat_A25 = stats.ttest_rel(MaxDistance_ShoulderS.MeanMaxDistanceA25_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistanceA25_Shoulderv_r/DistTarget,nan_policy = 'omit')
#MaxDist_Stat_AM25 = stats.ttest_rel(MaxDistance_ShoulderS.MeanMAxDistanceAM25_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMAxDistanceAM25_Shoulderv_r/DistTarget,nan_policy = 'omit')


#TRUNK
#3. Max Distance
AllDist = np.append(MaxDistance_TrunkS.MeanMaxDistance_Trunkv_r,MaxDistance_TrunkL.MeanMaxDistance_Trunkv_r)
plt.figure(dpi = 1200)
data = {'Bias':  AllDist,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFarTrunk),
        'Groups': np.repeat(['MaxDistanceTrunk'],numberPatientSoFarTrunk*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar))
for i in iteration:#range(0,len(MaxDistance_TrunkS.MeanMaxDistance_Trunkv_r)):
    print(i)
    plt.plot([-0.20,0.20],[MaxDistance_TrunkS.MeanMaxDistance_Trunkv_r[i],MaxDistance_TrunkL.MeanMaxDistance_Trunkv_r[i]],color='lightgrey')

plt.title('MaxDistanceTrunk') 
plt.ylabel('Distance Trunk')   
plt.legend(loc = 'upper right')
plt.savefig('MaxDistanceTrunk')
#MaxDist_Stat = stats.ttest_rel(MaxDistance_TrunkS.MeanMaxDistance_Trunkv_r,MaxDistance_TrunkL.MeanMaxDistance_Trunkv_r)

#3.1 With Angle

AllDist = np.append(MaxDistance_TrunkS.MeanMaxDistanceA0_Trunkv_r,MaxDistance_TrunkL.MeanMaxDistanceA0_Trunkv_r)
AllDist2 = np.append(MaxDistance_TrunkS.MeanMaxDistanceA25_Trunkv_r,MaxDistance_TrunkL.MeanMaxDistanceA25_Trunkv_r)
AllDist3 = np.append(MaxDistance_TrunkS.MeanMAxDistanceAM25_Trunkv_r,MaxDistance_TrunkL.MeanMAxDistanceAM25_Trunkv_r)
AllDist = np.append(AllDist,AllDist2)
AllDist = np.append(AllDist,AllDist3)
plt.figure(dpi = 1200)
data = {'Bias':  AllDist,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFarTrunk),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFarTrunk*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar))
for i in iteration:#range(0,len(MaxDistance_TrunkS.MeanMaxDistance_Trunkv_r)):
  plt.plot([-0.20,0.20],[MaxDistance_TrunkS.MeanMaxDistanceA0_Trunkv_r[i],MaxDistance_TrunkL.MeanMaxDistanceA0_Trunkv_r[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxDistance_TrunkS.MeanMaxDistanceA25_Trunkv_r[i],MaxDistance_TrunkL.MeanMaxDistanceA25_Trunkv_r[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxDistance_TrunkS.MeanMAxDistanceAM25_Trunkv_r[i],MaxDistance_TrunkL.MeanMAxDistanceAM25_Trunkv_r[i]],color='lightgrey')

plt.title('MaxDistanceTrunk Angle') 
plt.ylabel('Distance Trunk')   
plt.legend(loc = 'upper right')
plt.savefig('MaxDistanceTrunk Angle')
#MaxDist_Stat_A0 = stats.ttest_rel(MaxDistance_TrunkS.MeanMaxDistanceA0_Trunkv_r,MaxDistance_TrunkL.MeanMaxDistanceA0_Trunkv_r)
#MaxDist_Stat_A25 = stats.ttest_rel(MaxDistance_TrunkS.MeanMaxDistanceA25_Trunkv_r,MaxDistance_TrunkL.MeanMaxDistanceA25_Trunkv_r)
#MaxDist_Stat_AM25 = stats.ttest_rel(MaxDistance_TrunkS.MeanMAxDistanceAM25_Trunkv_r,MaxDistance_TrunkL.MeanMAxDistanceAM25_Trunkv_r)

#-----------------------------------------------------------------------------------------------------------------------------------------
#4. Max Peak
AllVelocity = np.append(MaxPeakVelocity_S.MaxPeakv_r,MaxPeakVelocity_L.MaxPeakv_r)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
      'Groups': np.repeat(['Velocity'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxPeakVelocity_L.MaxPeakA0v_r)):
  plt.plot([-0.20,0.20],[MaxPeakVelocity_S.MaxPeakv_r[i],MaxPeakVelocity_L.MaxPeakv_r[i]],color='lightgrey')

plt.title('MaxVelocity') 
plt.ylabel('Velocity [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocity')
#MaxPeak_Stat = stats.ttest_rel(MaxPeakVelocity_S.MaxPeakv_r,MaxPeakVelocity_L.MaxPeakv_r)

#3.1 With Angle

AllVelocity = np.append(MaxPeakVelocity_S.MaxPeakA0v_r,MaxPeakVelocity_L.MaxPeakA0v_r)
AllVelocity2 = np.append(MaxPeakVelocity_S.MaxPeakA25v_r,MaxPeakVelocity_L.MaxPeakA25v_r)
AllVelocity3 = np.append(MaxPeakVelocity_S.MaxPeakAM25v_r,MaxPeakVelocity_L.MaxPeakAM25v_r)
AllVelocity = np.append(AllVelocity,AllVelocity2)
AllVelocity = np.append(AllVelocity,AllVelocity3)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxPeakVelocity_L.MaxPeakA0v_r)):
  plt.plot([-0.20,0.20],[MaxPeakVelocity_S.MaxPeakA0v_r[i],MaxPeakVelocity_L.MaxPeakA0v_r[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxPeakVelocity_S.MaxPeakA25v_r[i],MaxPeakVelocity_L.MaxPeakA25v_r[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxPeakVelocity_S.MaxPeakAM25v_r[i],MaxPeakVelocity_L.MaxPeakAM25v_r[i]],color='lightgrey')

plt.title('Max Velocity Angle') 
plt.ylabel('Velocity  [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocity Angle')
#MaxPeak_Stat_A0 = stats.ttest_rel(MaxPeakVelocity_S.MaxPeakA0v_r,MaxPeakVelocity_L.MaxPeakA0v_r)
#MaxPeak_Stat_A25 = stats.ttest_rel(MaxPeakVelocity_S.MaxPeakA25v_r,MaxPeakVelocity_L.MaxPeakA25v_r)
#MaxPeak_Stat_AM25 = stats.ttest_rel(MaxPeakVelocity_S.MaxPeakAM25v_r,MaxPeakVelocity_L.MaxPeakAM25v_r)
#Velocityz Wrist
#4. Max Peak
AllVelocity = np.append(MaxPeakVelocity_zS.MaxPeakv_z,MaxPeakVelocity_zL.MaxPeakv_z)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
      'Groups': np.repeat(['Velocity'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxPeakVelocity_zL.MaxPeakA0v_z)):
  plt.plot([-0.20,0.20],[MaxPeakVelocity_zS.MaxPeakv_z[i],MaxPeakVelocity_zL.MaxPeakv_z[i]],color='lightgrey')

plt.title('MaxVelocityz') 
plt.ylabel('Velocity z[m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocityz')
#MaxPeak_Stat = stats.ttest_rel(MaxPeakVelocity_zS.MaxPeakv_z,MaxPeakVelocity_zL.MaxPeakv_z)

#3.1 With Angle

AllVelocity = np.append(MaxPeakVelocity_zS.MaxPeakA0v_z,MaxPeakVelocity_zL.MaxPeakA0v_z)
AllVelocity2 = np.append(MaxPeakVelocity_zS.MaxPeakA25v_z,MaxPeakVelocity_zL.MaxPeakA25v_z)
AllVelocity3 = np.append(MaxPeakVelocity_zS.MaxPeakAM25v_z,MaxPeakVelocity_zL.MaxPeakAM25v_z)
AllVelocity = np.append(AllVelocity,AllVelocity2)
AllVelocity = np.append(AllVelocity,AllVelocity3)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxPeakVelocity_zL.MaxPeakA0v_z)):
  plt.plot([-0.20,0.20],[MaxPeakVelocity_zS.MaxPeakA0v_z[i],MaxPeakVelocity_zL.MaxPeakA0v_z[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxPeakVelocity_zS.MaxPeakA25v_z[i],MaxPeakVelocity_zL.MaxPeakA25v_z[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxPeakVelocity_zS.MaxPeakAM25v_z[i],MaxPeakVelocity_zL.MaxPeakAM25v_z[i]],color='lightgrey')

plt.title('Max Velocity z Angle') 
plt.ylabel('Velocity  z [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocity z Angle')
#MaxPeak_Stat_A0 = stats.ttest_rel(MaxPeakVelocity_zS.MaxPeakA0v_z,MaxPeakVelocity_zL.MaxPeakA0v_z)
#MaxPeak_Stat_A25 = stats.ttest_rel(MaxPeakVelocity_zS.MaxPeakA25v_z,MaxPeakVelocity_zL.MaxPeakA25v_z)
#MaxPeak_Stat_AM25 = stats.ttest_rel(MaxPeakVelocity_zS.MaxPeakAM25v_z,MaxPeakVelocity_zL.MaxPeakAM25v_z)

#ELBOW
#4. Max Peak
AllVelocity = np.append(MaxPeakVelocity_zElbowS.MaxPeakElbowv_z,MaxPeakVelocity_zElbowL.MaxPeakElbowv_z)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
      'Groups': np.repeat(['Velocity'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxPeakVelocity_zElbowL.MaxPeakElbowA0v_z)):
  plt.plot([-0.20,0.20],[MaxPeakVelocity_zElbowS.MaxPeakElbowv_z[i],MaxPeakVelocity_zElbowL.MaxPeakElbowv_z[i]],color='lightgrey')

plt.title('MaxVelocityElbowz') 
plt.ylabel('Velocity Elbow z[m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocityElbowz')
#MaxPeak_Stat = stats.ttest_rel(MaxPeakVelocity_zElbowS.MaxPeakElbowv_z,MaxPeakVelocity_zElbowL.MaxPeakElbowv_z)

#radialvelocity
AllVelocity = np.append(MaxPeakVelocity_ElbowS.MaxPeakElbowv_r,MaxPeakVelocity_ElbowL.MaxPeakElbowv_r)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
      'Groups': np.repeat(['Velocity'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxPeakVelocity_ElbowL.MaxPeakElbowA0v_r)):
  plt.plot([-0.20,0.20],[MaxPeakVelocity_ElbowS.MaxPeakElbowv_r[i],MaxPeakVelocity_ElbowL.MaxPeakElbowv_r[i]],color='lightgrey')

plt.title('MaxVelocityElbow') 
plt.ylabel('Velocity Elbow [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocityElbow')
#MaxPeak_Stat = stats.ttest_rel(MaxPeakVelocity_ElbowS.MaxPeakElbowv_r,MaxPeakVelocity_ElbowL.MaxPeakElbowv_r)

#3.1 With Angle

AllVelocity = np.append(MaxPeakVelocity_zElbowS.MaxPeakElbowA0v_z,MaxPeakVelocity_zElbowL.MaxPeakElbowA0v_z)
AllVelocity2 = np.append(MaxPeakVelocity_zElbowS.MaxPeakElbowA25v_z,MaxPeakVelocity_zElbowL.MaxPeakElbowA25v_z)
AllVelocity3 = np.append(MaxPeakVelocity_zElbowS.MaxPeakElbowAM25v_z,MaxPeakVelocity_zElbowL.MaxPeakElbowAM25v_z)
AllVelocity = np.append(AllVelocity,AllVelocity2)
AllVelocity = np.append(AllVelocity,AllVelocity3)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxPeakVelocity_zElbowL.MaxPeakElbowA0v_z)):
  plt.plot([-0.20,0.20],[MaxPeakVelocity_zElbowS.MaxPeakElbowA0v_z[i],MaxPeakVelocity_zElbowL.MaxPeakElbowA0v_z[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxPeakVelocity_zElbowS.MaxPeakElbowA25v_z[i],MaxPeakVelocity_zElbowL.MaxPeakElbowA25v_z[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxPeakVelocity_zElbowS.MaxPeakElbowAM25v_z[i],MaxPeakVelocity_zElbowL.MaxPeakElbowAM25v_z[i]],color='lightgrey')

plt.title('Max Velocity Elbow z Angle') 
plt.ylabel('Velocity Elbow z [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocity Elbowz Angle')
#MaxPeak_Stat_A0 = stats.ttest_rel(MaxPeakVelocity_zElbowS.MaxPeakElbowA0v_z,MaxPeakVelocity_zElbowL.MaxPeakElbowA0v_z)
#MaxPeak_Stat_A25 = stats.ttest_rel(MaxPeakVelocity_zElbowS.MaxPeakElbowA25v_z,MaxPeakVelocity_zElbowL.MaxPeakElbowA25v_z)
#MaxPeak_Stat_AM25 = stats.ttest_rel(MaxPeakVelocity_zElbowS.MaxPeakElbowAM25v_z,MaxPeakVelocity_zElbowL.MaxPeakElbowAM25v_z)

# radial velocity 
AllVelocity = np.append(MaxPeakVelocity_ElbowS.MaxPeakElbowA0v_r,MaxPeakVelocity_ElbowL.MaxPeakElbowA0v_r)
AllVelocity2 = np.append(MaxPeakVelocity_ElbowS.MaxPeakElbowA25v_r,MaxPeakVelocity_ElbowL.MaxPeakElbowA25v_r)
AllVelocity3 = np.append(MaxPeakVelocity_ElbowS.MaxPeakElbowAM25v_r,MaxPeakVelocity_ElbowL.MaxPeakElbowAM25v_r)
AllVelocity = np.append(AllVelocity,AllVelocity2)
AllVelocity = np.append(AllVelocity,AllVelocity3)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxPeakVelocity_ElbowL.MaxPeakElbowA0v_r)):
  plt.plot([-0.20,0.20],[MaxPeakVelocity_ElbowS.MaxPeakElbowA0v_r[i],MaxPeakVelocity_ElbowL.MaxPeakElbowA0v_r[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxPeakVelocity_ElbowS.MaxPeakElbowA25v_r[i],MaxPeakVelocity_ElbowL.MaxPeakElbowA25v_r[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxPeakVelocity_ElbowS.MaxPeakElbowAM25v_r[i],MaxPeakVelocity_ElbowL.MaxPeakElbowAM25v_r[i]],color='lightgrey')

plt.title('Max Velocity Elbow r Angle') 
plt.ylabel('Velocity Elbow r [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocity Elbow Angle')
#MaxPeak_Stat_A0 = stats.ttest_rel(MaxPeakVelocity_ElbowS.MaxPeakElbowA0v_r,MaxPeakVelocity_ElbowL.MaxPeakElbowA0v_r)
#MaxPeak_Stat_A25 = stats.ttest_rel(MaxPeakVelocity_ElbowS.MaxPeakElbowA25v_r,MaxPeakVelocity_ElbowL.MaxPeakElbowA25v_r)
#MaxPeak_Stat_AM25 = stats.ttest_rel(MaxPeakVelocity_ElbowS.MaxPeakElbowAM25v_r,MaxPeakVelocity_ElbowL.MaxPeakElbowAM25v_r)

#Shoulder
#4. Max Peak
AllVelocity = np.append(MaxPeakVelocity_zShoulderS.MaxPeakShoulderv_z,MaxPeakVelocity_zShoulderL.MaxPeakShoulderv_z)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
      'Groups': np.repeat(['Velocity'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar)) #here we remove the participant 9 from the stats because of bad quality data

for i in iteration:
  plt.plot([-0.20,0.20],[MaxPeakVelocity_zShoulderS.MaxPeakShoulderv_z[i],MaxPeakVelocity_zShoulderL.MaxPeakShoulderv_z[i]],color='lightgrey')

plt.title('MaxVelocityShoulderz') 
plt.ylabel('Velocity Shoulder z[m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocityShoulderz')
#MaxPeak_Stat = stats.ttest_rel(MaxPeakVelocity_zShoulderS.MaxPeakShoulderv_z,MaxPeakVelocity_zShoulderL.MaxPeakShoulderv_z)
#radial
AllVelocity = np.append(MaxPeakVelocity_ShoulderS.MaxPeakShoulderv_r,MaxPeakVelocity_ShoulderL.MaxPeakShoulderv_r)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar-1),
      'Groups': np.repeat(['Velocity'],(numberPatientSoFar-1)*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "darkgrey",'Standard': "r"}
my_pal2 = {'Elongated': "grey",'Standard': "r"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar)) #here we remove the participant 9 from the stats because of bad quality data

for i in iteration:
  plt.plot([-0.20,0.20],[MaxPeakVelocity_ShoulderS.MaxPeakShoulderv_r[i],MaxPeakVelocity_ShoulderL.MaxPeakShoulderv_r[i]],color='lightgrey')

plt.title('MaxVelocityShoulder') 
plt.ylabel('Velocity Shoulder [m/s]')   
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('Figure_Paper/MaxVelocityShoulder',bbox_extra_artists=(lgd,), bbox_inches='tight')
#MaxPeak_Stat = stats.ttest_rel(MaxPeakVelocity_zShoulderS.MaxPeakShoulderv_z,MaxPeakVelocity_zShoulderL.MaxPeakShoulderv_z)

#3.1 With Angle

AllVelocity = np.append(MaxPeakVelocity_zShoulderS.MaxPeakShoulderA0v_z,MaxPeakVelocity_zShoulderL.MaxPeakShoulderA0v_z)
AllVelocity2 = np.append(MaxPeakVelocity_zShoulderS.MaxPeakShoulderA25v_z,MaxPeakVelocity_zShoulderL.MaxPeakShoulderA25v_z)
AllVelocity3 = np.append(MaxPeakVelocity_zShoulderS.MaxPeakShoulderAM25v_z,MaxPeakVelocity_zShoulderL.MaxPeakShoulderAM25v_z)
AllVelocity = np.append(AllVelocity,AllVelocity2)
AllVelocity = np.append(AllVelocity,AllVelocity3)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar)) #here we remove the participant 9 from the stats because of bad quality data

for i in iteration:
  plt.plot([-0.20,0.20],[MaxPeakVelocity_zShoulderS.MaxPeakShoulderA0v_z[i],MaxPeakVelocity_zShoulderL.MaxPeakShoulderA0v_z[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxPeakVelocity_zShoulderS.MaxPeakShoulderA25v_z[i],MaxPeakVelocity_zShoulderL.MaxPeakShoulderA25v_z[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxPeakVelocity_zShoulderS.MaxPeakShoulderAM25v_z[i],MaxPeakVelocity_zShoulderL.MaxPeakShoulderAM25v_z[i]],color='lightgrey')

plt.title('Max Velocity Shoulder z Angle') 
plt.ylabel('Velocity Shoulder z [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocity Shoulderz Angle')
#MaxPeak_Stat_A0 = stats.ttest_rel(MaxPeakVelocity_zShoulderS.MaxPeakShoulderA0v_z,MaxPeakVelocity_zShoulderL.MaxPeakShoulderA0v_z)
#MaxPeak_Stat_A25 = stats.ttest_rel(MaxPeakVelocity_zShoulderS.MaxPeakShoulderA25v_z,MaxPeakVelocity_zShoulderL.MaxPeakShoulderA25v_z)
#MaxPeak_Stat_AM25 = stats.ttest_rel(MaxPeakVelocity_zShoulderS.MaxPeakShoulderAM25v_z,MaxPeakVelocity_zShoulderL.MaxPeakShoulderAM25v_z)
#radial
AllVelocity = np.append(MaxPeakVelocity_ShoulderS.MaxPeakShoulderA0v_r,MaxPeakVelocity_ShoulderL.MaxPeakShoulderA0v_r)
AllVelocity2 = np.append(MaxPeakVelocity_ShoulderS.MaxPeakShoulderA25v_r,MaxPeakVelocity_ShoulderL.MaxPeakShoulderA25v_r)
AllVelocity3 = np.append(MaxPeakVelocity_ShoulderS.MaxPeakShoulderAM25v_r,MaxPeakVelocity_ShoulderL.MaxPeakShoulderAM25v_r)
AllVelocity = np.append(AllVelocity,AllVelocity2)
AllVelocity = np.append(AllVelocity,AllVelocity3)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar)) #here we remove the participant 9 from the stats because of bad quality data

for i in iteration:
  plt.plot([-0.20,0.20],[MaxPeakVelocity_ShoulderS.MaxPeakShoulderA0v_r[i],MaxPeakVelocity_ShoulderL.MaxPeakShoulderA0v_r[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxPeakVelocity_ShoulderS.MaxPeakShoulderA25v_r[i],MaxPeakVelocity_ShoulderL.MaxPeakShoulderA25v_r[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxPeakVelocity_ShoulderS.MaxPeakShoulderAM25v_r[i],MaxPeakVelocity_ShoulderL.MaxPeakShoulderAM25v_r[i]],color='lightgrey')

plt.title('Max Velocity Shoulder  Angle') 
plt.ylabel('Velocity Shoulder  [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocity Shoulder Angle')
#MaxPeak_Stat_A0 = stats.ttest_rel(MaxPeakVelocity_ShoulderS.MaxPeakShoulderA0v_r,MaxPeakVelocity_ShoulderL.MaxPeakShoulderA0v_r)
#MaxPeak_Stat_A25 = stats.ttest_rel(MaxPeakVelocity_ShoulderS.MaxPeakShoulderA25v_r,MaxPeakVelocity_ShoulderL.MaxPeakShoulderA25v_r)
#MaxPeak_Stat_AM25 = stats.ttest_rel(MaxPeakVelocity_ShoulderS.MaxPeakShoulderAM25v_r,MaxPeakVelocity_ShoulderL.MaxPeakShoulderAM25v_r)

#Trunk
#4. Max Peak
AllVelocity = np.append(MaxPeakVelocity_zTrunkS.MaxPeakTrunkv_z,MaxPeakVelocity_zTrunkL.MaxPeakTrunkv_z)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFarTrunk),
      'Groups': np.repeat(['Velocity'],numberPatientSoFarTrunk*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar)) #here we remove the participant 9 from the stats because of bad quality data

for i in iteration: #np.array([0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]):
  print(i)
  plt.plot([-0.20,0.20],[MaxPeakVelocity_zTrunkS.MaxPeakTrunkv_z[i],MaxPeakVelocity_zTrunkL.MaxPeakTrunkv_z[i]],color='lightgrey')

plt.title('MaxVelocityTrunkz') 
plt.ylabel('Velocity Trunk z[m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocityTrunkz')
#MaxPeak_Stat = stats.ttest_rel(MaxPeakVelocity_zTrunkS.MaxPeakTrunkv_z,MaxPeakVelocity_zTrunkL.MaxPeakTrunkv_z)
#radial
AllVelocity = np.append(MaxPeakVelocity_TrunkS.MaxPeakTrunkv_r,MaxPeakVelocity_TrunkL.MaxPeakTrunkv_r)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFarTrunk),
      'Groups': np.repeat(['Velocity'],numberPatientSoFarTrunk*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar)) #here we remove the participant 9 from the stats because of bad quality data

for i in iteration:
  plt.plot([-0.20,0.20],[MaxPeakVelocity_TrunkS.MaxPeakTrunkv_r[i],MaxPeakVelocity_TrunkL.MaxPeakTrunkv_r[i]],color='lightgrey')

plt.title('MaxVelocityTrunk') 
plt.ylabel('Velocity Trunk [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocityTrunk')
#MaxPeak_Stat = stats.ttest_rel(MaxPeakVelocity_zTrunkS.MaxPeakTrunkv_z,MaxPeakVelocity_zTrunkL.MaxPeakTrunkv_z)

#3.1 With Angle

AllVelocity = np.append(MaxPeakVelocity_zTrunkS.MaxPeakTrunkA0v_z,MaxPeakVelocity_zTrunkL.MaxPeakTrunkA0v_z)
AllVelocity2 = np.append(MaxPeakVelocity_zTrunkS.MaxPeakTrunkA25v_z,MaxPeakVelocity_zTrunkL.MaxPeakTrunkA25v_z)
AllVelocity3 = np.append(MaxPeakVelocity_zTrunkS.MaxPeakTrunkAM25v_z,MaxPeakVelocity_zTrunkL.MaxPeakTrunkAM25v_z)
AllVelocity = np.append(AllVelocity,AllVelocity2)
AllVelocity = np.append(AllVelocity,AllVelocity3)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFarTrunk),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFarTrunk*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar)) #here we remove the participant 9 from the stats because of bad quality data

for i in iteration:
  plt.plot([-0.20,0.20],[MaxPeakVelocity_zTrunkS.MaxPeakTrunkA0v_z[i],MaxPeakVelocity_zTrunkL.MaxPeakTrunkA0v_z[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxPeakVelocity_zTrunkS.MaxPeakTrunkA25v_z[i],MaxPeakVelocity_zTrunkL.MaxPeakTrunkA25v_z[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxPeakVelocity_zTrunkS.MaxPeakTrunkAM25v_z[i],MaxPeakVelocity_zTrunkL.MaxPeakTrunkAM25v_z[i]],color='lightgrey')

plt.title('Max Velocity Trunk z Angle') 
plt.ylabel('Velocity Trunk z [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocity Trunkz Angle')
#MaxPeak_Stat_A0 = stats.ttest_rel(MaxPeakVelocity_zTrunkS.MaxPeakTrunkA0v_z,MaxPeakVelocity_zTrunkL.MaxPeakTrunkA0v_z)
#MaxPeak_Stat_A25 = stats.ttest_rel(MaxPeakVelocity_zTrunkS.MaxPeakTrunkA25v_z,MaxPeakVelocity_zTrunkL.MaxPeakTrunkA25v_z)
#MaxPeak_Stat_AM25 = stats.ttest_rel(MaxPeakVelocity_zTrunkS.MaxPeakTrunkAM25v_z,MaxPeakVelocity_zTrunkL.MaxPeakTrunkAM25v_z)
#radial
AllVelocity = np.append(MaxPeakVelocity_TrunkS.MaxPeakTrunkA0v_r,MaxPeakVelocity_TrunkL.MaxPeakTrunkA0v_r)
AllVelocity2 = np.append(MaxPeakVelocity_TrunkS.MaxPeakTrunkA25v_r,MaxPeakVelocity_TrunkL.MaxPeakTrunkA25v_r)
AllVelocity3 = np.append(MaxPeakVelocity_TrunkS.MaxPeakTrunkAM25v_r,MaxPeakVelocity_TrunkL.MaxPeakTrunkAM25v_r)
AllVelocity = np.append(AllVelocity,AllVelocity2)
AllVelocity = np.append(AllVelocity,AllVelocity3)
plt.figure(dpi = 1200)
data = {'Bias':  AllVelocity,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFarTrunk),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFarTrunk*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
iteration = np.append(np.arange(0,8),np.arange(9,numberPatientSoFar)) #here we remove the participant 9 from the stats because of bad quality data

for i in iteration:
  plt.plot([-0.20,0.20],[MaxPeakVelocity_TrunkS.MaxPeakTrunkA0v_r[i],MaxPeakVelocity_TrunkL.MaxPeakTrunkA0v_r[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxPeakVelocity_TrunkS.MaxPeakTrunkA25v_r[i],MaxPeakVelocity_TrunkL.MaxPeakTrunkA25v_r[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxPeakVelocity_TrunkS.MaxPeakTrunkAM25v_r[i],MaxPeakVelocity_TrunkL.MaxPeakTrunkAM25v_r[i]],color='lightgrey')

plt.title('Max Velocity Trunk  Angle') 
plt.ylabel('Velocity Trunk  [m/s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxVelocity Trunk Angle')
#MaxPeak_Stat_A0 = stats.ttest_rel(MaxPeakVelocity_TrunkS.MaxPeakTrunkA0v_r,MaxPeakVelocity_TrunkL.MaxPeakTrunkA0v_r)
#MaxPeak_Stat_A25 = stats.ttest_rel(MaxPeakVelocity_TrunkS.MaxPeakTrunkA25v_r,MaxPeakVelocity_TrunkL.MaxPeakTrunkA25v_r)
#MaxPeak_Stat_AM25 = stats.ttest_rel(MaxPeakVelocity_TrunkS.MaxPeakTrunkAM25v_r,MaxPeakVelocity_TrunkL.MaxPeakTrunkAM25v_r)

#-----------------------------------------------------------------------------------------------------------------------------------------
#5. Max Peak latency
AllLatency = np.append(MaxLatencyVelocity_S.MaxPeakLatencyv_r,MaxLatencyVelocity_L.MaxPeakLatencyv_r)
MaxPeakLatency_Stat = stats.ttest_rel(MaxLatencyVelocity_S.MaxPeakLatencyv_r,MaxLatencyVelocity_L.MaxPeakLatencyv_r)

plt.figure(dpi = 1200)
data = {'Bias':  AllLatency,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
     'Groups': np.repeat(['Latency'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxLatencyVelocity_L.MaxPeakLatencyv_r)):
  plt.plot([-0.20,0.20],[MaxLatencyVelocity_S.MaxPeakLatencyv_r[i],MaxLatencyVelocity_L.MaxPeakLatencyv_r[i]],color='lightgrey')
 
plt.title('MaxLatency') 
plt.ylabel('Time [s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxLatency')

#3.1 With Angle

Latency = np.append(MaxLatencyVelocity_S.MaxLatencyA0v_r,MaxLatencyVelocity_L.MaxLatencyA0v_r)
AllLatency2 = np.append(MaxLatencyVelocity_S.MaxLatencyA25v_r,MaxLatencyVelocity_L.MaxLatencyA25v_r)
AllLatency3 = np.append(MaxLatencyVelocity_S.MaxLatencyAM25v_r,MaxLatencyVelocity_L.MaxLatencyAM25v_r)
AllLatency = np.append(AllLatency,AllLatency2)
AllLatency = np.append(AllLatency,AllLatency3)
plt.figure(dpi = 1200)
data = {'Bias':  AllLatency,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MaxLatencyVelocity_L.MaxPeakLatencyv_r)):
  plt.plot([-0.20,0.20],[MaxLatencyVelocity_S.MaxLatencyA0v_r[i],MaxLatencyVelocity_L.MaxLatencyA0v_r[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MaxLatencyVelocity_S.MaxLatencyA25v_r[i],MaxLatencyVelocity_L.MaxLatencyA25v_r[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MaxLatencyVelocity_S.MaxLatencyAM25v_r[i],MaxLatencyVelocity_L.MaxLatencyAM25v_r[i]],color='lightgrey')

plt.title('Max Latency Angle') 
plt.ylabel('Time  [s]')   
plt.legend(loc = 'upper right')
plt.savefig('MaxLatency Angle')
MaxPeakLatency_Stat_A0 = stats.ttest_rel(MaxLatencyVelocity_S.MaxLatencyA0v_r,MaxLatencyVelocity_L.MaxLatencyA0v_r)
MaxPeakLatency_Stat_A25 = stats.ttest_rel(MaxLatencyVelocity_S.MaxLatencyA25v_r,MaxLatencyVelocity_L.MaxLatencyA25v_r)
MaxPeakLatency_Stat_AM25 = stats.ttest_rel(MaxLatencyVelocity_S.MaxLatencyAM25v_r,MaxLatencyVelocity_L.MaxLatencyAM25v_r)


#-----------------------------------------------------------------------------------------------------------------------------------------
#6. Movement Duration
MDuration = np.append(MovementDuration_S.MeanMovementDurationv_r,MovementDuration_L.MeanMovementDurationv_r)
MDuration_Stat = stats.ttest_rel(MovementDuration_S.MeanMovementDurationv_r,MovementDuration_L.MeanMovementDurationv_r,nan_policy='omit')

plt.figure(dpi = 1200)
data = {'Bias':  MDuration,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
     'Groups': np.repeat(['Duration'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MovementDuration_L.MeanMovementDurationv_r)):
  plt.plot([-0.20,0.20],[MovementDuration_S.MeanMovementDurationv_r[i],MovementDuration_L.MeanMovementDurationv_r[i]],color='lightgrey')

plt.title('Movement Duration') 
plt.ylabel('Time [s]')   
plt.legend(loc = 'upper right')
plt.savefig('MovementDuration')

#3.1 With Angle

MDuration = np.append(MovementDuration_S.MeanMovementDurationA0v_r,MovementDuration_L.MeanMovementDurationA0v_r)
MDuration2 = np.append(MovementDuration_S.MeanMovementDurationA25v_r,MovementDuration_L.MeanMovementDurationA25v_r)
MDuration3 = np.append(MovementDuration_S.MeanMovementDurationAM25v_r,MovementDuration_L.MeanMovementDurationAM25v_r)
MDuration = np.append(MDuration,MDuration2)
MDuration = np.append(MDuration,MDuration3)
plt.figure(dpi = 1200)
data = {'Bias': MDuration,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
#ax3 = sns.line(x="Groups", y="Bias", hue="Conditions",data=df, palette='lightgrey', dodge=True)

pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MovementDuration_L.MeanMovementDurationA0v_r)):
  plt.plot([-0.20,0.20],[MovementDuration_S.MeanMovementDurationA0v_r[i],MovementDuration_L.MeanMovementDurationA0v_r[i]],color='lightgrey')
  plt.plot([0.80,1.20],[MovementDuration_S.MeanMovementDurationA25v_r[i],MovementDuration_L.MeanMovementDurationA25v_r[i]],color='lightgrey')
  plt.plot([1.80,2.20],[MovementDuration_S.MeanMovementDurationAM25v_r[i],MovementDuration_L.MeanMovementDurationAM25v_r[i]],color='lightgrey')
#MDuration_Stat_A0 = stats.ttest_rel(MovementDuration_S.MeanMovementDurationA0v_r,MovementDuration_L.MeanMovementDurationA0v_r)
#MDuration_Stat_A25 = stats.ttest_rel(MovementDuration_S.MeanMovementDurationA25v_r,MovementDuration_L.MeanMovementDurationA25v_r)
#MDuration_Stat_AM25 = stats.ttest_rel(MovementDuration_S.MeanMovementDurationAM25v_r,MovementDuration_L.MeanMovementDurationAM25v_r)

plt.title('MovementDuration Angle') 
plt.ylabel('Time  [s]')   
plt.legend(loc = 'upper right')
plt.savefig('MovementDuration Angle')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#7. PinchDistance
#7. PinchDistance
AllPinch = np.append(PinchDistance_S['Mean PinchDistance'],PinchDistance_L['Mean PinchDistance'])
print("Pinch",stats.ttest_rel(PinchDistance_S['Mean PinchDistance'],PinchDistance_L['Mean PinchDistance'],nan_policy = 'omit'))

plt.figure(dpi = 1200)
data = {'Bias':  AllPinch,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
     'Groups': np.repeat(['PinchDistance'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(PinchDistance_S['Mean PinchDistance'])):
    plt.plot([-0.20,0.20],[PinchDistance_S['Mean PinchDistance'][i],PinchDistance_L['Mean PinchDistance'][i]],color='lightgrey')


plt.title('PinchDistance') 
plt.ylabel('Distance [cm]')   
plt.legend(loc = 'upper right')
plt.savefig('PinchDistance')



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#8. SelfConf
AllSelf = np.append(SelfConf_S.SelfConf1,SelfConf_L.SelfConf1)
Self_Stat = stats.ttest_rel(SelfConf_S.SelfConf1,SelfConf_L.SelfConf1)

plt.figure(dpi = 1200)
data = {'Bias':  AllSelf,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
     'Groups': np.repeat(['Self'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
for i in range(0,len(SelfConf_S.SelfConf1)):
  plt.plot([-0.20,0.20],[SelfConf_S.SelfConf1[i],SelfConf_L.SelfConf1[i]],color='lightgrey')

pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
plt.title('SelfConf') 
plt.ylabel('ratings')   
plt.legend(loc = 'upper right')
plt.savefig('SelfConf')
#let's check if there is an order effect: hypothesis: if you start with standard hand, you are over confident with eElongatedated one 
MyRegression = {'SF1': np.append(SelfConf_S.SelfConf1,SelfConf_L.SelfConf1), 
                'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),1),
                'Order': np.tile(np.append(np.append(np.append(np.tile([1,2],7),[2]),np.tile([1,2],4)),1),2),
        }
df = pd.DataFrame(MyRegression,columns=['SF1','Conditions','Order'])

model = ols('SF1 ~ C(Conditions)+C(Order)+C(Conditions)*C(Order)', data = df,missing='drop')
modeL = model.fit()
modeL.summary()
res = smm.stats.anova_lm(modeL, typ= 2)
mc = MultiComparison(df['SF1'], df['Conditions'],df['Order'])
result = mc.tukeyhsd()
print(result)
print(mc.groupsunique)
f.interaction_plot(df['Conditions'], df['Order'],df['SF1'],colors=['red','blue'], markers=['D','^'], ms=10)
stats.ttest_ind(SelfConf_S.SelfConf1[Order_S_L],SelfConf_S.SelfConf1[Order_L_S],nan_policy='omit')
stats.ttest_ind(SelfConf_L.SelfConf1[Order_S_L],SelfConf_L.SelfConf1[Order_L_S],nan_policy='omit')
stats.ttest_ind(SelfConf_S.SelfConf1[Order_S_L],SelfConf_L.SelfConf1[Order_S_L],nan_policy='omit')
stats.ttest_ind(SelfConf_S.SelfConf1[Order_L_S],SelfConf_L.SelfConf1[Order_L_S],nan_policy='omit')


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#9. WristAccuracy
WA = np.append(WristAccuracy_S.Meanwristaccuracy,WristAccuracy_L.Meanwristaccuracy)
WA_Stat = stats.ttest_rel(WristAccuracy_S.Meanwristaccuracy,WristAccuracy_L.Meanwristaccuracy)

plt.figure(dpi = 1200)
data = {'Bias': WA,
        'Conditions': np.repeat(['Standard','Elongated'],numberPatientSoFar),
     'Groups': np.repeat(['Accuracy'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(WristAccuracy_S.Meanwristaccuracy)):
  plt.plot([-0.20,0.20],[WristAccuracy_S.Meanwristaccuracy[i],WristAccuracy_L.Meanwristaccuracy[i]],color='lightgrey')

plt.title('Wrist Accuracy') 
plt.ylabel('Accuracy [cm]')   
plt.legend(loc = 'upper right')
plt.savefig('WristAccuracy')

#3.1 With Angle

WA = np.append(WristAccuracy_S.MeanwristaccuracyA0,WristAccuracy_L.MeanwristaccuracyA0)
WA2 = np.append(WristAccuracy_S.MeanwristaccuracyA25,WristAccuracy_L.MeanwristaccuracyA25)
WA3 = np.append(WristAccuracy_S.MeanwristaccuracyAM25,WristAccuracy_L.MeanwristaccuracyAM25)
WA = np.append(WA,WA2)
WA = np.append(WA,WA3)
plt.figure(dpi = 1200)
data = {'Bias': WA,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['A0','A25','AM25'],numberPatientSoFar*2,axis = 0)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
my_pal = {'Elongated': "r",'Standard': "mistyrose"}
my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(MovementDuration_L.MeanMovementDurationA0v_r)):
  plt.plot([-0.20,0.20],[WristAccuracy_S.MeanwristaccuracyA0[i],WristAccuracy_L.MeanwristaccuracyA0[i]],color='lightgrey')
  plt.plot([0.80,1.20],[WristAccuracy_S.MeanwristaccuracyA25[i],WristAccuracy_L.MeanwristaccuracyA25[i]],color='lightgrey')
  plt.plot([1.80,2.20],[WristAccuracy_S.MeanwristaccuracyAM25[i],WristAccuracy_L.MeanwristaccuracyAM25[i]],color='lightgrey')
WA_Stat_A0 = stats.ttest_rel(WristAccuracy_S.MeanwristaccuracyA0,WristAccuracy_L.MeanwristaccuracyA0)
WA_Stat_A25 = stats.ttest_rel(WristAccuracy_S.MeanwristaccuracyA25,WristAccuracy_L.MeanwristaccuracyA25)
WA_Stat_AM25 = stats.ttest_rel(WristAccuracy_S.MeanwristaccuracyAM25,WristAccuracy_L.MeanwristaccuracyAM25)

plt.title('Wrist Accuracy Angle') 
plt.ylabel('Accuracy [cm]')   
plt.legend(loc = 'upper right')
plt.savefig('WristAccuracy Angle')
# Start Stop to see what start to move first (wrist shoulder elbow or trunk and if this changes between conditions)
#1. Compute the difference of starting the movement between wrist-elbow/wrist-shoulder/wrist-Trunk
#2. compare this difference between conditions
#1
WRISTELBOW_S = np.mean(StartWrist_S.iloc[:, 1:42]*(1/120) - StartElbow_S.iloc[:, 1:42]*(1/120),1)
WRISTSHOULDER_S= np.mean(StartWrist_S.iloc[np.append(np.arange(0,8),np.arange(9,24)), 1:42]*(1/120) - StartShoulder_S.iloc[:, 1:42]*(1/120),1)
WRISTTRUNK_S = np.mean(StartWrist_S.iloc[np.append(np.arange(0,8),np.arange(9,24)), 1:42]*(1/120) - StartTrunk_S.iloc[:, 1:42]*(1/120),1)

WRISTELBOW_L = np.mean(StartWrist_L.iloc[:, 1:42]*(1/120) - StartElbow_L.iloc[:, 1:42]*(1/120),1)
WRISTSHOULDER_L= np.mean(StartWrist_L.iloc[np.append(np.arange(0,8),np.arange(9,24)), 1:42]*(1/120) - StartShoulder_L.iloc[:, 1:42]*(1/120),1)
WRISTTRUNK_L = np.mean(StartWrist_L.iloc[np.append(np.arange(0,8),np.arange(9,24)), 1:42]*(1/120) - StartTrunk_L.iloc[:, 1:42]*(1/120),1)

WE = np.append(WRISTELBOW_S,WRISTELBOW_L)
WS = np.append(WRISTSHOULDER_S,WRISTSHOULDER_L)
WT = np.append(WRISTTRUNK_S,WRISTTRUNK_L)
TIMEDIFF = np.append(WE,WS)
TIMEDIFF = np.append(TIMEDIFF,WT)

data = {'Bias': TIMEDIFF,
        'Conditions': np.append(np.repeat(['Standard','Elongated'],numberPatientSoFar),np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFarTrunk),2)),
        'Groups': np.append(np.repeat(['WE'],numberPatientSoFar*2),np.repeat(['WS','WT'],numberPatientSoFarTrunk*2,axis = 0))}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups'])
model = ols('Bias ~ C(Conditions)+C(Groups)', data = df,missing='drop')
modeL = model.fit()
modeL.summary()
res = smm.stats.anova_lm(modeL, typ= 2)
mc = MultiComparison(df['Bias'], df['Conditions'])
result = mc.tukeyhsd()
print(result)
print(mc.groupsunique)
f.interaction_plot(df['Groups'],df['Conditions'],df['Bias'],colors=['red','blue'], markers=['D','^'], ms=10)
stats.ttest_rel(WRISTTRUNK_L,WRISTTRUNK_S,nan_policy = 'omit')
#my_pal = {'Elongated': "r",'Standard': "mistyrose"}
#my_pal2 = {'Elongated': "darkred",'Standard': "palevioletred"}
my_pal = {'WE': "r",'WS': "mistyrose",'WT':"pink"}
my_pal2 = {'WE': "darkred",'WS': "palevioletred",'WT': "orange"}
ax = sns.stripplot(x="Conditions", y="Bias", hue="Groups",data=df, palette=my_pal2, dodge=True)
pp = sns.boxplot(y='Bias', x='Conditions', data=df, palette=my_pal,hue='Groups')
#for i in range(0,len(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r)):
#  plt.plot([-0.20,0.20],[MaxDistance_ElbowS.MeanMaxDistanceA0_Elbowv_r[i]/DistTarget[i],MaxDistance_ElbowL.MeanMaxDistanceA0_Elbowv_r[i]/DistTarget[i]],color='lightgrey')
#  plt.plot([0.80,1.20],[MaxDistance_ElbowS.MeanMaxDistanceA25_Elbowv_r[i]/DistTarget[i],MaxDistance_ElbowL.MeanMaxDistanceA25_Elbowv_r[i]/DistTarget[i]],color='lightgrey')
#  plt.plot([1.80,2.20],[MaxDistance_ElbowS.MeanMAxDistanceAM25_Elbowv_r[i]/DistTarget[i],MaxDistance_ElbowL.MeanMAxDistanceAM25_Elbowv_r[i]/DistTarget[i]],color='lightgrey')

plt.title('Timing Diff') 
plt.ylabel('Time Wrist-X [s]')   
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('TimingDIff',bbox_extra_artists=(lgd,), bbox_inches='tight')

#--------------------------------------------
#Fitt's Law?
#MT = a+b*log2(2*Distance object/Width Object)
#Let's see if this is true for us
a = -2.5
PredictedMDuration = a + np.log2(2*DistanceTarget_.DistTarget/0.075)
plt.figure()
#for i in range(0,len(MovementDuration_L.MeanMovementDurationA0v_r)):
plt.plot(np.arange(0,len(MovementDuration_L.MeanMovementDurationA0v_r)),MovementDuration_L.MeanMovementDurationv_r,'*r')
plt.plot(np.arange(0,len(MovementDuration_L.MeanMovementDurationA0v_r)),PredictedMDuration,'ob')


#---------------------------------------------------------------------------------------------------------------
#BL ANALYSIS
AffectedArmData_S = pd.read_excel('Data_BKL_BL_FINAL_header.xlsx', sheet_name='Standard')
AffectedArmData_L = pd.read_excel('Data_BKL_BL_FINAL_header.xlsx', sheet_name='Large')
AffectedArmData_B = pd.read_excel('Data_BKL_BL_FINAL_header.xlsx', sheet_name='Baseline')

numberPatientSoFar = 24



AffectedHandWidthDiff_S = np.zeros(numberPatientSoFar)
AffectedHandLengthDiff_S = np.zeros(numberPatientSoFar)
AffectedArmWidthDiff_S= np.zeros(numberPatientSoFar)
AffectedArmLengthDiff_S = np.zeros(numberPatientSoFar)

AffectedHandWidthDiff_B = np.zeros(numberPatientSoFar)
AffectedHandLengthDiff_B = np.zeros(numberPatientSoFar)
AffectedArmWidthDiff_B= np.zeros(numberPatientSoFar)
AffectedArmLengthDiff_B = np.zeros(numberPatientSoFar)

AffectedHandWidthDiff_L = np.zeros(numberPatientSoFar)
AffectedHandLengthDiff_L = np.zeros(numberPatientSoFar)
AffectedArmWidthDiff_L= np.zeros(numberPatientSoFar)
AffectedArmLengthDiff_L = np.zeros(numberPatientSoFar)
FullUpperLimbLength_S=   np.zeros(numberPatientSoFar)
FullUpperLimbLength_B=   np.zeros(numberPatientSoFar)
FullUpperLimbLength_L=   np.zeros(numberPatientSoFar)


for i in range(0,numberPatientSoFar):
  AffectedHandWidthDiff_S[i] = AffectedArmData_S.iloc[i,2]/AffectedArmData_S.iloc[i,1];
  AffectedArmWidthDiff_S[i] = AffectedArmData_S.iloc[i,4]/AffectedArmData_S.iloc[i,3]
  AffectedHandLengthDiff_S[i] = ((( AffectedArmData_S.iloc[i,6]+AffectedArmData_S.iloc[i,8])/2)/((AffectedArmData_S.iloc[i,7]+AffectedArmData_S.iloc[i,5])/2))
  AffectedArmLengthDiff_S[i] = ((AffectedArmData_S.iloc[i,10] + AffectedArmData_S.iloc[i,12])/2)/((AffectedArmData_S.iloc[i,11]+AffectedArmData_S.iloc[i,9])/2)
  FullUpperLimbLength_S[i] = (((AffectedArmData_S.iloc[i,6]+AffectedArmData_S.iloc[i,8])/2)+((AffectedArmData_S.iloc[i,10]+ AffectedArmData_S.iloc[i,12])/2))/(((AffectedArmData_S.iloc[i,7]+AffectedArmData_S.iloc[i,5])/2)+((AffectedArmData_S.iloc[i,11]+AffectedArmData_S.iloc[i,9])/2))


for i in range(0,numberPatientSoFar):
  AffectedHandWidthDiff_B[i] = AffectedArmData_B.iloc[i,2]/AffectedArmData_B.iloc[i,1];
  AffectedArmWidthDiff_B[i] = AffectedArmData_B.iloc[i,4]/AffectedArmData_B.iloc[i,3]
  AffectedHandLengthDiff_B[i] = ((( AffectedArmData_B.iloc[i,6]+AffectedArmData_B.iloc[i,8])/2)/((AffectedArmData_B.iloc[i,7]+AffectedArmData_B.iloc[i,5])/2))
  AffectedArmLengthDiff_B[i] = ((AffectedArmData_B.iloc[i,10] + AffectedArmData_B.iloc[i,12])/2)/((AffectedArmData_B.iloc[i,11]+AffectedArmData_B.iloc[i,9])/2)
  FullUpperLimbLength_B[i] = (((AffectedArmData_B.iloc[i,6]+AffectedArmData_B.iloc[i,8])/2)+((AffectedArmData_B.iloc[i,10]+ AffectedArmData_B.iloc[i,12])/2))/(((AffectedArmData_B.iloc[i,7]+AffectedArmData_B.iloc[i,5])/2)+((AffectedArmData_B.iloc[i,11]+AffectedArmData_B.iloc[i,9])/2))

for i in range(0,numberPatientSoFar):
  AffectedHandWidthDiff_L[i] = AffectedArmData_L.iloc[i,2]/AffectedArmData_L.iloc[i,1];
  AffectedArmWidthDiff_L[i] = AffectedArmData_L.iloc[i,4]/AffectedArmData_L.iloc[i,3]
  AffectedHandLengthDiff_L[i] = ((( AffectedArmData_L.iloc[i,6]+AffectedArmData_L.iloc[i,8])/2)/((AffectedArmData_L.iloc[i,7]+AffectedArmData_L.iloc[i,5])/2))
  AffectedArmLengthDiff_L[i] = ((AffectedArmData_L.iloc[i,10] + AffectedArmData_L.iloc[i,12])/2)/((AffectedArmData_L.iloc[i,11]+AffectedArmData_L.iloc[i,9])/2)
  FullUpperLimbLength_L[i] = (((AffectedArmData_L.iloc[i,6]+AffectedArmData_L.iloc[i,8])/2)+((AffectedArmData_L.iloc[i,10]+ AffectedArmData_L.iloc[i,12])/2))/(((AffectedArmData_L.iloc[i,7]+AffectedArmData_L.iloc[i,5])/2)+((AffectedArmData_L.iloc[i,11]+AffectedArmData_L.iloc[i,9])/2))


HandLength = AffectedHandLengthDiff_B
HandLength = np.append(HandLength, AffectedHandLengthDiff_S)
HandLength = np.append(HandLength, AffectedHandLengthDiff_L)

ArmLength = AffectedArmLengthDiff_B
ArmLength = np.append(ArmLength, AffectedArmLengthDiff_S)
ArmLength = np.append(ArmLength, AffectedArmLengthDiff_L)

FullArmLength = FullUpperLimbLength_B
FullArmLength = np.append(FullArmLength,FullUpperLimbLength_S)
FullArmLength = np.append(FullArmLength,FullUpperLimbLength_L) 

HandWidth = AffectedHandWidthDiff_B
HandWidth = np.append(HandWidth, AffectedHandWidthDiff_S)
HandWidth = np.append(HandWidth, AffectedHandWidthDiff_L)

ArmWidth = AffectedArmWidthDiff_B
ArmWidth = np.append(ArmWidth, AffectedArmWidthDiff_S)
ArmWidth = np.append(ArmWidth, AffectedArmWidthDiff_L)

Alltogether = np.append(HandLength,ArmLength)
Alltogether = np.append(Alltogether,FullArmLength)
Alltogether = np.append(Alltogether,HandWidth)
Alltogether = np.append(Alltogether,ArmWidth)


#Check if normaly distributed
plt.hist(Alltogether)
plt.title('Normaldistribution of Bias')
plt.savefig('Bias normaldist')
plt.show()

plt.figure(dpi = 1200)
data = {'Bias':  Alltogether,
        'Conditions': np.tile(np.repeat(['Baseline','Standard','Elongated'],numberPatientSoFar),5),
        'Groups': np.repeat(['HandLength','ArmLength','FullArmLength','HandWidth','ArmWidth'],numberPatientSoFar*3,axis = 0),
        'ID': np.tile(np.arange(1,numberPatientSoFar+1),15)}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups','ID'])
my_pal = {'Elongated': "darkgrey",'Baseline': "white",'Standard':"r"}
my_pal2 = {'Elongated': "grey",'Baseline': "antiquewhite",'Standard':"r"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
plt.hlines(1, -1, 5, colors='k', linestyles='dashed')
plt.title('Upper limb perception')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('Figure_Paper/BL_YBKL',bbox_extra_artists=(lgd,), bbox_inches='tight')




#------------ LENGTH
AllLENGTH = np.append(HandLength,ArmLength)
AllLENGTH = np.append(AllLENGTH,FullArmLength)
plt.figure(dpi = 1200)
data = {'Bias':  AllLENGTH,
        'Conditions': np.tile(np.repeat(['Baseline','Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['HandLength','ArmLength','FullArmLength'],numberPatientSoFar*3,axis = 0),
        'ID': np.append(np.append(np.tile(np.arange(1,numberPatientSoFar+1),3),np.tile(np.arange(numberPatientSoFar+1,2*(numberPatientSoFar)+1),3)),np.tile(np.arange((numberPatientSoFar*2)+1,3*(numberPatientSoFar)+1),3))}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups','ID'])
my_pal = {'Elongated': "darkgrey",'Baseline': "white",'Standard':"r"}
my_pal2 = {'Elongated': "grey",'Baseline': "antiquewhite",'Standard':"r"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
plt.hlines(1, -1, 3, colors='k', linestyles='dashed')
plt.title('Upper limb perception')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('Figure_Paper/BL_Length',bbox_extra_artists=(lgd,), bbox_inches='tight')
#WIDTH
AllWIDTH = np.append(HandWidth,ArmWidth)
plt.figure(dpi = 1200)
data = {'Bias':  AllWIDTH,
        'Conditions': np.tile(np.repeat(['Baseline','Standard','Elongated'],numberPatientSoFar),2),
        'Groups': np.repeat(['HandWidth','ArmWidth'],numberPatientSoFar*3,axis = 0),
        'ID': np.append(np.append(np.tile(np.arange(1,numberPatientSoFar+1),2),np.tile(np.arange(numberPatientSoFar+1,2*(numberPatientSoFar)+1),2)),np.tile(np.arange((numberPatientSoFar*2)+1,3*(numberPatientSoFar)+1),2))}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups','ID'])
my_pal = {'Elongated': "darkgrey",'Baseline': "white",'Standard':"r"}
my_pal2 = {'Elongated': "grey",'Baseline': "antiquewhite",'Standard':"r"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
plt.hlines(1, -1, 2, colors='k', linestyles='dashed')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('Figure_Paper/BL_Width',bbox_extra_artists=(lgd,), bbox_inches='tight')

import pingouin as pg
aov = pg.mixed_anova(dv='Bias', between='Groups', within='Conditions', subject='ID', data=df)

stats.ttest_rel(AffectedHandLengthDiff_S- AffectedHandLengthDiff_B,AffectedHandLengthDiff_L- AffectedHandLengthDiff_B)

mc = MultiComparison(df['Bias'], data['Conditions'])
result = mc.tukeyhsd()
plt.show()
# mc = MultiComparison(df['Bias'], data['Conditions'])
# result = mc.tukeyhsd()
# print(result)
# print(mc.groupsunique)

#to visualize when we substract the aseline to the other condition
HandLength_B = AffectedHandLengthDiff_S- AffectedHandLengthDiff_B
HandLength_B = np.append(HandLength_B, AffectedHandLengthDiff_L-AffectedHandLengthDiff_B)

ArmLength_B = AffectedArmLengthDiff_S-AffectedArmLengthDiff_B
ArmLength_B = np.append(ArmLength_B, AffectedArmLengthDiff_L-AffectedArmLengthDiff_B)

FullArmLength_B = FullUpperLimbLength_S-FullUpperLimbLength_B
FullArmLength_B = np.append(FullArmLength_B,FullUpperLimbLength_L-FullUpperLimbLength_B) 
AllLENGTH_B = np.append(HandLength_B,ArmLength_B)
AllLENGTH_B = np.append(AllLENGTH_B,FullArmLength_B)
plt.figure(dpi = 1200)
data = {'Bias':  AllLENGTH_B,
        'Conditions': np.tile(np.repeat(['Standard','Elongated'],numberPatientSoFar),3),
        'Groups': np.repeat(['HandLength','ArmLength','FullArmLength'],numberPatientSoFar*2,axis = 0),
        'ID': np.append(np.append(np.tile(np.arange(1,numberPatientSoFar+1),2),np.tile(np.arange(numberPatientSoFar+1,2*(numberPatientSoFar)+1),2)),np.tile(np.arange((numberPatientSoFar*2)+1,3*(numberPatientSoFar)+1),2))}
df = pd.DataFrame(data, columns = ['Bias','Conditions', 'Groups','ID'])
my_pal = {'Elongated': "darkgrey",'Standard':"r"}
my_pal2 = {'Elongated': "grey",'Standard':"r"}
ax = sns.stripplot(x="Groups", y="Bias", hue="Conditions",data=df, palette=my_pal2, dodge=True)
sns.boxplot(y='Bias', x='Groups', data=df, palette=my_pal,hue='Conditions')
for i in range(0,len(AffectedHandLengthDiff_S)):
  plt.plot([-0.20,0.20],[HandLength_B[i],HandLength_B[i+24]],color='lightgrey')
  plt.plot([0.80,1.20],[ArmLength_B[i],ArmLength_B[i+24]],color='lightgrey')
  plt.plot([1.80,2.20],[FullArmLength_B[i],FullArmLength_B[i+24]],color='lightgrey')

plt.title('difference of upper limb perception compared to baseline')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig('Figure_Paper/BL_Length_substractbaseline',bbox_extra_artists=(lgd,), bbox_inches='tight')


plt.show()

import pingouin as pg
pg.mixed_anova(dv='Bias', between='Groups', within='Conditions', subject='ID', data=df)
stats.f_oneway(FullUpperLimbLength_B,FullUpperLimbLength_S,FullUpperLimbLength_L)
stats.ttest_rel(AffectedHandLengthDiff_S- AffectedHandLengthDiff_B,AffectedHandLengthDiff_L- AffectedHandLengthDiff_B)
stats.ttest_rel(AffectedArmLengthDiff_S- AffectedArmLengthDiff_B,AffectedArmLengthDiff_L- AffectedArmLengthDiff_B)
stats.ttest_rel(FullUpperLimbLength_S-FullUpperLimbLength_B,FullUpperLimbLength_L-FullUpperLimbLength_B)
#----------------------------------------------------------------------------------
#Let's now try to predict the data from the bodylandmark (specifically the length of the full Arm), based on the kinematics data.
# We have on one side 3 different parameters that we can try to predict: Full arm length, hand length and arm length, in the order of interest respectively.
#On the other side, we Have several parameters of interest which are the Wrist, Elbow , Shoulder and Trunk Distance, Their velocity, Movement Duration etc
#But Looking at the data, right now the Trunk distance is not well labelized yet to be of use right now. Putting this on the side,
#Since there is no difference of Velocity for the Wrist in Both radial and z direction,The first model we can do is to take into account only the distance of the Wrist Shoulder and Elbow Component.
#But we have three different target positon, but to keep it simple the first model would be to try to take the mean of the target position
#this will give the following model: FullUpperLimbLength_i = X1* WristDistance + X2 * ElbowDistance + X3* ShoulderDistance + C;
#Then it is also possi ble to try with only with the target 0, or to add the velocity in it, but this will be for later
#MeanOf TArgetPosition 
plt.hist(np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget))
plt.title('Normaldistribution of WristDist')
plt.savefig('Wrist normaldist ')
plt.show()
MyRegression = {'WristDistance': np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget), 
                'ElbowDistance':np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget),
                'ShoulderDistance':np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget),
                'FullUpperLimbLength': np.append(FullUpperLimbLength_S-FullUpperLimbLength_B,FullUpperLimbLength_L-FullUpperLimbLength_B)
        }
df = pd.DataFrame(MyRegression,columns=['WristDistance','ElbowDistance','ShoulderDistance','FullUpperLimbLength'])


X = df[['WristDistance','ElbowDistance','ShoulderDistance']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['FullUpperLimbLength']
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X, missing = 'drop')
result = model.fit()
predictions = result.predict(X) 
 
print_model = result.summary()
print(print_model)
plt.plot(np.arange(0,len(df['FullUpperLimbLength'])),df['FullUpperLimbLength'],'ro',np.arange(0,len(predictions)),predictions, 'k--')
plt.savefig('Model_WES_Mean.png')
# With A0 only (angle 0, in front of the participant)
MyRegressionA0 = {'WristDistance': np.append(MaxDistance_S.MeanMaxDistanceA0v_r/DistTarget,MaxDistance_L.MeanMaxDistanceA0v_r/DistTarget), 
                'ElbowDistance':np.append(MaxDistance_ElbowS.MeanMaxDistanceA0_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistanceA0_Elbowv_r/DistTarget),
                'ShoulderDistance':np.append(MaxDistance_ShoulderS.MeanMaxDistanceA0_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistanceA0_Shoulderv_r/DistTarget),
                'FullUpperLimbLength': np.append(FullUpperLimbLength_S-FullUpperLimbLength_B,FullUpperLimbLength_L-FullUpperLimbLength_B)
        }
df = pd.DataFrame(MyRegressionA0,columns=['WristDistance','ElbowDistance','ShoulderDistance','FullUpperLimbLength'])


XA0 = df[['WristDistance','ElbowDistance','ShoulderDistance']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
YA0 = df['FullUpperLimbLength']
XA0 = sm.add_constant(XA0) # adding a constant
 
modelA0 = sm.OLS(YA0, XA0, missing = 'drop')
resultA0 = modelA0.fit()
predictionsA0 = resultA0.predict(XA0) 
 
print_modelA0 = resultA0.summary()
print(print_modelA0)
plt.plot(np.arange(0,len(df['FullUpperLimbLength'])),df['FullUpperLimbLength'],'ro',np.arange(0,len(predictionsA0)),predictionsA0, 'k--')
plt.savefig('Model_WES_A0.png')

#------------------------
#With Only 2 terms
MyRegressionA02 = {'WristDistance': np.append(MaxDistance_S.MeanMaxDistanceA0v_r/DistTarget,MaxDistance_L.MeanMaxDistanceA0v_r/DistTarget), 
                   'WristPeak':np.append(MaxPeakVelocity_S.MaxPeakv_r,MaxPeakVelocity_L.MaxPeakv_r),
                'HandLength': np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B)
        }
df = pd.DataFrame(MyRegressionA02,columns=['WristDistance','WristPeak','HandLength'])


XA02 = df[['WristDistance','WristPeak']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
YA02 = df['HandLength']
XA02 = sm.add_constant(XA02) # adding a constant
 
modelA02 = sm.OLS(YA02, XA02, missing = 'drop')
resultA02 = modelA02.fit()
predictionsA02 = resultA02.predict(XA02) 
 
print_modelA02 = resultA02.summary()
print(print_modelA02)
plt.plot(np.arange(0,len(df['FullUpperLimbLength'])),df['FullUpperLimbLength'],'ro',np.arange(0,len(predictionsA02)),predictionsA02, 'k--')
plt.savefig('Model_WE_A0.png')


MyRegressionA02 = {'WristDistance': np.append(MaxDistance_S.MeanMaxDistanceA0v_r/DistTarget,MaxDistance_L.MeanMaxDistanceA0v_r/DistTarget), 
                   'WristPeak':np.append(MaxPeakVelocity_S.MaxPeakv_r,MaxPeakVelocity_L.MaxPeakv_r),
                'HandLength': np.append(AffectedHandLengthDiff_S,AffectedHandLengthDiff_L)
        }
df = pd.DataFrame(MyRegressionA02,columns=['WristDistance','WristPeak','HandLength'])


XA02 = df[['WristDistance','WristPeak']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
YA02 = df['HandLength']
XA02 = sm.add_constant(XA02) # adding a constant
 
modelA02 = sm.OLS(YA02, XA02, missing = 'drop')
resultA02 = modelA02.fit()
predictionsA02 = resultA02.predict(XA02) 
 
print_modelA02 = resultA02.summary()
print(print_modelA02)
plt.plot(np.arange(0,len(df['FullUpperLimbLength'])),df['FullUpperLimbLength'],'ro',np.arange(0,len(predictionsA02)),predictionsA02, 'k--')
plt.savefig('Model_WE_A0.png')

MyRegressionA02 = {   'WristPeak':np.append(MaxPeakVelocity_S.MaxPeakv_r,MaxPeakVelocity_L.MaxPeakv_r),
                'HandLength': np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B)
        }
df = pd.DataFrame(MyRegressionA02,columns=['WristPeak','HandLength'])


XA02 = df[['WristPeak']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
YA02 = df['HandLength']
XA02 = sm.add_constant(XA02) # adding a constant
 
modelA02 = sm.OLS(YA02, XA02, missing = 'drop')
resultA02 = modelA02.fit()
predictionsA02 = resultA02.predict(XA02) 
 
print_modelA02 = resultA02.summary()
print(print_modelA02)

#------------------ 
#prediction of armlength only
MyRegression = {'WristDistance': np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget), 
                'ElbowDistance':np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r),
                'ShoulderDistance':np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r),
                'FullUpperLimbLength': np.append(AffectedArmLengthDiff_S-AffectedArmLengthDiff_B,AffectedArmLengthDiff_L-AffectedArmLengthDiff_B)
        }
df = pd.DataFrame(MyRegression,columns=['WristDistance','ElbowDistance','ShoulderDistance','FullUpperLimbLength'])


X = df[['WristDistance','ElbowDistance','ShoulderDistance']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['FullUpperLimbLength']
X = sm.add_constant(X) # adding a constant
 
modelArmL = sm.OLS(Y, X, missing = 'drop')
resultArmL = modelArmL.fit()
predictionsArmL = resultArmL.predict(X) 
 
print_modelArmL = resultArmL.summary()
print(print_modelArmL)
plt.plot(np.arange(0,len(df['FullUpperLimbLength'])),df['FullUpperLimbLength'],'ro',np.arange(0,len(predictions)),predictions, 'k--')
plt.savefig('Model_WES_Mean_PArm.png')

#With the HandNow
MyRegression = {'WristDistance': np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget), 
                'ElbowDistance':np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget),
                'ShoulderDistance':np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget),
                'FullUpperLimbLength': np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B)
        }
df = pd.DataFrame(MyRegression,columns=['WristDistance','ElbowDistance','ShoulderDistance','FullUpperLimbLength'])


X = df[['WristDistance','ElbowDistance','ShoulderDistance']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['FullUpperLimbLength']
X = sm.add_constant(X) # adding a constant
 
modelHandL = sm.OLS(Y, X, missing = 'drop')
resultHandL = modelHandL.fit()
predictionsHandL = resultHandL.predict(X) 
 
print_modelHandL = resultHandL.summary()
print(print_modelHandL)
plt.plot(np.arange(0,len(df['FullUpperLimbLength'])),df['FullUpperLimbLength'],'ro',np.arange(0,len(predictionsHandL)),predictionsHandL, 'k--')
plt.savefig('Model_WES_Mean_PHand.png')

#---------------------
#Let's try to predict the eElongatedated only
MyRegression = {'WristDistance': MaxDistance_L.MeanMaxDistancev_r, 
                'ElbowDistance':MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r,
                'ShoulderDistance':MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r,
                'FullUpperLimbLength': FullUpperLimbLength_L-FullUpperLimbLength_B
        }
df = pd.DataFrame(MyRegression,columns=['WristDistance','ElbowDistance','ShoulderDistance','FullUpperLimbLength'])


X = df[['WristDistance','ElbowDistance','ShoulderDistance']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['FullUpperLimbLength']
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X, missing = 'drop')
result = model.fit()
predictions = result.predict(X) 
 
print_model = result.summary()
print(print_model)
plt.plot(np.arange(0,len(df['FullUpperLimbLength'])),df['FullUpperLimbLength'],'ro',np.arange(0,len(predictions)),predictions, 'k--')
plt.savefig('Model_WES_Mean_Large.png')

#---------------------
#Let's try to predict the standard only
MyRegression = {'WristDistance': MaxDistance_S.MeanMaxDistancev_r, 
                'ElbowDistance':MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r,
                'ShoulderDistance':MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r,
                'FullUpperLimbLength': FullUpperLimbLength_S-FullUpperLimbLength_B
        }
df = pd.DataFrame(MyRegression,columns=['WristDistance','ElbowDistance','ShoulderDistance','FullUpperLimbLength'])


X = df[['WristDistance','ElbowDistance','ShoulderDistance']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['FullUpperLimbLength']
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X, missing = 'drop')
result = model.fit()
predictions = result.predict(X) 
 
print_model = result.summary()
print(print_model)
plt.plot(np.arange(0,len(df['FullUpperLimbLength'])),df['FullUpperLimbLength'],'ro',np.arange(0,len(predictions)),predictions, 'k--')
plt.savefig('Model_WES_MeanStandard.png')

#Let's try to predict the differene eElongatedated-standard with the difference of distance elongated standard
MyRegression = {'WristDistance': MaxDistance_L.MeanMaxDistancev_r - MaxDistance_S.MeanMaxDistancev_r, 
                'ElbowDistance':MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r - MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r,
                'ShoulderDistance':MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r - MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r,
                'FullUpperLimbLength': (FullUpperLimbLength_L-FullUpperLimbLength_B) - (FullUpperLimbLength_S-FullUpperLimbLength_B)
        }
df = pd.DataFrame(MyRegression,columns=['WristDistance','ElbowDistance','ShoulderDistance','FullUpperLimbLength'])


X = df[['WristDistance','ElbowDistance','ShoulderDistance']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['FullUpperLimbLength']
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X, missing = 'drop')
result = model.fit()
predictions = result.predict(X) 
 
print_model = result.summary()
print(print_model)
plt.plot(np.arange(0,len(df['FullUpperLimbLength'])),df['FullUpperLimbLength'],'ro',np.arange(0,len(predictions)),predictions, 'k--')
plt.savefig('Model_WES_MeanDifference.png')

#--------------------------------------------------------------------------------
#Now we can divide participants between group that didnt notice the differences of size between condition and thos that did realize
SizeDiff = Embodiment_S.Size-Embodiment_L.Size # compute the difference of ratings
GroupDiffSize = np.nonzero(SizeDiff > 1) # if the scaling difference is bigger than 1, they noticed ( I decided of this cutoff, maybe it is not a good one)
GroupNoDiffSize = np.nonzero(SizeDiff< 1)
# results with no group division
stats.f_oneway(AffectedHandWidthDiff_B,AffectedHandWidthDiff_S,AffectedHandWidthDiff_L)
stats.f_oneway(AffectedArmWidthDiff_B,AffectedArmWidthDiff_S,AffectedArmWidthDiff_L)
stats.f_oneway(FullUpperLimbLength_B,FullUpperLimbLength_S,FullUpperLimbLength_L)
stats.ttest_rel(FullUpperLimbLength_S-FullUpperLimbLength_B,FullUpperLimbLength_L-FullUpperLimbLength_B)
stats.ttest_rel(AffectedHandWidthDiff_S-AffectedHandWidthDiff_B,AffectedHandWidthDiff_L-AffectedHandWidthDiff_B)
stats.ttest_rel(AffectedArmWidthDiff_S-AffectedArmWidthDiff_B,AffectedArmWidthDiff_L-AffectedArmWidthDiff_B)
stats.ttest_rel(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B)
stats.ttest_rel(AffectedArmLengthDiff_S-AffectedArmLengthDiff_B,AffectedArmLengthDiff_L-AffectedArmLengthDiff_B)

# with group division, when group has noticed size difference
stats.f_oneway(AffectedHandWidthDiff_B[GroupDiffSize],AffectedHandWidthDiff_S[GroupDiffSize],AffectedHandWidthDiff_L[GroupDiffSize])
stats.f_oneway(AffectedArmWidthDiff_B[GroupDiffSize],AffectedArmWidthDiff_S[GroupDiffSize],AffectedArmWidthDiff_L[GroupDiffSize])
stats.f_oneway(FullUpperLimbLength_B[GroupDiffSize],FullUpperLimbLength_S[GroupDiffSize],FullUpperLimbLength_L[GroupDiffSize])
stats.ttest_rel(FullUpperLimbLength_S[GroupDiffSize]-FullUpperLimbLength_B[GroupDiffSize],FullUpperLimbLength_L[GroupDiffSize]-FullUpperLimbLength_B[GroupDiffSize])
# with group that didnt notice
stats.f_oneway(AffectedHandWidthDiff_B[GroupNoDiffSize],AffectedHandWidthDiff_S[GroupNoDiffSize],AffectedHandWidthDiff_L[GroupNoDiffSize])
stats.f_oneway(AffectedArmWidthDiff_B[GroupNoDiffSize],AffectedArmWidthDiff_S[GroupNoDiffSize],AffectedArmWidthDiff_L[GroupNoDiffSize])
stats.f_oneway(FullUpperLimbLength_B[GroupNoDiffSize],FullUpperLimbLength_S[GroupNoDiffSize],FullUpperLimbLength_L[GroupNoDiffSize])
stats.ttest_rel(FullUpperLimbLength_S[GroupNoDiffSize]-FullUpperLimbLength_B[GroupNoDiffSize],FullUpperLimbLength_L[GroupNoDiffSize]-FullUpperLimbLength_B[GroupNoDiffSize])

#---------------------------------------------------------------------------------
#Another thing to check: the error of the wrist 
# Maybe the wristn is more shifted in the larger condition --> lets see. We are gonna focus on the z error 
IntWrist_Z_S_err = AffectedArmData_S.INTWRIST_Z_real-AffectedArmData_S.INTWRIST_Z_per
IntWrist_Z_L_err = AffectedArmData_L.INTWRIST_Z_real-AffectedArmData_L.INTWRIST_Z_per
IntWrist_Z_B_err = AffectedArmData_B.INTWRIST_Z_real-AffectedArmData_B.INTWRIST_Z_per
stats.ttest_rel(IntWrist_Z_S_err-IntWrist_Z_B_err,IntWrist_Z_L_err-IntWrist_Z_B_err)

ExtWrist_Z_S_err = AffectedArmData_S.EXTWRIST_Z_real-AffectedArmData_S.EXTWRIST_Z_per
ExtWrist_Z_L_err = AffectedArmData_L.EXTWRIST_Z_real-AffectedArmData_L.EXTWRIST_Z_per
ExtWrist_Z_B_err = AffectedArmData_B.EXTWRIST_Z_real-AffectedArmData_B.EXTWRIST_Z_per
stats.ttest_rel(ExtWrist_Z_S_err-ExtWrist_Z_B_err,ExtWrist_Z_L_err-ExtWrist_Z_B_err)


Index_Z_S_err = AffectedArmData_S.INDEX_Z_real-AffectedArmData_S.INDEX_Z_per
Index_Z_L_err = AffectedArmData_S.INDEX_Z_real-AffectedArmData_L.INDEX_Z_per
Index_Z_B_err = AffectedArmData_S.INDEX_Z_real-AffectedArmData_B.INDEX_Z_per
stats.ttest_rel(Index_Z_S_err-Index_Z_B_err,Index_Z_L_err-Index_Z_B_err)


Ring_Z_S_err = AffectedArmData_S.RING_Z_real-AffectedArmData_S.RING_Z_per
Ring_Z_L_err = AffectedArmData_L.RING_Z_real-AffectedArmData_L.RING_Z_per
Ring_Z_B_err = AffectedArmData_B.RING_Z_real-AffectedArmData_B.RING_Z_per
stats.ttest_rel(Ring_Z_S_err-Ring_Z_B_err,Ring_Z_L_err-Ring_Z_B_err)


Elbow_Z_S_err = AffectedArmData_S.ELBOW_Z_real-AffectedArmData_S.ELBOW_Z_per
Elbow_Z_L_err = AffectedArmData_L.ELBOW_Z_real-AffectedArmData_L.ELBOW_Z_per
Elbow_Z_B_err = AffectedArmData_B.ELBOW_Z_real-AffectedArmData_B.ELBOW_Z_per
stats.ttest_rel(Elbow_Z_S_err-Elbow_Z_B_err,Elbow_Z_L_err-Elbow_Z_B_err)





#--------------------------------------------------------------------------
#Linear relation ship?
import numpy.polynomial.polynomial as poly


xp = np.linspace(0.2, 0.8, 10000) 
#Wrist dist
#m,b = np.polyfit(np.append(AffectedHandLengthDiff_S,AffectedHandLengthDiff_L),np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget),1)

p = poly.polyfit(np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),1)
ffit = poly.polyval(xp, p)
plt.plot(np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),'ro',xp,ffit, color='C1')
stats.spearmanr(np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B))
plt.xlabel('Wrist Dist')
plt.ylabel('Handlength Bias')
plt.title('Wrist Dist vs UL perception Hand length')
plt.savefig('Linear_WristHandlength.png')

#m = np.polyfit(np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget),np.append(AffectedHandLengthDiff_S,AffectedHandLengthDiff_L),2)

p = poly.polyfit(np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),1)
ffit = poly.polyval(xp, p)
plt.plot(np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),'ro',xp,ffit, color='C1')
plt.xlabel('Elbow Dist')
plt.ylabel('Handlength Bias')
plt.title('Elbow Dist vs UL perception Hand length')
plt.savefig('Linear_ElbowHandlength.png')
np.corrcoef(np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B))

p = poly.polyfit(np.append(DistTarget,DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),1)
ffit = poly.polyval(xp, p)
plt.plot(np.append(DistTarget,DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),'ro',xp,ffit, color='C1')
plt.xlabel('Elbow Dist')
plt.ylabel('Handlength Bias')
np.corrcoef(np.append(DistTarget,DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B))


#m,b = np.polyfit(np.append(AffectedHandLengthDiff_S,AffectedHandLengthDiff_L),np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget),1)
p = poly.polyfit(np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),1)
ffit = poly.polyval(xp, p)
plt.plot(np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),'ro',xp,ffit,color='C1')
plt.xlabel('Shoulder Dist')
plt.ylabel('Handlength Bias')
plt.title('Shoulder Dist vs UL perception Hand length')
plt.savefig('Linear_ShoulderHandlength.png')
np.corrcoef(np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B))

# with peak vel
p = poly.polyfit(np.append(MaxPeakVelocity_S.MaxPeakv_r,MaxPeakVelocity_L.MaxPeakv_r),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),1)
ffit = poly.polyval(xp, p)
plt.plot(np.append(MaxPeakVelocity_S.MaxPeakv_r,MaxPeakVelocity_L.MaxPeakv_r),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),'ro',xp,ffit, color='C1')
plt.xlabel('Wrist PeakVel')
plt.ylabel('Handlength Bias')
plt.title('Wrist Vel vs UL perception Hand length')
plt.savefig('Linear_WristVelHandlength.png')
np.corrcoef(np.append(MaxPeakVelocity_S.MaxPeakv_r,MaxPeakVelocity_L.MaxPeakv_r),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B))

p,stats = poly.polyfit(np.append(MaxPeakVelocity_ElbowS.MaxPeakElbowv_r,MaxPeakVelocity_ElbowL.MaxPeakElbowv_r),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),1)
ffit = poly.polyval(xp, p)
plt.plot(np.append(MaxPeakVelocity_ElbowS.MaxPeakElbowv_r,MaxPeakVelocity_ElbowL.MaxPeakElbowv_r),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),'ro',xp,ffit, color='C1')
plt.xlabel('Elbow PeakVel')
plt.ylabel('Handlength Bias')
plt.title('Elbow Vel vs UL perception Hand length')
plt.savefig('Linear_ElbowVelHandlength.png')
xp = np.linspace(0, 0.2, 10000) 
np.corrcoef(np.append(MaxPeakVelocity_ElbowS.MaxPeakElbowv_r,MaxPeakVelocity_ElbowL.MaxPeakElbowv_r),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B))


p = poly.polyfit(np.append(MaxPeakVelocity_ShoulderS.MaxPeakShoulderv_r,MaxPeakVelocity_ShoulderL.MaxPeakShoulderv_r),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),1)
ffit = poly.polyval(xp, p)
plt.plot(np.append(MaxPeakVelocity_ShoulderS.MaxPeakShoulderv_r,MaxPeakVelocity_ShoulderL.MaxPeakShoulderv_r),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B),'ro',xp,ffit, color='C1')
plt.xlabel('Shoulder PeakVel')
plt.ylabel('Handlength Bias')
plt.title('Shoulder Vel vs UL perception Hand length')
plt.savefig('Linear_ShoulderVelHandlength.png')
np.corrcoef(np.append(MaxPeakVelocity_ShoulderS.MaxPeakShoulderv_r,MaxPeakVelocity_ShoulderL.MaxPeakShoulderv_r),np.append(AffectedHandLengthDiff_S-AffectedHandLengthDiff_B,AffectedHandLengthDiff_L-AffectedHandLengthDiff_B))

#Colinearity?
p = poly.polyfit(np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget),np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget),1)
ffit = poly.polyval(xp, p)
plt.plot(np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget),np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget),'ro',xp,ffit, color='C1')
plt.xlabel('Shoulder  Dist')
plt.ylabel('Elbow Dist')
plt.title('Shoulder Dist vs ElbowDist')
plt.savefig('CoLinear_ShoulderElbow.png')
np.corrcoef(np.append(MaxDistance_ShoulderS.MeanMaxDistance_Shoulderv_r/DistTarget,MaxDistance_ShoulderL.MeanMaxDistance_Shoulderv_r/DistTarget),np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget))

p = poly.polyfit(np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget),np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget),1)
ffit = poly.polyval(xp, p)
plt.plot(np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget),np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget),'ro',xp,ffit, color='C1')
plt.xlabel('Wrist  Dist')
plt.ylabel('Elbow Dist')
plt.title('Wrist Dist vs ElbowDist')
plt.savefig('CoLinear_WristElbow.png')
np.corrcoef(np.append(MaxDistance_S.MeanMaxDistancev_r/DistTarget,MaxDistance_L.MeanMaxDistancev_r/DistTarget),np.append(MaxDistance_ElbowS.MeanMaxDistance_Elbowv_r/DistTarget,MaxDistance_ElbowL.MeanMaxDistance_Elbowv_r/DistTarget))

#----------------------------
#ANOVA
import statsmodels.formula.api as smf
import statsmodels.api as smm
from statsmodels.formula.api import ols
MyRegression = {'WristDistance': MaxDistance_S.MeanMaxDistancev_r - MaxDistance_L.MeanMaxDistancev_r, 
                'WristPeak':MaxPeakVelocity_ShoulderS.MaxPeakShoulderv_r-MaxPeakVelocity_ShoulderL.MaxPeakShoulderv_r,
                'DistTarget': DistTarget,
                'HandUpperLimbLength': (AffectedHandLengthDiff_S-AffectedHandLengthDiff_B)-(AffectedHandLengthDiff_L-AffectedHandLengthDiff_B)
        }
df = pd.DataFrame(MyRegression,columns=['WristDistance','WristPeak','HandUpperLimbLength'])

model = ols('HandUpperLimbLength ~ C(WristDistance)+C(WristPeak)', data = df,missing='drop')
modeL = model.fit()
   res = smm.stats.anova_lm(modeL, typ= 2)
   res
import pingouin as pg
aov = pg.anova(dv='HandUpperLimbLength', between=['WristDistance', 'WristPeak'], data=df,
             detailed=True)


SizeDiff = Embodiment_S.Size-Embodiment_L.Size # compute the difference of ratings
GroupDiffSize = np.nonzero(SizeDiff > 1) # if the scaling difference is bigger than 1, they noticed ( I decided of this cutoff, maybe it is not a good one)
GroupNoDiffSize = np.nonzero(SizeDiff< 1)
HL1 = np.append(AffectedHandLengthDiff_S[GroupDiffSize]-AffectedHandLengthDiff_B[GroupDiffSize],AffectedHandLengthDiff_L[GroupDiffSize]-AffectedHandLengthDiff_B[GroupDiffSize])
HL1 = np.append(HL1,AffectedHandLengthDiff_S[GroupNoDiffSize]-AffectedHandLengthDiff_B[GroupNoDiffSize])
HL1 = np.append(HL1,AffectedHandLengthDiff_L[GroupNoDiffSize]-AffectedHandLengthDiff_B[GroupNoDiffSize])
MyRegressionA02 = {'HandLength': HL1,
                   'Conditions': np.append(np.repeat(['Standard','Elongated'],12),np.repeat(['Standard','Elongated'],12)),
        'Groups': np.append(np.repeat('DiffSize',12*2),np.repeat('NoDiffSize',12*2))}
df = pd.DataFrame(MyRegressionA02,columns=['HandLength','Conditions','Groups'])
mc = MultiComparison(df['HandLength'], df['Groups'])
result = mc.tukeyhsd()
print(result)
print(mc.groupsunique)
f.interaction_plot(df['Conditions'], df['Groups'],df['HandLength'],colors=['red','blue'], markers=['D','^'], ms=10)

HL2 = np.append(AffectedHandLengthDiff_S[GroupDiffSize],AffectedHandLengthDiff_L[GroupDiffSize])
HL2 = np.append(HL2,AffectedHandLengthDiff_S[GroupNoDiffSize])
HL2 = np.append(HL2,AffectedHandLengthDiff_L[GroupNoDiffSize])
MyRegression2 = {'HandLength': HL2,
                   'Conditions': np.append(np.repeat(['Standard','Elongated'],12),np.repeat(['Standard','Elongated'],12)),
        'Groups': np.append(np.repeat('DiffSize',12*2),np.repeat('NoDiffSize',12*2))}
df = pd.DataFrame(MyRegression2,columns=['HandLength','Conditions','Groups'])
mc = MultiComparison(df['HandLength'], df['Groups'])
result = mc.tukeyhsd()
print(result)
print(mc.groupsunique)
f.interaction_plot(df['Conditions'], df['Groups'],df['HandLength'],colors=['red','blue'], markers=['D','^'], ms=10)

HL3 = np.append(MaxDistance_S.MeanMaxDistancev_r[GroupDiffSize]/DistTarget[GroupDiffSize],MaxDistance_L.MeanMaxDistancev_r[GroupDiffSize]/DistTarget[GroupDiffSize])
HL3 = np.append(HL3,MaxDistance_S.MeanMaxDistancev_r[GroupNoDiffSize]/DistTarget[GroupNoDiffSize])
HL3 = np.append(HL3,MaxDistance_L.MeanMaxDistancev_r[GroupNoDiffSize]/DistTarget[GroupNoDiffSize])
MyRegressionA02 = {'HandLength': HL3,
                   'Conditions': np.append(np.repeat(['Standard','Elongated'],12),np.repeat(['Standard','Elongated'],12)),
        'Groups': np.append(np.repeat('DiffSize',12*2),np.repeat('NoDiffSize',12*2))}
df = pd.DataFrame(MyRegressionA02,columns=['HandLength','Conditions','Groups'])
mc = MultiComparison(df['HandLength'], df['Groups'])
result = mc.tukeyhsd()
print(result)
print(mc.groupsunique)
f.interaction_plot(df['Conditions'], df['Groups'],df['HandLength'],colors=['red','blue'], markers=['D','^'], ms=10)


#-------------------------------
import statsmodels.formula.api as smf
import statsmodels.api as smm
from statsmodels.formula.api import ols
HL1 = np.append(AffectedHandLengthDiff_S[GroupDiffSize]-AffectedHandLengthDiff_B[GroupDiffSize],AffectedHandLengthDiff_L[GroupDiffSize]-AffectedHandLengthDiff_B[GroupDiffSize])
HL1 = np.append(HL1,AffectedHandLengthDiff_S[GroupNoDiffSize]-AffectedHandLengthDiff_B[GroupNoDiffSize])
HL1 = np.append(HL1,AffectedHandLengthDiff_L[GroupNoDiffSize]-AffectedHandLengthDiff_B[GroupNoDiffSize])
MyRegressionA02 = {'HandLength': HL1,
                   'Conditions': np.append(np.repeat(['Standard','Elongated'],12),np.repeat(['Standard','Elongated'],12)),
        'Groups': np.append(np.repeat('DiffSize',12*2),np.repeat('NoDiffSize',12*2))}
df = pd.DataFrame(MyRegressionA02,columns=['HandLength','Conditions','Groups'])

model = ols('HandLength ~ C(Conditions)+C(Groups) + C(Conditions)*C(Groups)', data = df,missing='drop')
modeL = model.fit()
res = smm.stats.anova_lm(modeL, typ= 2)
res
   import statsmodels.stats.anova as hh
HL1 = np.append(AffectedHandLengthDiff_S[GroupDiffSize],AffectedHandLengthDiff_L[GroupDiffSize])
HL1 = np.append(HL1,AffectedHandLengthDiff_S[GroupNoDiffSize])
HL1 = np.append(HL1,AffectedHandLengthDiff_L[GroupNoDiffSize])
MyRegressionA02 = {'HandLength': HL1,
                   'Conditions': np.append(np.repeat(['Standard','Elongated'],12),np.repeat(['Standard','Elongated'],12)),
        'Groups': np.append(np.repeat('DiffSize',12*2),np.repeat('NoDiffSize',12*2))}
df = pd.DataFrame(MyRegressionA02,columns=['HandLength','Conditions','Groups'])

model = ols('HandLength ~ C(Conditions)+C(Groups) + C(Conditions)*C(Groups)', data = df,missing='drop')
modeL = model.fit()
res = smm.stats.anova_lm(modeL, typ= 2)
res
      
HL3 = np.append(MaxDistance_S.MeanMaxDistancev_r[GroupDiffSize]/DistTarget[GroupDiffSize],MaxDistance_L.MeanMaxDistancev_r[GroupDiffSize]/DistTarget[GroupDiffSize])
HL3 = np.append(HL3,MaxDistance_S.MeanMaxDistancev_r[GroupNoDiffSize]/DistTarget[GroupNoDiffSize])
HL3 = np.append(HL3,MaxDistance_L.MeanMaxDistancev_r[GroupNoDiffSize]/DistTarget[GroupNoDiffSize])
MyRegressionA02 = {'WristDist': HL3,
                   'Conditions': np.append(np.repeat(['Standard','Long'],12),np.repeat(['Standard','Long'],12)),
        'Groups': np.append(np.repeat('DiffSize',12*2),np.repeat('NoDiffSize',12*2)),
        'SubID': np.append(np.tile(MaxPeakVelocity_S.ParticipantName[GroupDiffSize],2),np.tile(MaxPeakVelocity_S.ParticipantName[GroupNoDiffSize],2)),
      }
df = pd.DataFrame(MyRegressionA02,columns=['WristDist','Conditions','Groups','SubID'])
res = hh.AnovaRM(df, 'WristDist', 'SubID', within=['Conditions'], aggregate_func=None).fit()

model = ols('WristDist ~ C(Conditions)+C(Groups) + C(Conditions)*C(Groups)', data = df,missing='drop')
modeL = model.fit()
res = smm.stats.anova_lm(modeL, typ= 2)
res
 
 
HL3 = np.append(MaxPeakVelocity_S.MaxPeakv_r[GroupDiffSize],MaxPeakVelocity_L.MaxPeakv_r[GroupDiffSize])
HL3 = np.append(HL3,MaxPeakVelocity_S.MaxPeakv_r[GroupNoDiffSize])
HL3 = np.append(HL3,MaxPeakVelocity_L.MaxPeakv_r[GroupNoDiffSize])
MyRegressionA02 = {'WristPeak': HL3,
                   'Conditions': np.append(np.repeat(['Standard','Long'],12),np.repeat(['Standard','Long'],12)),
        'Groups': np.append(np.repeat('DiffSize',12*2),np.repeat('NoDiffSize',12*2))}
df = pd.DataFrame(MyRegressionA02,columns=['WristPeak','Conditions','Groups'])

model = ols('WristPeak ~ C(Conditions)+C(Groups) + C(Conditions)*C(Groups)', data = df,missing='drop')
hh.AnovaRM(df, df['WristPeak'], subject = MaxPeakVelocity_S.ParticipantName, within=df['Conditions'], aggregate_func=None)
modeL = model.fit()
res = smm.stats.anova_lm(modeL, typ= 2)
res