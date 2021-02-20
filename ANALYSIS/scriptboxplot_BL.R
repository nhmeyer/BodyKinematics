library(ggplot2)
library(Rmisc)
library(lattice)
library(plyr)
library(dplyr)
library(ggpubr)
library(grid)
library(cowplot)
library(lmerTest)
library(car)
library(lsmeans)
library(Rmisc)
library(ggplot2)
library(DescTools)
library(Hmisc)
library(ltm)
library(ggpubr)


#BOXPLOT
library(dplyr)
library(ggplot2)
library(readxl)
#Analysis of BodyLandmark
#3weeks

#Comparison of perceived vs real upper limb dimension between patient
#1. Import Files
#IMPORT FILES---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MyData <- read_xlsx(("Data_BK_BL_OK.xlsx"),sheet = 2, range = "B1:BW200", col_names = FALSE,col_types = NULL, na = "", trim_ws = TRUE, skip = 0,progress = readxl_progress(), .name_repair = "unique");
MyData_S <- read_xlsx(("Data_BK_BL_OK.xlsx"),sheet = 3, range = "B1:BW200", col_names = FALSE,col_types = NULL, na = "", trim_ws = TRUE, skip = 0,progress = readxl_progress(), .name_repair = "unique");
MyData_L <- read_xlsx(("Data_BK_BL_OK.xlsx"),sheet = 4, range = "B1:BW200", col_names = FALSE,col_types = NULL, na = "", trim_ws = TRUE, skip = 0,progress = readxl_progress(), .name_repair = "unique");

#\\\\files2\\data\\nhmeyer\\My Documents\\PhD\\Research\\WP1\\Bodylandmark\\matlabcode\\WP1_analysis\\
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##How Many Patient Did You have so far?
numberPatientSoFar <- 7

Index = seq(from = 1,to = numberPatientSoFar,by = 1)

#Healthy Arm Data
ArmData <- MyData[Index]


#Calcul of Difference
#Difference perceived-real in healthy arm
LengthHealthy_B = seq(from = 1,to = dim(ArmData)[1],by = 1)
HandWidthDiff_B <- vector('double')
HandLengthDiff_B<- vector('double')
ArmWidthDiff_B<- vector('double')
ArmLengthDiff_B<- vector('double')
for (i in c(1:numberPatientSoFar)){
  HandWidthDiff_B[i] <-as.numeric(as.character(ArmData[i,2]))/as.numeric(as.character(ArmData[i,1]));
  ArmWidthDiff_B[i] <-as.numeric(as.character(ArmData[i,4]))/as.numeric(as.character(ArmData[i,3]))
  HandLengthDiff_B[i] <-((as.numeric(as.character(ArmData[i,6]))/as.numeric(as.character(ArmData[i,5])))+(as.numeric(as.character(ArmData[i,8]))/as.numeric(as.character(ArmData[i,7]))))/2
  ArmLengthDiff_B[i] <-((as.numeric(as.character(ArmData[i,10]))/as.numeric(as.character(ArmData[i,9])))+(as.numeric(as.character(ArmData[i,12]))/as.numeric(as.character(ArmData[i,11]))))/2
  if (i == numberPatientSoFar){
    meanHHandWidthD <- mean(HandWidthDiff_B[1:i]);
    meanHHandLengthD <- mean(HandLengthDiff_B[1:i]);
    meanHArmWidthD <- mean(ArmWidthDiff_B[1:i]);
    meanHArmLengthD <- mean(ArmLengthDiff_B[1:i])
  }
  
}
ArmData_S <- MyData_S[Index]


LengthHealthy_S = seq(from = 1,to = dim(ArmData_S)[1],by = 1)
HandWidthDiff_S <- vector('double')
HandLengthDiff_S<- vector('double')
ArmWidthDiff_S<- vector('double')
ArmLengthDiff_S<- vector('double')
for (i in c(1:numberPatientSoFar)){
  HandWidthDiff_S[i] <-as.numeric(as.character(ArmData_S[i,2]))/as.numeric(as.character(ArmData_S[i,1]));
  ArmWidthDiff_S[i] <-as.numeric(as.character(ArmData_S[i,4]))/as.numeric(as.character(ArmData_S[i,3]))
  HandLengthDiff_S[i] <-((as.numeric(as.character(ArmData_S[i,6]))/as.numeric(as.character(ArmData_S[i,5])))+(as.numeric(as.character(ArmData_S[i,8]))/as.numeric(as.character(ArmData_S[i,7]))))/2
  ArmLengthDiff_S[i] <-((as.numeric(as.character(ArmData_S[i,10]))/as.numeric(as.character(ArmData_S[i,9])))+(as.numeric(as.character(ArmData_S[i,12]))/as.numeric(as.character(ArmData_S[i,11]))))/2
  if (i == numberPatientSoFar){
    meanHHandWidthD_S <- mean(HandWidthDiff_S[1:i]);
    meanHHandLengthD_S <- mean(HandLengthDiff_S[1:i]);
    meanHArmWidthD_S <- mean(ArmWidthDiff_S[1:i]);
    meanHArmLengthD_S <- mean(ArmLengthDiff_S[1:i])
  }
  
}
ArmData_L <- MyData_L[Index]

LengthHealthy_L = seq(from = 1,to = dim(ArmData_L)[1],by = 1)
HandWidthDiff_L <- vector('double')
HandLengthDiff_L<- vector('double')
ArmWidthDiff_L<- vector('double')
ArmLengthDiff_L<- vector('double')
for (i in c(1:numberPatientSoFar)){
  HandWidthDiff_L[i] <-as.numeric(as.character(ArmData_L[i,2]))/as.numeric(as.character(ArmData_L[i,1]));
  ArmWidthDiff_L[i] <-as.numeric(as.character(ArmData_L[i,4]))/as.numeric(as.character(ArmData_L[i,3]))
  HandLengthDiff_L[i] <-((as.numeric(as.character(ArmData_L[i,6]))/as.numeric(as.character(ArmData_L[i,5])))+(as.numeric(as.character(ArmData_L[i,8]))/as.numeric(as.character(ArmData_L[i,7]))))/2
  ArmLengthDiff_L[i] <-((as.numeric(as.character(ArmData_L[i,10]))/as.numeric(as.character(ArmData_L[i,9])))+(as.numeric(as.character(ArmData_L[i,12]))/as.numeric(as.character(ArmData_L[i,11]))))/2
  if (i == numberPatientSoFar){
    meanHHandWidthD <- mean(HandWidthDiff_L[1:i]);
    meanHHandLengthD <- mean(HandLengthDiff_L[1:i]);
    meanHArmWidthD <- mean(ArmWidthDiff_L[1:i]);
    meanHArmLengthD <- mean(ArmLengthDiff_L[1:i])
  }
  
}

----------------------------------------------------------------------------------------------

plot.new()


#Boxplot Arm and Hand Length and Width
Conditions <- rep(c("Baseline","Standard","Large"),4)
MyBoxPlot <- data.frame(
  
  #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff)),
  #Groups = rep(c("Hand Width","Hand Length"),each = 32),
 # Conditions = rep(rep(1:2,each = 16),2))
  Val= c(c(HandLengthDiff_B ,HandLengthDiff_S,HandLengthDiff_L),c(ArmLengthDiff_B,ArmLengthDiff_S,ArmLengthDiff_L),c(HandWidthDiff_B ,HandWidthDiff_S,HandWidthDiff_L),c(ArmWidthDiff_B, ArmWidthDiff_S,ArmWidthDiff_L)),
  Groups = rep(c("Hand Length","Arm Length","Hand Width","Arm Width" ),each = numberPatientSoFar),
  Conditions = rep(rep(1:3,each = numberPatientSoFar),4))


  #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff), c(AffectedArmWidthDiff ,HealthyArmWidthDiff),c(AffectedArmLengthDiff, HealthyArmLengthDiff)),
  #Groups = c(rep("Hand Width",16),rep("Hand Length",16),rep("Arm Width",16),rep("Arm Length",16)),
   # Conditions = rep(c(rep("Affected Side",4),rep("Healthy Side",4)),16))
MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("Hand Length","Arm Length","Hand Width","Arm Width" ))
e <- ggplot(MyBoxPlot,aes(Groups,Val,fill = factor(Conditions)))+geom_hline(yintercept = 1, linetype="dashed")
e+geom_boxplot()+ ylab("Perceived/Real")+
  ggtitle("upper limb perception at 3 weeks")+
  theme_bw() + 
  scale_fill_manual(name="Upper Limb",values = c("grey40","grey80","grey10"),labels=c("Baseline", "Standard","Large")) +geom_jitter(position = position_dodge(width = 0.75),data =MyBoxPlot    )   
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Further dividing Length and Width for Right and Left
#Length Right
#Right
plot.new()


#Boxplot Arm and Hand Length and Width
Conditions <- rep(c("Baseline","Standard","Large"),4)
MyBoxPlot <- data.frame(
  
  #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff)),
  #Groups = rep(c("Hand Width","Hand Length"),each = 32),
  # Conditions = rep(rep(1:2,each = 16),2))
  Val= c(c(R_AffectedHandLengthDiff ,R_HealthyHandLengthDiff),c(R_AffectedArmLengthDiff, R_HealthyArmLengthDiff)),
  Groups = rep(c("Hand Length","Arm Length" ),each = 12),
  Conditions = rep(rep(1:2,each = 6),2))


#Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff), c(AffectedArmWidthDiff ,HealthyArmWidthDiff),c(AffectedArmLengthDiff, HealthyArmLengthDiff)),
#Groups = c(rep("Hand Width",16),rep("Hand Length",16),rep("Arm Width",16),rep("Arm Length",16)),
# Conditions = rep(c(rep("Affected Side",4),rep("Healthy Side",4)),16))
MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("Hand Length","Arm Length" ))
e <- ggplot(MyBoxPlot,aes(Groups,Val,fill = factor(Conditions)))+geom_hline(yintercept = 1, linetype="dashed")
e+geom_boxplot()+ ylab("Perceived/Real")+
  ggtitle("UL length perception in LBD patients at 3 weeks")+
  theme_bw() + 
  scale_fill_manual(name="Upper Limb",values = c("grey40","grey80"),labels=c("Affected", "Healthy")) +geom_jitter(position = position_dodge(width = 0.75),data =MyBoxPlot    )   
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Further dividing Length and Width for Right and Left
#Width Right
#Right
plot.new()


#Boxplot Arm and Hand Length and Width
Conditions <- rep(c("Affected Side","Healthy Side"),4)
MyBoxPlot <- data.frame(
  
  #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff)),
  #Groups = rep(c("Hand Width","Hand Length"),each = 32),
  # Conditions = rep(rep(1:2,each = 16),2))
  Val= c(c(R_AffectedHandWidthDiff ,R_HealthyHandWidthDiff),c(R_AffectedArmWidthDiff, R_HealthyArmWidthDiff)),
         Groups = rep(c("Hand Width","Arm Width"),each = 12),
         Conditions = rep(rep(1:2,each = 6),2))
  
  
  #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff), c(AffectedArmWidthDiff ,HealthyArmWidthDiff),c(AffectedArmLengthDiff, HealthyArmLengthDiff)),
  #Groups = c(rep("Hand Width",16),rep("Hand Length",16),rep("Arm Width",16),rep("Arm Length",16)),
  # Conditions = rep(c(rep("Affected Side",4),rep("Healthy Side",4)),16))
  MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                            levels = c("Hand Width","Arm Width" ))
  e <- ggplot(MyBoxPlot,aes(Groups,Val,fill = factor(Conditions)))+geom_hline(yintercept = 1, linetype="dashed")
  e+geom_boxplot()+ ylab("Perceived/Real")+
    ggtitle("UL width perception in LBD patients at 3 weeks")+
    theme_bw() + 
    scale_fill_manual(name="Upper Limb",values = c("grey40","grey80"),labels=c("Affected", "Healthy")) +geom_jitter(position = position_dodge(width = 0.75),data =MyBoxPlot    )   
  #--------------------------------------------------------------------------------------------------------------------------------------------------------
  #Length Left
  #Left
  plot.new()
  
  
  #Boxplot Arm and Hand Length and Width
 # Conditions <- rep(c("Affected Side","Healthy Side"),4)
  MyBoxPlot <- data.frame(
    
    #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff)),
    #Groups = rep(c("Hand Width","Hand Length"),each = 32),
    # Conditions = rep(rep(1:2,each = 16),2))
    Val= c(c(L_AffectedHandLengthDiff ,L_HealthyHandLengthDiff),c(L_AffectedArmLengthDiff, L_HealthyArmLengthDiff)),
           Groups = rep(c("Hand Length","Arm Length" ),each = 20),
           Conditions = rep(rep(1:2,each = 10),2))
    
    
    #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff), c(AffectedArmWidthDiff ,HealthyArmWidthDiff),c(AffectedArmLengthDiff, HealthyArmLengthDiff)),
    #Groups = c(rep("Hand Width",16),rep("Hand Length",16),rep("Arm Width",16),rep("Arm Length",16)),
    # Conditions = rep(c(rep("Affected Side",4),rep("Healthy Side",4)),16))
    MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                              levels = c("Hand Length","Arm Length" ))
    e <- ggplot(MyBoxPlot,aes(Groups,Val,fill = factor(Conditions)))+geom_hline(yintercept = 1, linetype="dashed")
    e+geom_boxplot()+ ylab("Perceived/Real")+
      ggtitle("UL length perception in RBD patients at 3 weeks")+
      theme_bw() + 
      scale_fill_manual(name="Upper Limb",values = c("grey40","grey80"),labels=c("Affected", "Healthy")) +geom_jitter(position = position_dodge(width = 0.75),data =MyBoxPlot    )   
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Further dividing Length and Width for Right and Left
    #Width Left
    #Left
    plot.new()
    
    
    #Boxplot Arm and Hand Length and Width
    Conditions <- rep(c("Affected Side","Healthy Side"),4)
    MyBoxPlot <- data.frame(
      
      #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff)),
      #Groups = rep(c("Hand Width","Hand Length"),each = 32),
      # Conditions = rep(rep(1:2,each = 16),2))l
      Val= c(c(L_AffectedHandWidthDiff ,L_HealthyHandWidthDiff),c(L_AffectedArmWidthDiff, L_HealthyArmWidthDiff)),
             Groups = rep(c("Hand Width","Arm Width"),each = 20),
             Conditions = rep(rep(1:2,each = 10),2))
      
      
      #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff), c(AffectedArmWidthDiff ,HealthyArmWidthDiff),c(AffectedArmLengthDiff, HealthyArmLengthDiff)),
      #Groups = c(rep("Hand Width",16),rep("Hand Length",16),rep("Arm Width",16),rep("Arm Length",16)),
      # Conditions = rep(c(rep("Affected Side",4),rep("Healthy Side",4)),16))
      MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                                levels = c("Hand Width","Arm Width" ))
      e <- ggplot(MyBoxPlot,aes(Groups,Val,fill = factor(Conditions)))+geom_hline(yintercept = 1, linetype="dashed")
      e+geom_boxplot()+ ylab("Perceived/Real")+
        ggtitle("UL width perception in RBD patients at 3 weeks")+
        theme_bw() + 
        scale_fill_manual(name="Upper Limb",values = c("grey40","grey80"),labels=c("Affected", "Healthy")) +geom_jitter(position = position_dodge(width = 0.75),data =MyBoxPlot    )   
      
      #---------------------------------------------------------------------------------------------------------------------------------------------
      #Evolution 3weeks and 3 months
      #divided by length and Width
    
      plot.new()
      
      
      #Boxplot Arm and Hand Length and Width
      Conditions <- rep(c("Affected Side","Healthy Side"),4)
      MyBoxPlot <- data.frame(
        
        #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff)),
        #Groups = rep(c("Hand Width","Hand Length"),each = 32),
        # Conditions = rep(rep(1:2,each = 16),2))
        Val= c(c(A3W3M_AffectedHandLengthDiff ,H3W3M_HealthyHandLengthDiff,AffectedHandLengthDiff_3M,HealthyHandLengthDiff_3M),c(A3W3M_AffectedArmLengthDiff, H3W3M_HealthyArmLengthDiff,AffectedArmLengthDiff_3M,HealthyArmLengthDiff_3M)),
        Groups = rep(c("Hand Length","Arm Length" ),each = 24),
        Conditions = rep(rep(1:4,each = 6),2))
      
      
      #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff), c(AffectedArmWidthDiff ,HealthyArmWidthDiff),c(AffectedArmLengthDiff, HealthyArmLengthDiff)),
      #Groups = c(rep("Hand Width",16),rep("Hand Length",16),rep("Arm Width",16),rep("Arm Length",16)),
      # Conditions = rep(c(rep("Affected Side",4),rep("Healthy Side",4)),16))
      MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                                levels = c("Hand Length","Arm Length" ))
      e <- ggplot(MyBoxPlot,aes(Groups,Val,fill = factor(Conditions)))+geom_hline(yintercept = 1, linetype="dashed")
      e+geom_boxplot()+ ylab("Perceived/Real")+
        ggtitle("Evolution at 3 months UL length perception ")+
        theme_bw() + 
        scale_fill_manual(name="Upper Limb",values = c("grey50","grey80","grey20","grey100"),labels=c("Affected 3W", "Healthy 3W","Affected 3M", "Healthy 3M")) +geom_jitter(position = position_dodge(width = 0.75),data =MyBoxPlot    )   
      #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
      #    
      plot.new()
      
      
      #Boxplot Arm and Hand Length and Width
      Conditions <- rep(c("Affected Side","Healthy Side"),4)
      MyBoxPlot <- data.frame(
        
        #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff)),
        #Groups = rep(c("Hand Width","Hand Length"),each = 32),
        # Conditions = rep(rep(1:2,each = 16),2))
        Val= c(c(A3W3M_AffectedHandWidthDiff ,H3W3M_HealthyHandWidthDiff,AffectedHandWidthDiff_3M,HealthyHandWidthDiff_3M),c(A3W3M_AffectedArmWidthDiff, H3W3M_HealthyArmWidthDiff,AffectedArmWidthDiff_3M,HealthyArmWidthDiff_3M)),
        Groups = rep(c("Hand Width","Arm Width" ),each = 24),
        Conditions = rep(rep(1:4,each = 6),2))
      
      
      #Val= c(c(AffectedHandWidthDiff ,HealthyHandWidthDiff),c(AffectedHandLengthDiff, HealthyHandLengthDiff), c(AffectedArmWidthDiff ,HealthyArmWidthDiff),c(AffectedArmLengthDiff, HealthyArmLengthDiff)),
      #Groups = c(rep("Hand Width",16),rep("Hand Length",16),rep("Arm Width",16),rep("Arm Length",16)),
      # Conditions = rep(c(rep("Affected Side",4),rep("Healthy Side",4)),16))
      MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                                levels = c("Hand Width","Arm Width" ))
      e <- ggplot(MyBoxPlot,aes(Groups,Val,fill = factor(Conditions)))+geom_hline(yintercept = 1, linetype="dashed")
      e+geom_boxplot()+ ylab("Perceived/Real")+
        ggtitle("Evolution at 3 months UL width perception")+
        theme_bw() + 
        scale_fill_manual(name="Upper Limb",values = c("grey50","grey80","grey20","grey100"),labels=c("Affected 3W", "Healthy 3W","Affected 3M", "Healthy 3M")) +geom_jitter(position = position_dodge(width = 0.75),data =MyBoxPlot    )   
      #---