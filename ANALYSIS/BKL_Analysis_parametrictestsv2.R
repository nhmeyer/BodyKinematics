library(dplyr)
library(ggplot2)
library(readxl)
library(Hmisc)
library(stats)
library(car)
library(nnet)
library(MASS)
library(coxme)
library(survey)
library(lme4)
library(rlang)
library(readxl)
library(ggplot2)
library(ggpubr)
library(MASS)
#Analysis of BodyLandmark
#3weeks

#Comparison of perceived vs real upper limb dimension between patient
#1. Import Files
#IMPORT FILES---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
WristAccuracy_L <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 15, range = "B1:E25", col_types = c("numeric","numeric","numeric","numeric"),na = "");
PinchDistance_L <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 17, range = "B1:E25", col_types = c("numeric","numeric","numeric","numeric"),na = "");
Embodiment_L <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 18, range = "B1:G25",col_types = c("numeric","numeric","numeric","numeric","numeric","numeric"),na = "");
SelfConf_L <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 19, range = "B1:C25", col_types = c("numeric","numeric"),na = "");
MaxPeakVelocity_S <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 2, range = "B1:E25", col_types = c("numeric","numeric","numeric","numeric"),na = "");
MaxLatencyVelocity_S <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 3, range = "B1:E25", col_types = c("numeric","numeric","numeric","numeric"),na = "");
MovementDuration_S <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 4, range = "B1:E25", col_types = c("numeric","numeric","numeric","numeric"),na = "");
MaxDistance_S <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 5, range = "B1:E25", col_types = c("numeric","numeric","numeric","numeric"),na = "");
WristAccuracy_S <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 6, range = "B1:E25", col_types = c("numeric","numeric","numeric","numeric"),na = "");
PinchDistance_S <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 8, range = "B1:E25", col_types = c("numeric","numeric","numeric","numeric"),na = "");
Embodiment_S <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 9, range = "B1:G25",col_types = c("numeric","numeric","numeric","numeric","numeric","numeric"),na = "");
SelfConf_S <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 10, range = "B1:C25",  col_types = c("numeric","numeric"),na = "");
MaxPeakVelocity_L <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 11, range = "B1:E25",  col_types = c("numeric","numeric","numeric","numeric"),na = "");
MaxLatencyVelocity_L <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 12, range = "B1:E25",  col_types = c("numeric","numeric","numeric","numeric"),na = "");
MovementDuration_L <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 13, range = "B1:E25",  col_types = c("numeric","numeric","numeric","numeric"),na = "");
MaxDistance_L <- read_xlsx(("BKL_results_181119.xlsx"),sheet = 14, range = "B1:E25",  col_types = c("numeric","numeric","numeric","numeric"),na = "");
DistanceTarget <-read_xlsx(("BKL_results_181119.xlsx"),sheet = 20, range = "B1:B25",  col_types = c("numeric"),na = "");


Ownership_S <-vector('double')
Ownership_L <-vector('double')
Agency_S <-vector('double')
Agency_L <-vector('double')
Size_S <-vector('double')
Size_L <-vector('double')
Control_S <-vector('double')
Control_L <-vector('double')
ObjectSize_S <-vector('double')
ObjectSize_L <-vector('double')
ObjectDistance_S <-vector('double')
ObjectDistance_L <-vector('double')
DistanceTarget_ <- vector('double')
DistanceTarget_ <- as.numeric(unlist(DistanceTarget[,1]))
WristAcc_L <- as.numeric(unlist(WristAccuracy_L[,1]))
WristAcc_S <- as.numeric(unlist(WristAccuracy_S[,1]))
MaxDist_S <- as.numeric(unlist(MaxDistance_S[,1]))
MaxDist_L <- as.numeric(unlist(MaxDistance_L[,1]))
SC_S <- as.numeric(unlist(SelfConf_S[,1]))
SC_L <- as.numeric(unlist(SelfConf_L[,1]))
MaxP_S <- as.numeric(unlist(MaxPeakVelocity_S[,1]))
MaxP_L <- as.numeric(unlist(MaxPeakVelocity_L[,1]))
MaxLV_S <- as.numeric(unlist(MaxLatencyVelocity_S[,1]))
MaxLV_L <- as.numeric(unlist(MaxLatencyVelocity_L[,1]))
MDuration_S <- as.numeric(unlist(MovementDuration_S[,1]))
MDuration_L <- as.numeric(unlist(MovementDuration_L[,1]))
PinchD_S <- as.numeric(unlist(PinchDistance_S[,1]))
PinchD_L <- as.numeric(unlist(PinchDistance_L[,1]))

# with angle discrimination (A0,A25 and AM25)
WristAcc_L_A <- as.numeric(unlist(WristAccuracy_L[,2:4]))
WristAcc_S_A <- as.numeric(unlist(WristAccuracy_S[,2:4]))
MaxDist_S_A <- as.numeric(unlist(MaxDistance_S[,2:4]))
MaxDist_L_A <- as.numeric(unlist(MaxDistance_L[,2:4]))
MaxP_S_A <- as.numeric(unlist(MaxPeakVelocity_S[,2:4]))
MaxP_L_A <- as.numeric(unlist(MaxPeakVelocity_L[,2:4]))
MaxLV_S_A <- as.numeric(unlist(MaxLatencyVelocity_S[,2:4]))
MaxLV_L_A <- as.numeric(unlist(MaxLatencyVelocity_L[,2:4]))
MDuration_S_A <- as.numeric(unlist(MovementDuration_S[,2:4]))
MDuration_L_A <- as.numeric(unlist(MovementDuration_L[,2:4]))



Ownership_S <-as.numeric(unlist(Embodiment_S[,1]))
Ownership_L <-as.numeric(unlist(Embodiment_L[,1]))
Agency_S <-as.numeric(unlist(Embodiment_S[,2]))
Agency_L <-as.numeric(unlist(Embodiment_L[,2]))
Size_S <-as.numeric(unlist(Embodiment_S[,3]))
Size_L <-as.numeric(unlist(Embodiment_L[,3]))
Control_S <-as.numeric(unlist(Embodiment_S[,4]))
Control_L <-as.numeric(unlist(Embodiment_L[,4]))
ObjectSize_S <-as.numeric(unlist(Embodiment_S[,5]))
ObjectSize_L <-as.numeric(unlist(Embodiment_L[,5]))
ObjectDistance_S <-as.numeric(unlist(Embodiment_S[,6]))
ObjectDistance_L <-as.numeric(unlist(Embodiment_L[,6]))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#data visualization
#1. Embodiment
plot.new()



Conditions <- rep(c("Standard","Long"),4)
MyBoxPlot <- data.frame(
  Tal = c(c(t(Ownership_S),t(Ownership_L)),c(t(Agency_S),t(Agency_L)),c(t(Size_S),t(Size_L)),c(t(Control_S),t(Control_L))),
  Groups = rep(c("Ownership","Agency","Size","Control",4 ),each = dim(Embodiment_S)[1]*2),
  Conditions = rep(rep(1:2,each = dim(Embodiment_S)[1]),4),
  ID = c(rep(1:24,2),rep(25:48,2),rep(49:72,2),rep(73:96,2)))

MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("Ownership","Agency","Size","Control",4))

e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions)))+   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) # +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) 
e+geom_point()+ylab("ratings")+xlab("Conditions")+scale_x_discrete(labels = c("O","O","A","A","S","S","C","C"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Embodiment")#+
  #stat_compare_means(method = "t.test")
#stat_compare_means(method = "anova")+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plot.new()
MyBoxPlot_individual <- data.frame(
  #Tal = c(c(mean(t(Ownership_S)),mean(t(Ownership_L)),c(t(Agency_S),t(Agency_L)),c(t(Size_S),t(Size_L)),c(t(Control_S),t(Control_L)),c(t(ObjectSize_S),t(ObjectSize_L))),
  Tall = c(c(t(Ownership_S),t(Ownership_L)),c(t(Agency_S),t(Agency_L)),c(t(Size_S),t(Size_L)),c(t(Control_S),t(Control_L)),c(t(ObjectSize_S),t(ObjectSize_L)))
  
  Groups = rep(c("Ownership","Agency","Size","Control"),each = 48),
  Conditions = rep(rep(1:2,each = 24),4))


#Boxplot Arm and Hand Length and Width
Conditions <- rep(c("Standard","Long"),4)
MyBoxPlot <- data.frame(
  #Tal = c(c(mean(t(Ownership_S)),mean(t(Ownership_L)),c(t(Agency_S),t(Agency_L)),c(t(Size_S),t(Size_L)),c(t(Control_S),t(Control_L)),c(t(ObjectSize_S),t(ObjectSize_L))),
  Tal = c(c(mean(Ownership_S,na.rm=T),mean(Ownership_L,na.rm=T)),c(mean(Agency_S,na.rm=T),mean(Agency_L,na.rm=T)),c(mean(Size_S,na.rm=T),mean(Size_L,na.rm=T)),c(mean(Control_S,na.rm=T),mean(Control_L,na.rm=T)),c(mean(t(ObjectSize_S)),mean(ObjectSize_L,na.rm=T))),
          
  Groups = rep(c("Ownership","Agency","Size","Control","BallSize" ),each = 2),
  Conditions = rep(rep(1:2,each = 1),5))

MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("Ownership","Agency","Size","Control","BallSize"))
e <- ggplot(MyBoxPlot,aes(Groups,Tal,fill = factor(Conditions)))
e+geom_bar(stat = "identity",position=position_dodge())+ ylab("ratings")+
  ggtitle("Embodiment after Standard and Long conditions")+
  theme_bw() +geom_errorbar(aes(ymin=Tal-sd, ymax=Tal+sd), width=.2,position=position_dodge(.9))+

  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long")) +geom_jitter(position = position_dodge(width = 0.75),data =MyBoxPlot_individual)  
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#2. Object Distance

# 
plot.new()

Conditions <- c("Standard","Long")
MyBoxPlot <- data.frame(
  Tal = c(ObjectDistance_S,ObjectDistance_L),
  Groups = rep(c("ObjectDistance" ),each = length(ObjectDistance_S)*2),
  Conditions = rep(1:2,each = length(ObjectDistance_S)),
  ID = c(rep(1:24,2)))


MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("ObjectDistance"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+ylab("Distance [cm]")+xlab("Conditions")+scale_x_discrete(labels =c("Standard","Long"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Target Distance")+geom_point()+
  stat_compare_means(method = "t.test")

plot.new()


#Boxplot Arm and Hand Length and Width
Conditions <- rep(c("diff Standard-Long"),1)
MyBoxPlot <- data.frame(
  Tal = ObjectDistance_S - ObjectDistance_L,
    Groups = rep(c("BallDistance" ),each = dim(Embodiment_S)[1]*1),
  Conditions = rep(rep(1,each = dim(Embodiment_S)[1]),1),
ID = c(rep(1:24,1)))

MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("Object Distance"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+geom_point()+ylab("Distance [dm]")+xlab("Conditions")+scale_x_discrete(labels = c("Standard","Long"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Object Distance")
  

#H0 no difference of subjective ratings between perceived distance of object in condition 1 and 2 (the mean of the distance in cond 2 - distance in cond 1 = 0)
h<-t.test(Tal,mu = 0)
e+annotate("text",x=0.65, y=60,label=h$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Object Distance")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#3. Max Distance


plot.new()

Conditions <- c("Standard","Long")
MyBoxPlot <- data.frame(
  Tal = c(MaxDist_S/DistanceTarget_,MaxDist_L/DistanceTarget_),
  Groups = rep(c("MaxDistance" ),each = length(MaxDist_L)*2),
  Conditions = rep(1:2,each = length(MaxDist_L)),
  ID = c(rep(1:24,2)))

MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("MaxDistance"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions)))  +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) #+   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7)
e+geom_point()+ylab("cross Distance / target distance")+xlab("Conditions")+scale_x_discrete(labels = c("Standard","Long"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("cross Distance")#+
  #stat_compare_means(method = "t.test")
# H0 the mean of the Max Distance is not different between conditions, paired t-test because conditions are done within subject
h<-t.test(MaxDist_S-MaxDist_L)
e+annotate("text",x=0.65, y=0.25,label=h$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle(" Distance")
  #-


#3.1 With Angle
plot.new()

Conditions <- rep(c("Standard","Long"),3)
MyBoxPlot <- data.frame(
  Tal = c(c(MaxDist_S_A[1:length(MaxDist_L)]/DistanceTarget_,MaxDist_L_A[1:length(MaxDist_L)]/DistanceTarget_),c(MaxDist_S_A[(length(MaxDist_L)+1):(2*length(MaxDist_L))]/DistanceTarget_,MaxDist_L_A[(length(MaxDist_L)+1):(2*length(MaxDist_L))]/DistanceTarget_),c(MaxDist_S_A[(2*length(MaxDist_L)+1):(3*length(MaxDist_L))]/DistanceTarget_,MaxDist_L_A[(2*length(MaxDist_L)+1):(3*length(MaxDist_L))]/DistanceTarget_)),
  Groups = rep(c("A0" ,"A25","AM25"),each = length(MaxDist_L)*2),
  Conditions = rep(rep(1:2,each = length(MaxDist_L)),3),
ID = c(rep(1:24,2),rep(25:48,2),rep(49:72,2)))

MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("A0" ,"A25","AM25"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions)))+   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5)  +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) 
e+geom_point()+ylab("Cross Distance/Distance Target")+xlab("Conditions")+scale_x_discrete(labels = c("A0","A0","A25","A25","AM25","AM25"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Distance")
# H0 the mean of the difference between Cond2 and 1 is 0
h1<-t.test(MaxDist_S_A[1:length(MaxDist_L)]-MaxDist_L_A[1:length(MaxDist_L)],mu = 0)
h2<-t.test(MaxDist_S_A[(length(MaxDist_L)+1):(2*length(MaxDist_L))]-MaxDist_L_A[(length(MaxDist_L)+1):(2*length(MaxDist_L))],mu = 0)
h3<-t.test(MaxDist_S_A[(2*length(MaxDist_L)+1):(3*length(MaxDist_L))]-MaxDist_L_A[(2*length(MaxDist_L)+1):(3*length(MaxDist_L))],mu = 0)

e+#annotate("text",x=2, y=0.3,label=h1$p.value)+#annotate("text",x=4, y=0.3,label=h2$p.value)+annotate("text",x=5.5, y=0.3,label=h3$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle(" Distance")+ylab("Cross Distance/Distance Target")+xlab("Conditions")+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))

#-----------------------------------------------------------------------------------------------------------------------------------------
#4. Max Peak
plot.new()

Conditions <- c("Standard","Long")
MyBoxPlot <- data.frame(
  Tal = c(MaxP_S,MaxP_L),
  Groups = rep(c("MaxPeakVelocity" ),each = length(MaxP_L)*2),
  Conditions = rep(1:2,each = length(MaxP_L)),
  ID = c(rep(1:24,2)))


MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("MaxPeakVelocity"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+ylab("Peak Velocity [m/s]")+xlab("Conditions")+scale_x_discrete(labels =c("Standard","Long"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("peak Velocity")+geom_point()+
  stat_compare_means(method = "t.test")

#H0 mean of the difference between Cond 2 and 1 is 0
h1<-t.test(MaxP_L-MaxP_S,mu=0)
e+annotate("text",x=1.5, y=1.3,label=h1$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle(" Peak Velocity")+ylab("Peak Velocity [m/s]")#+xlab("Standard","Long")
#4.1 With Angle
plot.new()

Conditions <- rep(c("Standard","Long"),3)
MyBoxPlot <- data.frame(
  Tal = c(c(MaxP_S_A[1:length(MaxP_L)],MaxP_L_A[1:length(MaxP_L)]),c(MaxP_S_A[(length(MaxP_L)+1):(2*length(MaxP_L))],MaxP_L_A[(length(MaxP_L)+1):(2*length(MaxP_L))]),c(MaxP_S_A[(2*length(MaxP_L)+1):(3*length(MaxP_L))],MaxP_L_A[(2*length(MaxP_L)+1):(3*length(MaxP_L))])),
  Groups = rep(c("A0" ,"A25","AM25"),each = length(MaxP_L)*2),
  Conditions = rep(rep(1:2,each = length(MaxP_L)),3),

ID = c(rep(1:24,2),rep(25:48,2),rep(49:72,2)))

MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("A0" ,"A25","AM25"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+ylab("Velocity [m/s]")+xlab("Conditions")+scale_x_discrete(labels = c("A0","A0","A25","A25","AM25","AM25"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Max Peak Velocity")+geom_point()
 
  
  # H0 the mean of the difference between Cond2 and 1 is 0
h1<-t.test(MaxP_S_A[1:length(MaxP_L)]-MaxP_L_A[1:length(MaxP_L)],mu = 0)
h2<-t.test(MaxP_S_A[(length(MaxP_L)+1):(2*length(MaxP_L))]-MaxP_L_A[(length(MaxP_L)+1):(2*length(MaxP_L))],mu = 0)
h3<-t.test(MaxP_S_A[(2*length(MaxP_L)+1):(3*length(MaxP_L))]-MaxP_L_A[(2*length(MaxP_L)+1):(3*length(MaxP_L))],mu = 0)

e+annotate("text",x=2, y=1.4,label=h1$p.value)+annotate("text",x=4, y=1.4,label=h2$p.value)+annotate("text",x=6, y=1.4,label=h3$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle(" Peak Velocity")

#-----------------------------------------------------------------------------------------------------------------------------------------
#5. Max Peak latency
plot.new()

Conditions <- c("Standard","Long")
MyBoxPlot <- data.frame(
  Tal = c(MaxLV_S,MaxLV_L),
  Groups = rep(c("MaxPeakLatency" ),each = length(MaxLV_L)*2),
  Conditions = rep(1:2,each = length(MaxLV_L)),
  ID = c(rep(1:24,2)))

MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("MaxPeakLatency"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+geom_point()+ylab("Time [s]")+xlab("Conditions")+scale_x_discrete(labels =c("Standard","Long"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("peak Latency")+
  stat_compare_means(method = "t.test")
# H0 the mean of the difference between Cond2 and 1 is 0
h1<-t.test(MaxLV_S[1:length(MaxLV_L)]-MaxLV_L[1:length(MaxLV_L)],mu = 0)

e+annotate("text",x=1.5, y=0.7,label=h1$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle(" Peak Latency")

#5.1 With Angle
plot.new()

Conditions <- rep(c("Standard","Long"),3)
MyBoxPlot <- data.frame(
  Tal = c(c(MaxLV_S_A[1:length(MaxLV_L)],MaxLV_L_A[1:length(MaxLV_L)]),c(MaxLV_S_A[(length(MaxLV_L)+1):(2*length(MaxLV_L))],MaxLV_L_A[(length(MaxLV_L)+1):(2*length(MaxLV_L))]),c(MaxLV_S_A[(2*length(MaxLV_L)+1):(3*length(MaxLV_L))],MaxLV_L_A[(2*length(MaxLV_L)+1):(3*length(MaxLV_L))])),
  Groups = rep(c("A0" ,"A25","AM25"),each = length(MaxLV_L)*2),
  Conditions = rep(rep(1:2,each = length(MaxLV_L)),3),
  ID = c(rep(1:24,2),rep(25:48,2),rep(49:72,2)))

MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("A0" ,"A25","AM25"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+geom_point()+ylab("Time [s]")+xlab("Conditions")+scale_x_discrete(labels = c("A0","A0","A25","A25","AM25","AM25"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Max Peak Latency")+
  # H0 the mean of the difference between Cond2 and 1 is 0
h1<-t.test(MaxLV_S_A[1:length(MaxLV_L)]-MaxLV_L_A[1:length(MaxLV_L)],mu = 0)
h2<-t.test(MaxLV_S_A[(length(MaxLV_L)+1):(2*length(MaxLV_L))]-MaxLV_L_A[(length(MaxLV_L)+1):(2*length(MaxLV_L))],mu = 0)
h3<-t.test(MaxLV_S_A[(2*length(MaxLV_L)+1):(3*length(MaxLV_L))]-MaxLV_L_A[(2*length(MaxLV_L)+1):(3*length(MaxLV_L))],mu = 0)

e+annotate("text",x=2, y=0.7,label=h1$p.value)+annotate("text",x=4, y=0.7,label=h2$p.value)+annotate("text",x=6, y=0.7,label=h3$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle(" Peak Latency")

#-----------------------------------------------------------------------------------------------------------------------------------------
#6. Movement Duration
plot.new()

Conditions <- c("Standard","Long")
MyBoxPlot <- data.frame(
  Tal = c(MDuration_S,MDuration_L),
  Groups = rep(c("Movement Duration" ),each = length(MDuration_L)*2),
  Conditions = rep(1:2,each = length(MDuration_L)),
ID = c(rep(1:24,2)))
MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("Movement Duration"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+geom_point()+ylab("Time [s]")+xlab("Conditions")+scale_x_discrete(labels =c("Standard","Long"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Movement Duration")+
  
# H0 the mean of the difference between Cond2 and 1 is 0
h1<-t.test(MDuration_S-MDuration_L,mu = 0)
e+annotate("text",x=1.5, y=1.15,label=h1$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Movement Duration")

#6.1 Movement Duration Angle
plot.new()

Conditions <- rep(c("Standard","Long"),3)
MyBoxPlot <- data.frame(
  Tal = c(c(MDuration_S_A[1:length(MDuration_L)],MDuration_L_A[1:length(MDuration_L)]),c(MDuration_S_A[(length(MDuration_L)+1):(2*length(MDuration_L))],MDuration_L_A[(length(MDuration_L)+1):(2*length(MDuration_L))]),c(MDuration_S_A[(2*length(MDuration_L)+1):(3*length(MDuration_L))],MDuration_L_A[(2*length(MDuration_L)+1):(3*length(MDuration_L))])),
  Groups = rep(c("A0" ,"A25","AM25"),each = length(MDuration_L)*2),
  Conditions = rep(rep(1:2,each = length(MDuration_L)),3),
  ID = c(rep(1:24,2),rep(25:48,2),rep(49:72,2)))

MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("A0" ,"A25","AM25"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+geom_point()+ylab("Time [s]")+xlab("Conditions")+scale_x_discrete(labels = c("A0","A0","A25","A25","AM25","AM25"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Movement Duration")+
  # H0 the mean of the difference between Cond2 and 1 is 0
h1<-t.test(MDuration_S_A[1:length(MDuration_L)]-MDuration_L_A[1:length(MDuration_L)],mu = 0)
h2<-t.test(MDuration_S_A[(length(MDuration_L)+1):(2*length(MDuration_L))]-MDuration_L_A[(length(MDuration_L)+1):(2*length(MDuration_L))],mu = 0)
h3<-t.test(MDuration_S_A[(2*length(MDuration_L)+1):(3*length(MDuration_L))]-MDuration_L_A[(2*length(MDuration_L)+1):(3*length(MDuration_L))],mu = 0)

e+annotate("text",x=2, y=1.15,label=h1$p.value)+annotate("text",x=4, y=1.15,label=h2$p.value)+annotate("text",x=6, y=1.15,label=h3$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Movement Duration")
#

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#7. PinchDistance
plot.new()

Conditions <- c("Standard","Long")
MyBoxPlot <- data.frame(
  Tal = c(PinchD_S,PinchD_L),
  Groups = rep(c("PinchDistance" ),each = length(PinchD_L)*2),
  Conditions = rep(1:2,each = length(PinchD_L)),
  ID = c(rep(1:24,2)))
MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("Movement Duration"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+geom_point()+ylab("Time [s]")+xlab("Conditions")+scale_x_discrete(labels =c("Standard","Long"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Pinch Distance")+
  # H0 the mean of the difference between Cond2 and 1 is 0
h1<-t.test(PinchD_S-PinchD_L,mu = 0)
e+annotate("text",x=1.5, y=60,label=h1$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Pinch Distance")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#8. SelfConf
plot.new()

Conditions <- c("Standard","Long")
MyBoxPlot <- data.frame(
  Tal = c(SC_S,SC_L),
  Groups = rep(c("SelfConfidence" ),each = length(SC_L)*2),
  Conditions = rep(1:2,each = length(SC_L)),
  ID = c(rep(1:24,2)))

MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
                          levels = c("SelfConfidence"))
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+geom_point()+ylab("ratings")+xlab("Conditions")+scale_x_discrete(labels = c("Standard","Long"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("SelfConfidence")
 
  # H0 the mean of the difference between Cond2 and 1 is 0
  h1<-t.test(SC_S-SC_L,mu = 0)
e+annotate("text",x=1.5, y=1.15,label=h1$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("SelfConf")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#9. WristAccuracy
plot.new()

Conditions <- c("Standard","Long")
MyBoxPlot <- data.frame(
  Tal = c(WristAcc_S,WristAcc_L),
  Groups = rep(c("Wrist Accuracy" ),each = length(WristAcc_L)*2),
  Conditions = rep(1:2,each = length(WristAcc_L)),
  ID = c(rep(1:24,2)))


MyBoxPlot$Groups <-factor(MyBoxPlot$Groups, 
     levels = c("Wrist Accuracy"))

e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+geom_point()+ylab("Distance [dm]")+xlab("Conditions")+scale_x_discrete(labels = c("Standard","Long"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Wrist Accuracy")
# H0 the mean of the difference between Cond2 and 1 is 0
h1<-t.test(WristAcc_S-WristAcc_L,mu = 0)
e+annotate("text",x=1.5, y=0.65,label=h1$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("WristAccuracy")



   #geom_path(aes(group = Conditions,color = ID),data = MyBoxPlot,position = "jitter")
#9.1 Wrist Accuracy per target
plot.new()
Conditions <- rep(c("Standard","Long"),3)
Tal = c(c(WristAcc_S_A[1:length(WristAcc_L)],WristAcc_L_A[1:length(WristAcc_L)]),c(WristAcc_S_A[(length(WristAcc_L)+1):(2*length(WristAcc_L))],WristAcc_L_A[(length(WristAcc_L)+1):(2*length(WristAcc_L))]),c(WristAcc_S_A[(2*length(WristAcc_L)+1):(3*length(WristAcc_L))],WristAcc_L_A[(2*length(WristAcc_L)+1):(3*length(WristAcc_L))]))
Groups = rep(c("A0" ,"A25","AM25"),each = length(WristAcc_L)*2)
Conditions = rep(rep(1:2,each = length(WristAcc_L)),3)
ID = c(rep(1:24,2),rep(25:48,2),rep(49:72,2))
MyBoxPlot <- data.frame(Tal,Groups,Conditions,ID)
e<- ggplot(MyBoxPlot, aes(x = interaction(Conditions,Groups), y = Tal,fill = factor(Conditions))) +   geom_line(aes(group = interaction(ID,Groups)), size = 0.05, alpha = 0.7) +   geom_boxplot(aes(fill = factor(Conditions)), alpha = 0.5) 
e+geom_point()+ylab("Distance [dm]")+xlab("Conditions")+scale_x_discrete(labels = c("A0","A0","A25","A25","AM25","AM25"))+
  scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  ggtitle("Wrist Accuracy")+theme_bw()
# H0 the mean of the difference between Cond2 and 1 is 0
h1<-t.test(WristAcc_S_A[1:length(WristAcc_L)]-WristAcc_L_A[1:length(WristAcc_L)],mu = 0)
h2<-t.test(WristAcc_S_A[(length(WristAcc_L)+1):(2*length(WristAcc_L))]-WristAcc_L_A[(length(WristAcc_L)+1):(2*length(WristAcc_L))],mu = 0)
h3<-t.test(WristAcc_S_A[(2*length(WristAcc_L)+1):(3*length(WristAcc_L))]-WristAcc_L_A[(2*length(WristAcc_L)+1):(3*length(WristAcc_L))],mu = 0)

e+annotate("text",x=2, y=0.65,label=h1$p.value)+annotate("text",x=4, y=0.65,label=h2$p.value)+annotate("text",x=6, y=0.8,label=h3$p.value)+scale_fill_manual(name="Conditions",values = c("lightblue","deepskyblue"),labels=c("Standard","Long"))+
  theme_bw()+ggtitle("Wrist Accuracy")
#
 

#------------------------------------------------------------------------------------------------------------


