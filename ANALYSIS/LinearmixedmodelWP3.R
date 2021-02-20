library(car)
library(lsmeans)
library(Rmisc)
library(ggplot2)
library(lme4)

setwd ("G:/My Drive/Boost embodiment/T0vsVARESE/DRAFT_RESULTS_last_analysis/BL/analisi3_shapeindex")
mi<- read.csv('E:\\Switchdrive\\PhD\\Research\\PHRT\\WP3\\Data\\FORRDATA.csv',sep=";",dec=".",header=T,row.names=NULL)

#put as factor

mi$random <- factor(mi$ID)

#model
#Q1: what are the factor that affect the Bias between: body part(groups), Affected/unaffectedside, Population and deficit. using as covariate Age and maybe handedness?
#We know that the bodyparts always give a difference so we can use them for teh first factor
m1 <- lmer(MaxDistanceNormalized~1+(S_HL-B_HL)+(1|random),REML=FALSE,data=mi)
m1.2 <- lmer(MaxDistanceNormalized~1*(S_AL-B_AL)+(1|random),REML=FALSE,data=mi)
anova(m1,m1.2)
m2 <- lmer(MaxDistanceNormalized~1+(S_HL-B_HL)*(S_AL-B_AL)+(1|random),REML=FALSE,data=mi)
m3 <- lmer(MaxDistanceNormalized~1+(S_HL-B_HL)*Agency+(1|random),REML=FALSE,data=mi)
m3.2 <- lmer(MaxDistanceNormalized~1+(S_AL-B_AL)*Agency+(1|random),REML=FALSE,data=mi)
anova(m3,m3.2)
m4 <- lmer(MaxDistanceNormalized~1+(S_HL-B_HL)*Agency*Ownership+(1|random),REML=FALSE,data=mi)
m4.2 <- lmer(MaxDistanceNormalized~1+(S_AL-B_AL)*Agency*Ownership+(1|random),REML=FALSE,data=mi)
anova(m4,m4.2)
anova(m1,m2,m3)
anova(m1,m2,m3,m4,m1.2,m3.2,m4.2)
summary(m4)
library("psycho")
library(sjPlot) # table functions
library(sjmisc) # sample data
sjt.lmer(m4)
library(lme4) # fitting models
summary(results <- analyze(m4, CI = 95))
anova(m4)
lsmeans(m4, adjust="Tukey",  pairwise ~ Agency|S_HL-B_HL)

m1.b <- lmer(S_FL ~(1|random),REML=FALSE,data=mi)
m2.b <- lmer(S_FL ~MaxDistanceNormalized*Conditions +(1|random),REML=FALSE,data=mi)
m3.b <- lmer(S_FL ~MaxDistanceNormalized*Conditions*ElbowDist +(1|random),REML=FALSE,data=mi)
m4.b <- lmer(S_FL ~MaxDistanceNormalized*Conditions*Agency +(1|random),REML=FALSE,data=mi)
m5.b <- lmer(S_FL ~MaxDistanceNormalized*Conditions*Agency +(1|random),REML=FALSE,data=mi)
m6.b <- lmer(S_FL ~MaxDistanceNormalized*Conditions*Agency*ElbowDist+(1|random),REML=FALSE,data=mi)
m7.b <- lmer(S_FL ~MaxDistanceNormalized*Conditions*MaxPeakv_r+(1|random),REML=FALSE,data=mi)
m8.b <- lmer(S_FL ~MaxDistanceNormalized*Conditions*MaxPeakv_r*ElbowDist+(1|random),REML=FALSE,data=mi)
m9.b <- lmer(S_FL ~MaxDistanceNormalized*Conditions*MaxPeakv_r*ElbowDist*Agency+(1|random),REML=FALSE,data=mi)
m10.b <- lmer(S_FL ~MaxDistanceNormalized+Conditions+MaxPeakv_r+ElbowDist+Agency+(1|random),REML=FALSE,data=mi)

anova(m1.b,m2.b,m3.b,m4.b,m6.b,m7.b,m8.b,m9.b,m10.b)
anova(m3.b, m9.b)
summary(m9.b)
emmeans(m9.b, pairwise ~ Conditions)

m1.b <- lmer(S_AL ~(1|random),REML=FALSE,data=mi)
m2.b <- lmer(S_AL ~MaxDistanceNormalized +(1|random),REML=FALSE,data=mi)
m3.b <- lmer(S_AL ~MaxDistanceNormalized*ElbowDist +(1|random),REML=FALSE,data=mi)
m4.b <- lmer(S_AL ~MaxDistanceNormalized*Agency +(1|random),REML=FALSE,data=mi)
m5.b <- lmer(S_AL ~MaxDistanceNormalized*Agency +(1|random),REML=FALSE,data=mi)
m6.b <- lmer(S_AL ~MaxDistanceNormalized*Agency*ElbowDist+(1|random),REML=FALSE,data=mi)
anova(m1.b,m2.b,m3.b,m4.b,m6.b)



#######################################################################################
#                       WITH THE DATAFRAME PER TRIAL
#######################################################################################

 library(readxl)
library(ggplot2)
library(lme4)

Data <- read_xlsx(("BodyKinematics_dataframe_190221.xlsx"),sheet = 2, range = "A2:T2017", col_names = FALSE,col_types = NULL, na = "", trim_ws = TRUE, skip = 0,progress = readxl_progress(), .name_repair = "unique");
colnames(Data) <- c("ID","DistanceWrist","PeakvelocityWrist","MovementDurationWrist","LatencyWrist","DistanceElbow","PeakvelocityElbow","MovementDurationElbow","LatencyElbow","Conditions","Order","Trials","Ownership","Agency","Size","Control","ObjectSize","ObjectDistance","DistanceTarget","TrialNumber")

#remove outliers:
#Data      <- Data[Data$ID != 'Mock_Y20',] # super low score of performance less than 50% in 2 conditions
# Data      <- Data[Data$ID != 'Mock_Y30',] # super low score of performance less than 50% in 2 conditions
# Data      <- Data[Data$ID != 'Mock_Y18',] # Technical bug of tracking during BSC
# Data      <- Data[Data$ID != 'Mock_Y14',] #technical bug: mismatch between skin color (dark) and avatar skin color
# Data      <- Data[Data$ID != 'Mock_Y06',] #Answered always that the environment changed  
#Let's convert these parameters into factor
Data$Conditions   <- factor(Data$Conditions)
Data$Trials   <- factor(Data$Trials)


Data_S      <- Data[Data$Conditions == "Standard",]
Data_L      <- Data[Data$Conditions == "Large",]

df = aggregate(cbind(DistanceWrist,PeakvelocityWrist,MovementDurationWrist,LatencyWrist) ~  ID+Conditions+Trials,Data,mean)

#again let's just convert into factor and organize the level in the correct order
df$Conditions <-factor(df$Conditions, 
                         levels = c("Standard","Large"
                                   ))
# jpeg(file="CR_SYNCH_MOCKN24_Last_trials.jpeg",quality = 100, res = 200,width = 1000, height = 800)
ggplot(df,aes(x=Conditions, y = DistanceWrist, fill =Conditions))+    geom_boxplot(position=position_dodge(0.8))+
  geom_dotplot(binaxis='y', stackdir='center', position=position_dodge(0.8),dotsize = 0.5) + ggtitle('Perf')+scale_fill_manual(name="CR",values =c("red2","grey60"))
# dev.off()


summary(aov(DistanceWrist~Conditions +Error(ID),data = df))
summary(aov(PeakvelocityWrist~Conditions +Error(ID),data = df))
summary(aov(LatencyWrist~Conditions +Error(ID),data = df))
summary(aov(MovementDurationWrist~Conditions +Error(ID),data = df))



summary(lmer(DistanceWrist~Conditions*TrialNumber + (1|ID),data = Data))
summary(lmer(MovementDurationWrist~Conditions*TrialNumber + (1|ID),data = Data))
summary(lmer(PeakvelocityWrist~Conditions*TrialNumber + (1|ID),data = Data))
summary(lmer(LatencyWrist~Conditions*TrialNumber + (1|ID),data = Data))


plot(Data,aes(x=TrialNumber, y = DistanceWrist, fill =Conditions))+    
  geom_dotplot(binaxis='y', stackdir='center', position=position_dodge(0.8),dotsize = 0.5) +ggtitle('Perf')+scale_fill_manual(name="CR",values =c("red2","grey60"))



par(mfrow=c(6,4))
par(mar=c(1,1,1,1))
for (i in c(1:24))
{
  plot(Data_S$TrialNumber[Data$ID == unique(df$ID)[i]], Data_S$DistanceWrist[Data$ID == unique(df$ID)[i]], type="b", col="green", lwd=5, pch=15)
  lines(Data_S$TrialNumber[Data$ID == unique(df$ID)[i]], Data_S$DistanceWrist[Data$ID == unique(df$ID)[i]], type="b", col="red", lwd=2, pch=19)
}
