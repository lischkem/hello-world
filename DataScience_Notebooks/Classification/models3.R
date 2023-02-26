#Load Letters Dataset

letters.data <- read.csv('/home/mascha/Dokumente/Einsteiger-Kurs/Einsteiger-Kurs_6/lettersdata.csv', header = T)[,-1]

#Beschreibung https://archive.ics.uci.edu/ml/datasets/letter+recognition

data_sample<-letters.data[as.character(letters.data$lettr)%in%c('D','K','B'),]
data_sample$lettr<-as.factor(as.character(data_sample$lettr))


library(nnet) #multinomial regression
library(MASS) # lda
library(rpart) # entscheidungbaum
library(randomForest) # random forest
library(RSNNS) # deep learning
library(maboost) # boosting

#teile in test und training
n=nrow(data_sample)
set.seed(123)
test<-sample.int(n,0.3*n)

#--------------------------------------------------logistische Regression
lr_mod<-multinom(lettr~., data_sample[-test,])
lr_preds<-as.character(predict(lr_mod,newdata=data_sample[test,]))
summary(lr_mod)

#Qualitaet der Vorhersage
tb_lr<-table(data_sample$lettr[test],lr_preds)
tb_lr
lr_acc<-diag(prop.table(tb_lr, 1))
lr_acc

#visualize
with(data_sample[test,],plot(xy2br,x2bar,pch=as.character(lettr),col=as.factor(lr_preds)))
pairs(~yegvx+xy2br+xybar+x2bar,data=data_sample[test,],
      pch=as.character(data_sample$lettr[test]),col=as.factor(lr_preds))
#--------------------------------------------------LDA
ld_mod<- lda(lettr~., data=data_sample[-test,])
ld_preds<-as.character(predict(ld_mod,newdata=data_sample[test,])$class)
ld_mod

#Qualitaet der Vorhersage
tb_ld<-table(data_sample$lettr[test],ld_preds)
ld_acc<-diag(prop.table(tb_ld, 1))
ld_acc

#visualize
lds<-predict(ld_mod,newdata=data_sample[test,])$x
with(data_sample[test,],plot(lds[,1],lds[,2],pch=as.character(lettr),col=as.factor(ld_preds)))

#--------------------------------------------------Entscheidungsbaum
dt_mod<-rpart(lettr~.,data_sample[-test,],method='class')
dt_preds<-as.character(predict(dt_mod,newdata=data_sample[test,],type='class'))
dt_mod

#Qualitaet der Vorhersage
tb_dt<-table(data_sample$lettr[test],dt_preds)
dt_acc<-diag(prop.table(tb_dt, 1))
dt_acc

#Visualize
plot(dt_mod)
text(dt_mod)

#--------------------------------------------------Bagging
N=4000
M=100
dt_mods<-list(seq(1:M))
dt_predss<-matrix(NA,length(test),M)

for (j in 1:M){
  #Bootstrap
  set.seed(123+j)
  ind<-sample.int(n=nrow(data_sample),size=N,replace=T) 
  data_boot<-data_sample[ind,]
  
  # Entscheidungsbaum
  dt_mods[[j]]<-rpart(lettr~.,data_boot,method='class')
  dt_predss[,j]<-as.character(predict(dt_mods[[j]],newdata=data_sample[test,],type='class'))
}

maj_vote<-function(x){
  tab<-table(x)
  cl<-names(which.max(tab))
  return(cl)
}

#Qualitaet der Vorhersage
dt_pred<-apply(dt_predss,1,maj_vote)
tb_dts<-table(data_sample$lettr[test],dt_pred)
dts_acc<-diag(prop.table(tb_dts, 1))
dts_acc

#Visualize
par(mfrow=c(2,2),xpd=T)
set.seed(123)
samp<-sample.int(M,4,replace=F)
for(i in 1:length(samp)){
  plot(dt_mods[[i]])
  text(dt_mods[[i]])
}

par(mfrow=c(1,1))
#----------------------------------------------- Boosting
#boost
dtboost_mod<-maboost(formula=lettr~.,data=data_sample[-test,])
dtboost_preds<-predict(dtboost_mod,newdata=data_sample[test,])

#Qualitaet der Vorhersage
tb_dtboost<-table(data_sample$lettr[test],dtboost_preds)
boost_acc<-diag(prop.table(tb_dtboost, 1))
boost_acc

#Visualize
with(data_sample[test,],plot(xy2br,x2bar,pch=as.character(lettr),col=as.factor(dtboost_preds)))
pairs(~yegvx+xy2br+xybar+x2bar,data=data_sample[test,],
      pch=as.character(data_sample$lettr[test]),col=as.factor(dtboost_preds))
#------

#----------------------------------------------- Stacking

lr_fit<-predict(lr_mod,newdata=data_sample[-test,])
ld_fit<-predict(ld_mod,newdata=data_sample[-test,])$class
dt_fit<-predict(dt_mod,newdata=data_sample[-test,],type='class')

#fit
X<-cbind(model.matrix( ~ lr_ - 1, data=data.frame(lr_=lr_fit) ),#Vorhersage von der logistischen Regression
         model.matrix( ~ ld_ - 1, data=data.frame(ld_=ld_fit) ),#lda
         model.matrix( ~ dt_ - 1, data=data.frame(dt_=dt_fit) ))#Entscheidungsbaum

stack_data<-data.frame(lettr=data_sample$lettr[-test],X=X)
stack_mod<-rpart(lettr~.,stack_data,method='class')

#test
X_test<-cbind(model.matrix( ~ lr_ - 1, data=data.frame(lr_=lr_preds) ),#Vorhersage von der logistischen Regression
              model.matrix( ~ ld_ - 1, data=data.frame(ld_=ld_preds) ),#lda
              model.matrix( ~ dt_ - 1, data=data.frame(dt_=dt_preds) ))#Entscheidungsbaum
stack_data_test<-data.frame(lettr=data_sample$lettr[test],X=X_test)

stack_preds<-predict(stack_mod,newdata=stack_data_test,type='class')

stack_tb<-table(data_sample$lettr[test],stack_preds)
stack_acc<-diag(prop.table(stack_tb, 1))
stack_acc


acc<-rbind(lr_acc,ld_acc,dt_acc,dts_acc,boost_acc,stack_acc)
rownames(acc)<-c('Logistische Regression','LDA','Entscheidungsbaum','Bagging','Boosting','Stacking')
acc<-transform(acc,Gesamt=apply(acc,1,mean))
acc<-apply(acc,2,round,2)
print(acc)

