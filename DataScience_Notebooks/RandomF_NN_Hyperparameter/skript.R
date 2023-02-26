rm(list=ls())

#---------------------------------Data

#-------------Auswertungsanforderungen
load(file='/home/mascha/Dokumente/Einsteiger-Kurs/Einsteiger-Kurs_5/txt.RData')
#head(txt)
colnames(txt)[1]<-'Vorgangstyp'
#Load Letters Dataset


#-------------Buchstaben
letters.data <- read.csv('/home/mascha/Dokumente/Einsteiger-Kurs/Einsteiger-Kurs_6/lettersdata.csv', header = T)[,-1]

#Beschreibung https://archive.ics.uci.edu/ml/datasets/letter+recognition

data_sample<-letters.data[as.character(letters.data$lettr)%in%c('D','K','B'),]
data_sample$lettr<-as.factor(as.character(data_sample$lettr))


#--------------------------Vorbereitung

#teile in train und test 
set.seed(123)
test<-sample.int(nrow(txt),0.2*nrow(txt))
testb<-sample.int(nrow(data_sample),0.2*nrow(data_sample))


#--------------------------Random Forest
library(randomForest)

#------------- Regression
vars<-c('werbe','kunde','daten','adresse','pers','karte','cash','aktiv','konto','Zieldauer','Art')#,'kontonummer','kartennummer')

txt[,vars]<-apply(txt[,vars],2,scale)
txt<-data.frame(txt)

formula1<-as.formula(paste0('Dauer~',paste0(vars,collapse='+')))

set.seed(123)
rf_mod<-randomForest(formula1,data=txt[-test,],ntree=500)
rf_preds<-predict(rf_mod,newdata=txt[test,])

mse_rf<-mean((txt$Dauer[test]-rf_preds)^2)
mse_rf
#------------- Klassifikation
set.seed(123)
rf_mod_cl<-randomForest(lettr~.,data_sample[-testb,],method='class')
rf_preds_cl<-predict(rf_mod_cl, newdata=data_sample[testb,])

#Qualitaet der Vorhersage
tb_rf<-table(data_sample$lettr[testb],rf_preds_cl)
dt_rf<-diag(prop.table(tb_rf, 1))
dt_rf

#---------------------------- SVM
library(e1071) 

#--------------- Regression

#10-fold cross validation
df<-apply(data.frame(epsilon=rep(seq(0.1,1,by=0.1),each=5),cost=rep(seq(1,5,by=1)),times=length(seq(0.1,1,by=0.1))),1,as.list)
set.seed(123)
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = nrow(txt[-test,])))
cv_tmp <- matrix(NA, nrow = n_folds, ncol = length(df))
for (k in 1:n_folds) {
  test_i <- which(folds_i == k)
  train_cv <- txt[-test,][-test_i, ]
  test_cv <- txt[-test,][test_i, ]
  for(i in 1:length(df)){
    svm_tmp <- svm(formula1, data=train_cv,kernel='radial',epsilon=df[[i]]$epsilon,cost=df[[i]]$cost)
    preds_tmp<-predict(svm_tmp,newdata=test_cv[,vars])
    mse_tmp<-mean((test_cv$Dauer-preds_tmp)^2)
  cv_tmp[k,i]<-mse_tmp
  }
}
cv <- colMeans(cv_tmp)


svm_mod<-svm(formula1, data=txt[-test,],kernel='radial',epsilon=df[[which.min(cv)]]$epsilon,cost=df[[which.min(cv)]]$cost)
svm_preds<-predict(svm_mod,txt[test,vars])

mse_svm<-mean((txt$Dauer[test]-svm_preds)^2)
mse_svm
#--------------- Classification

svm_mod_cl<-svm(lettr~., data=data_sample[-testb,],kernel='radial',cost=4)
svm_preds_cl<-predict(svm_mod_cl,data_sample[testb,])
dt_svm<-diag(prop.table(table(data_sample$lettr[testb],svm_preds_cl), 1))
dt_svm

#--------------------------------------NN
library(nnet)

#----------------- Regression

sizes=seq(1:50)
mses<-numeric(max(sizes))
set.seed(123)
valid<-sample.int(nrow(txt[-test,]),0.1*nrow(txt[-test,]))

for(i in 1:max(sizes)){
  nn_mod<-nnet(formula1,data=txt[-test,][-valid,],size=i,linout=T)
  nn_preds<-predict(nn_mod,txt[-test,vars][valid,])
  
  mses[i]<-mean((txt$Dauer[-test][valid]-nn_preds)^2)
}
which.min(mses)

set.seed(123)
nn_mod<-nnet(formula1,data=txt[-test,],size=sizes[which.min(mses)],linout=T)
nn_preds<-predict(nn_mod,txt[test,vars])

mse_nn<-mean((txt$Dauer[test]-nn_preds)^2)
mse_nn

#---------------- Classification
ylettr<-model.matrix(~lettr-1,data=data_sample)

sizes=seq(5,40)
mses<-numeric(length(sizes))
set.seed(123)
valid<-sample.int(nrow(data_sample[-testb,]),0.1*nrow(data_sample[-testb,]))

for(i in sizes){
  nn_mod_cl<-nnet(lettr~.,data=data_sample[-testb,][-valid,],size=i,linout=T)
  nn_preds_cl<-predict(nn_mod_cl,data_sample[-testb,][valid,])
  
  nn_preds_cl_<-apply(nn_preds_cl,1,which.max)
  nn_preds_cl_<-colnames(ylettr)[nn_preds_cl_]
  mses[i]<-mean(diag(prop.table(table(data_sample$lettr[-testb][valid],nn_preds_cl_), 1)))
}
which.max(mses)

set.seed(123)
nn_mod_cl<-nnet(y=ylettr[-testb,],x=data_sample[-testb,-1],size=which.max(mses))
nn_preds_cl<-predict(nn_mod_cl,data_sample[testb,-1])

nn_preds_cl_<-apply(nn_preds_cl,1,which.max)
nn_preds_cl_<-colnames(ylettr)[nn_preds_cl_]

dt_nn<-diag(prop.table(table(data_sample$lettr[testb],nn_preds_cl_), 1))
dt_nn


#---------------------- Summary regression
acc<-rbind(mse_rf,mse_svm,mse_nn)
colnames(acc)<-'MSE'
rownames(acc)<-c('Random Forest','SVM','Neural Networks')
#acc<-transform(acc,Gesamt=apply(acc,1,mean))
acc<-apply(acc,2,round,2)
print(acc)


#---------------------- Summary classification
acc<-rbind(dt_rf,dt_svm,dt_nn)
rownames(acc)<-c('Random Forest','SVM','Neural Networks')
acc<-transform(acc,Gesamt=apply(acc,1,mean))
acc<-apply(acc,2,round,2)
print(acc)

