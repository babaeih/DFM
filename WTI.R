install.packages("dfms")

install.packages('dfms', repos = c('https://sebkrantz.r-universe.dev', 'https://cloud.r-project.org'))

library(dfms)
library(readxl)

data <- read_excel("D:/Economics/Commodity/WTI/data.xls",sheet = 'Feuil1')
nr<-nrow(data)
nc<-ncol(data)

# Factor model of production and world GDP
q.prod <- data[,c("GDP",  "Production")]
q.prod=q.prod[157:nrow(data),1:ncol(q.prod)]
q.prod <- as.matrix(q.prod)
#q <- as.numeric(q)
mod.prod = DFM(q.prod, r = 1, p = 12) 
summary(mod.prod)
plot(mod.prod)
pca_prod<-mod.prod$F_pca
colnames(pca_prod) <- c('PC_prod') 

# Factor model of WTI and predicting variables
q <- data[,c("WTI","GSCPI","r",  "Kilian","Refinery Net Inputs","OPEC_Surplus_Capacity")]
q=q[157:nrow(data),1:ncol(q)]
q[287:297,1]
q <- as.matrix(q)
mod = DFM(q, r = 6, p = 1) 
pca<-mod$F_pca


#combining the PCs of the production-GDP with the whole model
pca= cbind(pca[nrow(pca)-nrow(pca_prod)+1:nrow(pca),1:ncol(pca)],pca_prod)
wti<-q[1:nrow(q),1]
#Regressing WTI on the combined PCs
ols <- lm(wti ~ pca) 
summary(ols)

ols$coefficients["(Intercept)"]
coef=data.matrix(ols$coefficients)
coef[2:nrow(coef),1:1]
wti_predict<-fitted(ols)

# Forecasting 20 periods ahead
fc.prod = predict(mod.prod, h = 12)
print(fc.prod)
plot(fc.prod)
as.data.frame(fc.prod)

fc = predict(mod, h = 12)
plot(fc)

#combining forecasted PCs of the two factor models
pca_fcst=cbind(fc$F_fcst,fc.prod$F_fcst)
wti_fcst=pca_fcst%*%matrix(coef[2:nrow(coef),1:1],nrow=nrow(coef)-1,ncol=1)+coef[1,1]
wti_fcst<-exp(wti_fcst)
wti_fcst

# 'dfm' methods
summary(mod)
plot(mod)
as.data.frame(mod)



