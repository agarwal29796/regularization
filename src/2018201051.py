
# coding: utf-8

# # Assignment 7  : Regularization
# submitted by : <br>
# Archit Kumar<br>
# 2018201051
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[160]:


data  = pd.read_csv('data/data.csv')
data = data.iloc[:,1:]
data.head()


# In[161]:


data = data.sample(frac=1).reset_index(drop=True)
X, Y = data.iloc[:, : -1] , data.iloc[: ,-1]


# In[162]:


plt.scatter(X.iloc[: ,0] , X.iloc[:, 1])
plt.show()


# In[163]:


class linear_regression():
    def __init__(self, epochs = 100 , learning_rate  = 0.01 , regularization = None ,reg_coeff = 0.1):
        self.reg  = regularization  # lasso or ridge
        self.reg_coeff = reg_coeff
        self.epochs = epochs
        self.alpha  = learning_rate
        self.training_error = []
        self.val_error = []
        
    def fitk(self , X ,Y,  k):
        num = int(X.shape[0]/k)   # k = X.shape[0] for Leave-one-out cross-validation [LOOCV] 
        train_error = []
        val_error = []
        for i in range(k) :
            self.val_error = []
            self.training_error = []
            x_train =  pd.concat([X.iloc[:i*num,:] , X.iloc[(i+1)*num:,:]])
            y_train =  pd.concat([Y.iloc[:i*num] , Y.iloc[(i+1)*num:]])
            x_val, y_val = X.iloc[i*num : (i+1)*num,:] , Y.iloc[i*num : (i+1)*num]
            self.compute(x_train, y_train , x_val , y_val)
            val_error.append(self.val_error[-1])
            train_error.append(self.training_error[-1])
        return val_error ,  train_error
            
    
    def fit(self , X ,Y):
        x_train ,  y_train = X.iloc[:int(X.shape[0]*0.7),  :] , Y.iloc[:int(Y.shape[0]*0.7)]
        x_val ,  y_val = X.iloc[int(X.shape[0]*0.7):,  :] , Y.iloc[int(Y.shape[0]*0.7):]
        self.compute(x_train, y_train , x_val , y_val)
        
    
    def compute(self, x_train, y_train , x_val , y_val):
        self.mean = x_train.mean()
        self.std =  x_train.std()
        
        x_train  = (x_train - self.mean)/self.std
        x_val  = (x_val - self.mean)/self.std 
        
        x_train.insert( 0, "bias" , 1)
        x_val.insert( 0, "bias" , 1)
        np.random.seed(1)
        self.theta = np.random.randn(x_train.shape[1])
        
        for i in range(self.epochs):
            pred_train = np.dot(x_train,self.theta)
            pred_val = np.dot(x_val , self.theta)
                       
            te = np.sum(np.square(y_train-pred_train))/float(x_train.shape[0]) 
            ve = np.sum(np.square(y_val-pred_val))/float(x_val.shape[0])
            
            self.training_error.append(te)
            self.val_error.append(ve)
            
#             grad =  np.dot(x_train.T , pred_train-y_train)/float(x_train.shape[0])
            grad =  np.dot(x_train.T , pred_train-y_train)

            if self.reg == "ridge" :
                grad = grad + float(self.reg_coeff)*self.theta              
            elif  self.reg == "lasso":
                grad = grad + float(self.reg_coeff)*np.where(self.theta >= 0 , 1 , -1)          
            self.theta = self.theta  - (float(self.alpha)/x_train.shape[0])*grad
    
    def predict(self , X):
        X = (X  - self.mean)/self.std
        X.insert( 0, "bias" , 1)
        return np.round(np.dot(X , self.theta),  3)
            
            
            
        
            

        


# In[164]:


lr  = linear_regression(epochs = 100 , learning_rate = 0.05)
lr.fit(X ,Y)


# In[165]:


plt.figure(figsize = (20,8))
x_axis = range(lr.epochs)
y_axis1 = lr.training_error
y_axis2 = lr.val_error
plt.plot(x_axis, y_axis1 , 'r')
plt.plot(x_axis ,y_axis2 , 'g')
plt.xlabel('epochs')
plt.ylabel('error')
plt.legend(['training_error' , 'val_error'])
plt.show()


# ## Question 1 and 3 : Lasso Regression

# In[166]:


plt.figure(figsize = (20,8))
reg_coeffs = np.arange(0,2000,20)
val_errors = []
for lam in reg_coeffs :
    model = linear_regression(regularization=  "lasso" , reg_coeff = lam, learning_rate = 0.01)
    model.fit(X,Y)
    val_errors.append(model.val_error[-1])
plt.plot(reg_coeffs , val_errors , 'r')
plt.xlabel('Regularization coefficient')
plt.ylabel('validation error')
plt.show()


# ##  Question 2  and 3: Ridge Regression

# Ridge regression can reduce the variance (with an increasing bias) works best in situations where the OLS estimates have high variance<br>
#  Can improve predictive performance<br>
#  Works in situations where p < n<br>
#  Mathematically simple computations<br>
#  Ridge regression is not able to shrink
#  coefficients to exactly zero<br>
#  As a result, it cannot perform variable selection
# 

# In[147]:


plt.figure(figsize = (20,8))
reg_coeffs = np.arange(0 , 2000, 20)
val_errors = []
for lam in reg_coeffs :
    model = linear_regression(regularization=  "ridge" , reg_coeff = lam)
    model.fit(X,Y)
    val_errors.append(model.val_error[-1])
plt.plot(reg_coeffs , val_errors , 'g')
plt.xlabel('regularization coefficient')
plt.ylabel('error in validation set')
plt.show()


# ##  Question 4 : 

# ### 4.1 lasso  

# In[141]:


plt.figure(figsize = (20,8))
reg_coeffs = np.arange(0 , 2000, 20)
val_errors = []
colors= ['b', 'g' , 'r' , 'm' , 'c' , 'y' , 'k']
params = []
for lam in reg_coeffs:
    model = linear_regression(regularization=  "lasso" , reg_coeff = lam)
    model.fit(X,Y)
    val_errors.append(model.val_error[-1])
    params.append(list(model.theta))
print("ok")    
for i in range(len(params[0])):
    coeffs = []
    for j in range(len(params)):
        coeffs.append(params[j][i])
    plt.plot(reg_coeffs , coeffs,  colors[i%7])
plt.xlabel('regularization coefficient')
plt.ylabel('values of parameters')
plt.show()


# In[142]:


reg_coeffs = np.arange(0 , 500, 20)

for lam in reg_coeffs :
    theta = []
    model  =  linear_regression(regularization="lasso" , reg_coeff=  lam)
    model.fit(X,Y)
    theta = list(model.theta)
    theta.sort()
    plt.bar(range(len(theta)) , theta , 0.8)
    plt.show()


# ### 4.2 Ridge 

# In[143]:


plt.figure(figsize = (20,8))
reg_coeffs = np.arange(0 , 2000, 20)
val_errors = []
colors= ['b', 'g' , 'r' , 'm' , 'c' , 'y' , 'k']
params = []
for lam in reg_coeffs:
    model = linear_regression(regularization=  "ridge" , reg_coeff = lam)
    model.fit(X,Y)
    val_errors.append(model.val_error[-1])
    params.append(list(model.theta))
print("ok")    
for i in range(len(params[0])):
    coeffs = []
    for j in range(len(params)):
        coeffs.append(params[j][i])
    plt.plot(reg_coeffs , coeffs,  colors[i%7])
plt.xlabel('regularization coefficient')
plt.ylabel('values of parameters')
plt.show()


# In[144]:


reg_coeffs = np.arange(0 , 500, 20)

for lam in reg_coeffs :
    theta = []
    model  =  linear_regression(regularization="ridge" , reg_coeff=  lam)
    model.fit(X,Y)
    theta = list(model.theta)
    theta.sort()
    plt.bar(range(len(theta)) , theta , 0.8)
    plt.show()


# ## Question 5 : K fold cross validation and leave-one-out cross validation 

# In[157]:


K = [2,15,40, 70 ,100, 150 , 200, 230 , 270, 300 , 350 , 370, 400 , X.shape[0]]
valg = []
traing = []
for k in K :
    print(k)
    model = linear_regression()
    val_err , train_err = model.fitk(X,Y,k)
    valg.append(sum(val_err)/len(val_err))
    traing.append(sum(train_err)/len(train_err))
plt.plot(K , valg , 'g')
plt.plot(K, traing , 'r')
plt.xlabel('value of K')
plt.ylabel('error')
plt.show()


# #### LOOCV : internal loop of val and training errors 

# In[172]:


plt.plot( range(len(val_err)), val_err,  'b')
plt.plot(range(len(train_err)) , train_err , 'y')
plt.legend(['val error' , 'training error'])
plt.xlabel('ith loop')
plt.ylabel('error')
plt.show()

