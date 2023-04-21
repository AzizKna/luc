#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 18:59:48 2023

@author: lucvercoulen
"""
#importing the data and packages
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm
import scipy.stats
import statsmodels.api as sm
from statsmodels.imputation import mice
Webshop_CSV = pd.read_csv('2124800webshop.csv')

#removing the missing value's 
# Create a copy of the dataset
Webshop_CSV_copy = Webshop_CSV.copy()
# Check for missing values in the original dataset
Webshop_CSV.isnull().sum()
# Create a new copy of the original dataset
Clean_Webshop_CSV = Webshop_CSV_copy.copy()
# Remove missing values from the copy
Clean_Webshop_CSV.dropna(inplace=True)
# Check for missing values in the cleaned copy
Clean_Webshop_CSV.isnull().sum()


#creating Dummy variables 
Clean_Webshop_Dummies = pd.get_dummies(Clean_Webshop_CSV['Device'])
Clean_Webshop_CSV = pd.concat([Clean_Webshop_CSV, Clean_Webshop_Dummies], axis=1)
Clean_webshop_Dummies_find_website = pd.get_dummies(Clean_Webshop_CSV['Find_website'])
Clean_Webshop_CSV = pd.concat([Clean_Webshop_CSV, Clean_webshop_Dummies_find_website], axis=1)


#dealing with outliers
#first create a new dataset for the outliers
Clean_Webshop_Outlier = Clean_Webshop_CSV.copy()
#create a variable for the sample zise 
n = len(Clean_Webshop_Outlier)


#Creating single regression & outliers time spent on website
SR1_Time_Spent_on_Website = sm.ols('Purchase_Amount ~ Time_Spent_on_Website', data = Clean_Webshop_Outlier).fit()
print(SR1_Time_Spent_on_Website.summary())
CooksD = SR1_Time_Spent_on_Website.get_influence().cooks_distance
Clean_Webshop_Outlier['Outlier'] = CooksD[0] > 4/n
Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier1 = Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
mask = Clean_Webshop_Outlier['Time_Spent_on_Website']==-999
Clean_Webshop_Outlier = Clean_Webshop_Outlier[~mask]

SR1_Time_Spent_on_Website = sm.ols('Purchase_Amount ~ Time_Spent_on_Website', data = Clean_Webshop_Outlier).fit()
print(SR1_Time_Spent_on_Website.summary())
Clean_Webshop_Outlier = Clean_Webshop_Outlier.drop('Outlier', axis=1)

#Creating single regression & checking outliers number of products browsed
SR2_Number_of_products_browsed = sm.ols('Purchase_Amount ~ Number_of_products_browsed', data = Clean_Webshop_Outlier).fit()
print(SR2_Number_of_products_browsed.summary())

CooksD = SR2_Number_of_products_browsed.get_influence().cooks_distance
Clean_Webshop_Outlier['Outlier'] = CooksD[0] > 4/n
Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier2 = Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier = Clean_Webshop_Outlier.drop('Outlier', axis=1)

#Creating single regression & checking outliers number pictures
SR3_Pictures = sm.ols('Purchase_Amount ~ Pictures', data = Clean_Webshop_Outlier).fit()
print(SR3_Pictures.summary())

CooksD = SR3_Pictures.get_influence().cooks_distance
Clean_Webshop_Outlier['Outlier'] = CooksD[0] > 4/n
Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier3 =Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier = Clean_Webshop_Outlier.drop('Outlier', axis=1)

#Creating single regression & Shipping time
SR4_Shipping_Time = sm.ols('Purchase_Amount ~ Shipping_Time', data = Clean_Webshop_Outlier).fit()
print(SR4_Shipping_Time.summary())

CooksD = SR4_Shipping_Time.get_influence().cooks_distance
Clean_Webshop_Outlier['Outlier'] = CooksD[0] > 4/n
Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier4 =Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier = Clean_Webshop_Outlier.drop('Outlier', axis=1)

#Creating single regression & checking outliers review rating
SR5_Review_rating = sm.ols('Purchase_Amount ~ Review_rating', data = Clean_Webshop_Outlier).fit()
print(SR5_Review_rating.summary())

CooksD = SR5_Review_rating.get_influence().cooks_distance
Clean_Webshop_Outlier['Outlier'] = CooksD[0] > 4/n
Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier5 =Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier = Clean_Webshop_Outlier.drop('Outlier', axis=1)

#Creating single regression & checking outliers Ease of purchase
SR6_Ease_of_purchase = sm.ols('Purchase_Amount ~ Ease_of_purchase', data = Clean_Webshop_Outlier).fit()
print(SR6_Ease_of_purchase.summary())

CooksD = SR6_Ease_of_purchase.get_influence().cooks_distance
Clean_Webshop_Outlier['Outlier'] = CooksD[0] > 4/n
Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier6 =Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier = Clean_Webshop_Outlier.drop('Outlier', axis=1)

#Creating single regression & checking outliers Device
SR7_Device = sm.ols('Purchase_Amount ~ PC', data = Clean_Webshop_Outlier).fit()
print(SR7_Device.summary())

CooksD = SR7_Device.get_influence().cooks_distance
Clean_Webshop_Outlier['Outlier'] = CooksD[0] > 4/n
Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier7 =Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier = Clean_Webshop_Outlier.drop('Outlier', axis=1)

SR7_1_Device = sm.ols('Purchase_Amount ~ Mobile', data = Clean_Webshop_Outlier).fit()
print(SR7_1_Device.summary())

CooksD = SR7_1_Device.get_influence().cooks_distance
Clean_Webshop_Outlier['Outlier'] = CooksD[0] > 4/n
Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier7_1 =Clean_Webshop_Outlier[Clean_Webshop_Outlier.Outlier == True]
Clean_Webshop_Outlier = Clean_Webshop_Outlier.drop('Outlier', axis=1)

#Creating multiple regression found website
MR8_Find_website = sm.ols('Purchase_Amount ~ Friends_or_Family + Search_Engine + Social_Media_Advertisement', data = Clean_Webshop_Outlier).fit()
print(MR8_Find_website.summary())

MR8_1_Find_website_other_reference_group = sm.ols('Purchase_Amount ~ Other + Search_Engine + Social_Media_Advertisement', data = Clean_Webshop_Outlier).fit()
print(MR8_1_Find_website_other_reference_group.summary())

#Creating single regression Age 
SR9_age = sm.ols('Purchase_Amount ~ Age', data = Clean_Webshop_Outlier).fit()
print(SR9_age.summary())


#Checking for multicollinearity
corr_matrix = Clean_Webshop_Outlier.corr()
print(corr_matrix)

MR10_multicollinearity_check = sm.ols('Purchase_Amount ~ Number_of_products_browsed + Time_Spent_on_Website', data = Clean_Webshop_Outlier).fit()
print(MR10_multicollinearity_check.summary())

MR11_all_variables_Friends_family_Mobile_reference = sm.ols('Purchase_Amount ~ Time_Spent_on_Website + Age + Number_of_products_browsed + Pictures + Shipping_Time + Review_rating + Ease_of_purchase + Age + PC + Other + Search_Engine + Social_Media_Advertisement', data = Clean_Webshop_Outlier).fit()
print(MR11_all_variables_Friends_family_Mobile_reference.summary())


#Checking for non-liniar relationship
#Checking relationship time spent on website
sns.regplot(x = 'Time_Spent_on_Website', y = 'Purchase_Amount', data = Clean_Webshop_Outlier, lowess = True)

#Checking relationship number of products browsed
sns.regplot(x = 'Number_of_products_browsed', y = 'Purchase_Amount', data = Clean_Webshop_Outlier, lowess = True)

#Checking relationship pictures
sns.regplot(x = 'Pictures', y = 'Purchase_Amount', data = Clean_Webshop_Outlier, lowess = True)

#Checking relationship shipping time
sns.regplot(x = 'Shipping_Time', y = 'Purchase_Amount', data = Clean_Webshop_Outlier, lowess = True)

#Checking relationship review rating
sns.regplot(x = 'Review_rating', y = 'Purchase_Amount', data = Clean_Webshop_Outlier, lowess = True)

#Checking relationship ease of products
sns.regplot(x = 'Ease_of_purchase', y = 'Purchase_Amount', data = Clean_Webshop_Outlier, lowess = True)

#Checking relationship age
sns.regplot(x = 'Age', y = 'Purchase_Amount', data = Clean_Webshop_Outlier, lowess = True)


#polynomial transformation 
Age2 = pow(Clean_Webshop_Outlier.Age,2)
#comparing the linear model to polynomial model  
SR9_age = sm.ols('Purchase_Amount ~ Age', data = Clean_Webshop_Outlier).fit()
print(SR9_age.summary())
polynomial_model = sm.ols('Purchase_Amount ~ Age + Age2', data = Clean_Webshop_Outlier).fit()
print(polynomial_model.summary())


#Create the eventual model for the APA table
MR12_model1 = sm.ols('Purchase_Amount ~  Age + Age2 + Time_Spent_on_Website + Shipping_Time + Review_rating + Ease_of_purchase + PC + Ease_of_purchase + Friends_or_Family + Social_Media_Advertisement', data = Clean_Webshop_Outlier).fit()
print(MR12_model1.summary())

#The following code for creating APA style tables only works in Jupyterlab
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML
MR12_model1 = sm.ols('Purchase_Amount ~  Age + Age2 + Time_Spent_on_Website + Shipping_Time + Review_rating + Ease_of_purchase + PC + Ease_of_purchase + Friends_or_Family + Social_Media_Advertisement', data = Clean_Webshop_Outlier).fit()
print(MR12_model1.summary())

Table = Stargazer([MR12_model1])
Table.title =('Model predict purchase amount')
Table.show_model_numbers(False) 
Table.significant_digits(2)
Table.covariate_order(['Intercept', 'Age', 'Age2', 'Ease_of_purchase', 'Review_rating', 'Shipping_Time', 'Time_Spent_on_Website', 'PC', 'Friends_or_Family', 'Social_Media_Advertisement'])
Table.rename_covariates({'Ease_of_purchase':'Ease of purchase', 'Review_rating' : 'Review rating', 'Shipping_Time' : 'Shipping time', 'Time_Spent_on_Website' : 'Time spent on website', 'Friends_or_Family' : 'Friend or family', 'Search_Engine' : 'Search engine', 'Social_Media_Advertisement' : 'Social media advertisement'}) 
HTML(Table.render_html())


#standarization of variables
Clean_Webshop_Standardized = Clean_Webshop_Outlier.copy()
Clean_Webshop_Standardized[['Time_Spent_on_Website', 'Shipping_Time', 'Review_rating', 'Ease_of_purchase', 'Age']] = StandardScaler().fit_transform(Clean_Webshop_Standardized[['Time_Spent_on_Website', 'Shipping_Time', 'Review_rating', 'Ease_of_purchase', 'Age']])

#creating a model with the standardized variables
MR13_model2 = sm.ols('Purchase_Amount ~  Age + Age2 + Time_Spent_on_Website + Shipping_Time + Review_rating + Ease_of_purchase + PC + Ease_of_purchase + Friends_or_Family + Social_Media_Advertisement', data = Clean_Webshop_Standardized).fit()
print(MR13_model2.summary())




#creating a APA table with two models
Table = Stargazer([MR12_model1,MR13_model2])
Table.title =('Model predict purchase amount')
Table.show_model_numbers(False) 
Table.significant_digits(2)
Table.covariate_order(['Intercept', 'Age', 'Age2', 'Ease_of_purchase', 'Review_rating', 'Shipping_Time', 'Time_Spent_on_Website', 'PC', 'Friends_or_Family', 'Social_Media_Advertisement'])
Table.rename_covariates({'Ease_of_purchase':'Ease of purchase', 'Review_rating' : 'Review rating', 'Shipping_Time' : 'Shipping time', 'Time_Spent_on_Website' : 'Time spent on website', 'Friends_or_Family' : 'Friend or family', 'Search_Engine' : 'Search engine', 'Social_Media_Advertisement' : 'Social media advertisement'}) 

HTML(Table.render_html())



#predict a customer that's not in the dataset 
#creating the variable Age2 as a column to the data frame
Clean_Webshop_Outlier['Age2'] = Clean_Webshop_Outlier['Age']**2



#calculate the Age 2 for the new customer
new_customer_age = 35
new_customer_age2 = new_customer_age ** 2

New_customer = pd.DataFrame([[723,20, 3.4, 2.6, 4.5, 'Search_Engine' , 4, 35, 'PC', 0, 1, 1 , 0, 0,0, 1225]], columns = ['Time_Spent_on_Website', 'Number_of_products_browsed', 'Pictures', 'Shipping_Time', 'Review_rating', 'Find_website', 'Ease_of_purchase', 'Age', 'Device', 'Mobile', 'PC', 'Friends_or_Family', 'Other', 'Search_Engine', 'Social_Media_Advertisement', 'Age2'])
New_customer['Age2'] = new_customer_age2

print(MR12_model1.predict(New_customer))





#Dealing with missing values
Webshop_CSV_With_NA = Webshop_CSV.copy()
Webshop_CSV_With_NA = pd.get_dummies(Webshop_CSV_With_NA, dummy_na=True)

#removing outliers
mask1 = Webshop_CSV_With_NA['Time_Spent_on_Website']==-999
Webshop_CSV_With_NA = Webshop_CSV_With_NA[~mask1]

#Checking for multicollinearity
corr_matrix1 = Webshop_CSV_With_NA.corr()
print(corr_matrix1)

#Checking relationship time spent on website
sns.regplot(x = 'Time_Spent_on_Website', y = 'Purchase_Amount', data = Webshop_CSV_With_NA, lowess = True)

#Checking relationship number of products browsed
sns.regplot(x = 'Number_of_products_browsed', y = 'Purchase_Amount', data = Webshop_CSV_With_NA, lowess = True)

#Checking relationship pictures
sns.regplot(x = 'Pictures', y = 'Purchase_Amount', data = Webshop_CSV_With_NA, lowess = True)

#Checking relationship shipping time
sns.regplot(x = 'Shipping_Time', y = 'Purchase_Amount', data = Webshop_CSV_With_NA, lowess = True)

#Checking relationship review rating
sns.regplot(x = 'Review_rating', y = 'Purchase_Amount', data = Webshop_CSV_With_NA, lowess = True)

#Checking relationship ease of products
sns.regplot(x = 'Ease_of_purchase', y = 'Purchase_Amount', data = Webshop_CSV_With_NA, lowess = True)

#Checking relationship age
sns.regplot(x = 'Age', y = 'Purchase_Amount', data = Webshop_CSV_With_NA, lowess = True)

#polynomial transformation 
Age2 = pow(Webshop_CSV_With_NA.Age,2)
Webshop_CSV_With_NA['Age2'] = Webshop_CSV_With_NA['Age']**2

#creating a dataset with only the variables that are needed for the model
Webshop_CSV_With_NA_Onlymodelvariables = Webshop_CSV_With_NA[['Purchase_Amount', 'Age', 'Age2', 'Time_Spent_on_Website', 'Shipping_Time', 'Review_rating', 'Ease_of_purchase', 'Device_PC', 'Find_website_Friends_or_Family', 'Find_website_Social_Media_Advertisement']].copy()

# Turn the data into a MICE dataset
imp = mice.MICEData(Webshop_CSV_With_NA_Onlymodelvariables)
model = mice.MICE("Purchase_Amount ~ Age + Age2 + Time_Spent_on_Website + Shipping_Time + Review_rating + Ease_of_purchase + Device_PC + Find_website_Friends_or_Family + Find_website_Social_Media_Advertisement", model_class=sm.OLS, data=imp).fit()

print(model.summary())




