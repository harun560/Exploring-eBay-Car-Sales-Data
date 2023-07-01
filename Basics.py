#!/usr/bin/env python
# coding: utf-8

# This project focuses on cleansing and analyzing the data of used car listings. Additionally, it aims to introduce the advantages of using Jupyter Notebook for working with pandas.

# Analyzing Used Car Listings on eBay Kleinanzeigen
# We will be working on a dataset of used cars from eBay Kleinanzeigen, a classifieds section of the German eBay website.
# 
# The dataset was originally scraped and uploaded to Kaggle. The version of the dataset we are working with is a sample of 50,000 data points that was prepared by Dataquest including simulating a less-cleaned version of the data.
# 
# The data dictionary provided with data is as follows:
# 
# dateCrawled - When this ad was first crawled. All field-values are taken from this date.
# name - Name of the car.
# seller - Whether the seller is private or a dealer.
# offerType - The type of listing
# price - The price on the ad to sell the car.
# abtest - Whether the listing is included in an A/B test.
# vehicleType - The vehicle Type.
# yearOfRegistration - The year in which which year the car was first registered.
# gearbox - The transmission type.
# powerPS - The power of the car in PS.
# model - The car model name.
# kilometer - How many kilometers the car has driven.
# monthOfRegistration - The month in which which year the car was first registered.
# fuelType - What type of fuel the car uses.
# brand - The brand of the car.
# notRepairedDamage - If the car has a damage which is not yet repaired.
# dateCreated - The date on which the eBay listing was created.
# nrOfPictures - The number of pictures in the ad.
# postalCode - The postal code for the location of the vehicle.
# lastSeenOnline - When the crawler saw this ad last online.

# In[73]:


import pandas as pd  # importing pandas library 


# In[74]:


import numpy as np # importing numpy library 


# In[75]:


autos = pd.read_csv("autos.csv", encoding ="Latin- 1")  # reading file using pandas 


# In[76]:


#autos  


# In[77]:


autos.columns  # prints an array of the existing column names 


# In[78]:


autos.columns = autos.columns.str.replace("yearOfRegistration","registration_year" )


# In[79]:


autos.columns = autos.columns.str.replace("monthOfRegistration","registration_month" )


# In[80]:


autos.columns = autos.columns.str.replace("notRepairedDamage","unrepaired_damage" )


# In[81]:


autos.columns = autos.columns.str.replace("dateCreated","ad_created" )


# In[82]:


autos.columns = autos.columns.str.replace("offerType","offer_type" )


# In[83]:


autos.columns = autos.columns.str.replace("dateCrawled","date_crawled" )


# In[84]:


autos.columns = autos.columns.str.replace("vehicleType","vehicle_type" )


# In[85]:


autos.columns = autos.columns.str.replace("'powerPS","'power_pS" )


# In[86]:


autos.columns = autos.columns.str.replace("fuelType","'fuel_type" )


# In[87]:


autos.columns = autos.columns.str.replace("vehicleType","'vehicle_type" )


# In[88]:


autos.columns = autos.columns.str.replace("notRepairedDamage","'notRepaired_damage" )


# In[89]:


autos.columns = autos.columns.str.replace("dateCreate","date_create" )


# In[90]:


autos.columns = autos.columns.str.replace("nrOfPictures","nrof_pictures" )


# In[91]:


autos.columns = autos.columns.str.replace("postalCode","postal_code" )


# In[92]:


autos.columns = autos.columns.str.replace("lastSeen","last_seen" )


# In[93]:


#autos.columns


# In[94]:


autos.head()  # Display the first five rows


# we changed some of the column names from camelcase to snakecase and also we renamed some of the columns, after we printed using the dataframe.head() to see our changes 

# In[95]:


#autos.describe(include = "all")


# In[96]:


autos.drop("seller",axis = 1, inplace = True)


# In[97]:


autos.columns


# We dropped columns that have low unique values, we dropped seller, offer_type, abtest

# In[98]:


#autos.describe(include = "all")


# we will find columns that need more investigations 

# price and odometer columns have numeric values stored as text , we will remove non-numeric charactors and convert the column to numeric dtype 

# In[99]:


autos["price"] = autos["price"].str.replace("[$]", "")  #removed $ sign


# In[100]:


autos.head()


# In[101]:


autos["price"] = autos["price"].str.replace(",", "")


# In[102]:


autos["odometer"] = autos["odometer"].str.replace("km", "",)


# In[103]:


autos["odometer"] = autos["odometer"].str.replace("[,]", "")


# In[104]:


autos["odometer"] = autos["odometer"].str.replace(", ", "")


# In[105]:


autos.rename({"odometer": "odometer_km"}, axis = 1, inplace =True)


# In[106]:


autos["odometer_km"] = autos["odometer_km"].astype(int)


# In[107]:


autos["price"] = autos["price"].astype(int)


# In[108]:


autos["odometer_km"].unique().shape  
#shape is one dimensional and it has 13 unique values 


# In[109]:


autos["price"].unique().shape  
#shape is one dimensional and it has 2357 unique values 


# In[110]:


autos["odometer_km"].describe() 
# Call the describe() method on the "odometer_km" column
# to generate descriptive statistics


# In[111]:


autos["price"].describe() # Access the "price" column in the DataFrame "autos"
  # Call the describe() method on the "price" column


# we found an outlier for the odometer_km column of  150000km  we will remove since its very high milage on the used cars 

# In[112]:


autos["odometer_km"].value_counts().head().sort_index(ascending = False)

# Call the value_counts() method on the "odometer_km" column
# to compute the count of each unique value 


# In[113]:


autos["price"].value_counts().head().sort_index(ascending = True)
# Call the value_counts() method on the "price" column
# to compute the count of each unique value


# 

# In[114]:


autos


# The odometer column doesnt seem to have any outliers

# To begin, let's examine the formatting of the three string columns that contain timestamp values.

# In[115]:


autos[['date_crawled','ad_created','last_seen']][0:5]


# To determine the date range, we can extract the date values from the timestamp columns, generate a distribution using the Series.value_counts() method, and then sort the results by the index.

# In[116]:


print(autos['date_crawled'].str[:10])


# In[117]:


autos["last_seen"].sort_index()


# In[118]:


autos["date_crawled"].str[:10].value_counts(normalize = True, dropna =False).sort_index()


# looking the date_crawled column we can see there is consistant traffic for all of dates list

# In[119]:


autos['ad_created'].str[:10].value_counts(normalize=True, dropna=False).sort_index()


# In[120]:


autos['last_seen'].str[:10].value_counts(normalize=True, dropna=False).sort_index()


# last_seen seems to coincide or overlap with date_crawled

# The maximum and minimum years in the "registration_year" column suggest that they are outside the plausible range for car registrations. The mean value indicates that the majority of car registrations occurred in 2005. However, there are values in the "registration_year" column that seem improbable, such as years like 1111 or 4500, which are not consistent with the first car being made in 1885.

# lowest acceptable values was 1959 due to the amount of safety features introduce in the 1950's
# inspections are a common practice for car registration, so it would be best to include a range of cars with the most advance safety features

# 1962 had the lowest car registeration. 
# 
# 2009 had the highest car registeration

# 
# Provide the range of acceptable values for the "registration_year" column, determining the highest and lowest values

# In[ ]:





# The mean registeration was 2005 
# The most registration year was 2003 - 2008
# the data gave the weong min and max since those are out off probable year so it's incorrect

# One option is to remove the listings with these values. Let's determine what percentage of our data has invalid values in this column:

# In[121]:


(~autos["registration_year"].between(1900,2016)).sum() / autos.shape[0]


# In[122]:


#created varaible for the column registeation year
green = autos["registration_year"]
#created boolean series 
blue = autos["registration_year"].between(1900, 2016)
#created another varaible and then you applied the column named green to the boolean
result = green[blue]
#reasigned the result back to the column name
result= autos["registration_year"] 


# In[123]:


autos["registration_year"] .describe()


# In[124]:


autos["registration_year"].value_counts(normalize=True)


# 2008 had the most registration 
# between 1939 to 1952 had the lowest registration 
# 

# In[125]:


autos["brand"].unique()


# In[126]:


autos["brand"].describe()


# In[127]:


autos["brand"].value_counts(normalize=True)


# German manufacturers dominate the top brands, accounting for four out of the top five. They also make up almost half of the total listings. Among them, Volkswagen is the clear leader, with roughly twice as many cars for sale compared to the next two popular brands combined.
# 
# To focus our analysis on brands with substantial representation, we will only consider those that account for more than 5% of the total listings.

# we selected only the car models that have greater then 5 of the total listing 

# In[128]:


brand_counts = autos["brand"].value_counts(normalize=True)
common_brands = brand_counts[brand_counts > .05].index
print(common_brands)


# In[129]:


brand_mean_prices = {}

for brand in common_brands:
    brand_only = autos[autos["brand"] == brand]
    mean_price = brand_only["price"].mean()
    brand_mean_prices[brand] = int(mean_price)

brand_mean_prices


# Among the top five brands, there exists a notable difference in prices. Audi, BMW, and Mercedes Benz are positioned as higher-priced options, while Ford and Opel are comparatively more affordable. Volkswagen falls in the middle range, which could possibly explain its popularity as it offers a balanced choice, combining elements from both ends of the price spectrum.

# Exploring Mileage

# In[130]:


bmp_series = pd.Series(brand_mean_prices)
pd.DataFrame(bmp_series, columns=["mean_price"])


# In[131]:


brand_mean_mileage = {}

for brand in common_brands:
    brand_only = autos[autos["brand"] == brand]
    mean_mileage = brand_only["odometer_km"].mean()
    brand_mean_mileage[brand] = int(mean_mileage)

mean_mileage = pd.Series(brand_mean_mileage).sort_values(ascending=False)
mean_prices = pd.Series(brand_mean_prices).sort_values(ascending=False)


# In[132]:


brand_info = pd.DataFrame(mean_mileage,columns=['mean_mileage'])
brand_info


# In[133]:


brand_info["mean_price"] = mean_prices
brand_info


# The range of car mileages does not vary as much as the prices do by brand, instead all falling within 10% for the top brands. There is a slight trend to the more expensive vehicles having higher mileage, with the less expensive vehicles having lower mileage.
