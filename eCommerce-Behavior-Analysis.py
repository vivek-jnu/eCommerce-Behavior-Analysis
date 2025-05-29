
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_cust = pd.read_csv("CUSTOMERS.csv")

data_seller = pd.read_csv("Sellers.csv")
data_prod = pd.read_csv("Products.csv")
data_odr = pd.read_csv("Orders.csv")
data_odr_itms = pd.read_csv("Order_Items.csv")
data_odr_pyt = pd.read_csv("Order_Payments.csv")
data_odr_rev_rtgs = pd.read_csv("Order_Review_Ratings.csv")
data_geo_loc = pd.read_csv("Geo_Location.csv")
data_seller2 = pd.merge(data_seller,data_odr_itms, on ="seller_id", how = "inner")
data_odr_ptm = pd.merge(data_odr,data_odr_pyt, on= "order_id", how = "inner")
data_odr_ptm2 = pd.merge(data_odr_ptm,data_seller2, on="order_id", how = "inner")
data_odr_ptm3 = pd.merge(data_cust,data_odr_ptm2, on ="customer_id" , how = "inner" )
data_odr_ptm4 = pd.merge(data_prod,data_odr_ptm3, on = "product_id", how = "inner")

data1 = pd.merge(data_odr_itms, data_odr_rev_rtgs, on = "order_id", how = "inner")
data2 = pd.merge(data1,data_prod, on = "product_id", how = "inner" )
data3 = pd.merge(data2, data_seller, on = "seller_id", how = "inner")



data_cust.head()






# In[9]:


data_seller.columns = ['seller_id', 'zip_code_prefix', 'seller_city', 'seller_state']


# In[10]:


data_seller.head()


# In[11]:


data_prod.head()


# In[12]:


data_odr.head()


# In[13]:


data_odr_ptm.head()


# In[14]:


data_odr_itms.head()


# In[15]:


data_odr_pyt.head()


# In[16]:


data_odr_rev_rtgs.head()


# In[17]:


data_geo_loc.columns = ['zip_code_prefix', 'geolocation_lat', 'geolocation_lng',
       'geolocation_city', 'geolocation_state']


# In[19]:


data_geo_loc.head()


# In[ ]:





# In[20]:


data2.head()


# In[21]:


data3.columns = ['order_id', 'order_item_id', 'product_id', 'seller_id',
       'shipping_limit_date', 'price', 'freight_value', 'review_id',
       'review_score', 'review_creation_date', 'review_answer_timestamp',
       'product_category_name', 'product_name_lenght',
       'product_description_lenght', 'product_photos_qty', 'product_weight_g',
       'product_length_cm', 'product_height_cm', 'product_width_cm',
       'zip_code_prefix', 'seller_city', 'seller_state']


# In[ ]:





# In[22]:


data_geo_loc.columns = ['zip_code_prefix', 'geolocation_lat', 'geolocation_lng',
       'geolocation_city', 'geolocation_state']


# In[23]:


data_geo_loc.head()


# In[ ]:





# In[24]:


data4 = pd.merge(data3,data_geo_loc, on = "zip_code_prefix", how = "inner" )


# In[ ]:





# # 1. Perform Detailed exploratory analysis
# a. Define & calculate high level metrics 
# like(Total Revenue, Total quantity, Total 
# products, Total categories, Total sellers, Total locations, Total channels, Total 
# payment methods etc…) 

# In[25]:


# Total Revenue
total_revenue = data_odr_pyt['payment_value'].sum()

# Total Quantity
total_quantity = data_odr_itms['order_item_id'].sum()


# Total Products
total_products = data_prod['product_id'].nunique()

# Total Categories
total_categories = data_prod['product_category_name'].nunique()

# Total Sellers
total_sellers = data_seller['seller_id'].nunique()

# Total Locations
total_locations = data_geo_loc['zip_code_prefix'].nunique()

# Total Channels
total_channels = data_odr_pyt['payment_type'].nunique()

# Total Payment Methods
total_payment_methods = data_odr_pyt['payment_type'].nunique()

# Print the results
print("Total Revenue:", total_revenue)
print("Total Quantity:", total_quantity)
print("Total Products:", total_products)
print("Total Categories:", total_categories)
print("Total Sellers:", total_sellers)
print("Total Locations:", total_locations)
print("Total Channels:", total_channels)
print("Total Payment Methods:", total_payment_methods)


# In[ ]:





# # 1.b : Understanding how many new customers acquired every month

# In[26]:


data_odr['order_purchase_timestamp'] = pd.to_datetime(data_odr['order_purchase_timestamp'])


data_odr['year'] = data_odr['order_purchase_timestamp'].dt.year
data_odr['month'] = data_odr['order_purchase_timestamp'].dt.month

customer_acquisition = data_odr.groupby(['year', 'month'])['customer_id'].nunique().reset_index()
customer_acquisition.rename(columns={'customer_id': 'new_customers'}, inplace=True)
customer_acquisition.groupby("year")
customer_acquisition = customer_acquisition.sort_values(by = "month", ascending = True).reset_index()


customer_acquisition.groupby("year")["new_customers"].sum().reset_index()


# In[27]:


customer_acquisition.groupby("month")["new_customers"].sum().reset_index()


# # 1.c : Understand the retention of customers on month on month basis

# In[28]:


data_odr['order_purchase_timestamp'] = pd.to_datetime(data_odr['order_purchase_timestamp'])


data_odr['year'] = data_odr['order_purchase_timestamp'].dt.year
data_odr['month'] = data_odr['order_purchase_timestamp'].dt.month


customer_retention = data_odr.groupby(['year', 'month'])['customer_id'].nunique().reset_index()
customer_retention.rename(columns={'customer_id': 'total_customers'}, inplace=True)


customer_retention['prev_total_customers'] = customer_retention['total_customers'].shift(1)


customer_retention['retention_rate'] = (customer_retention['total_customers'] / customer_retention['prev_total_customers']) * 100


customer_retention['retention_rate'].fillna(100, inplace=True)

customer_retention


#  # 1.d :  How the revenues from existing/new customers on month on month basis

# In[29]:


data_odr_ptm = pd.merge(data_odr,data_odr_pyt, on= "order_id", how = "inner")


# In[ ]:





# In[30]:


monthly_revenue = data_odr_ptm.groupby(['year', 'month'])['payment_value'].sum().reset_index()


customer_acquisition = data_odr.groupby(['year', 'month'])['customer_id'].nunique().reset_index()
customer_acquisition.rename(columns={'customer_id': 'new_customers'}, inplace=True)


revenue_by_month = pd.merge(monthly_revenue, customer_acquisition, on=['year', 'month'], how='left')
revenue_by_month['new_customers'].fillna(0, inplace=True) 


revenue_by_month['revenue_existing_customers'] = revenue_by_month['payment_value'] - revenue_by_month['new_customers']

revenue_by_month


# In[ ]:





# # 1.e: Understand the trends/seasonality of sales, quantity by category, location, month, week, day, time, channel, payment method etc…

# In[31]:


data_odr_ptm['order_purchase_timestamp'] = pd.to_datetime(data_odr_ptm['order_purchase_timestamp'])


data_odr_ptm['year'] = data_odr_ptm['order_purchase_timestamp'].dt.year
data_odr_ptm['month'] = data_odr_ptm['order_purchase_timestamp'].dt.month

monthly_sales = data_odr_ptm.pivot_table(index =['year', 'month'], values = 'payment_value', aggfunc ="sum")


# In[32]:


monthly_sales


# In[ ]:





# #  1.f : Popular Products by month, seller, state, category.

# In[33]:


data_odr_ptm3


# In[34]:


data_odr_ptm3['order_purchase_timestamp'] = pd.to_datetime(data_odr_ptm3['order_purchase_timestamp'])
data_odr_ptm3['year'] = data_odr_ptm3['order_purchase_timestamp'].dt.year
data_odr_ptm3['month'] = data_odr_ptm3['order_purchase_timestamp'].dt.month

# Popular Products by Month
popular_products_by_month = data_odr_ptm3.groupby(['year', 'month', 'product_id'])['order_id'].count().reset_index()
popular_products_by_month.rename(columns={'order_id': 'order_count'}, inplace=True)

# Popular Products by Seller
popular_products_by_seller = data_odr_ptm3.groupby(['seller_id', 'product_id'])['order_id'].count().reset_index()
popular_products_by_seller.rename(columns={'order_id': 'order_count'}, inplace=True)

# Popular Products by State
popular_products_by_state = data_odr_ptm3.groupby(['customer_state', 'product_id'])['order_id'].count().reset_index()
popular_products_by_state.rename(columns={'order_id': 'order_count'}, inplace=True)

# Popular Products by Category
# Assuming you have a DataFrame named 'product_data' with product categories

popular_products_by_category = data_odr_ptm4.groupby(['product_category_name', 'product_id'])['order_id'].count().reset_index()
popular_products_by_category.rename(columns={'order_id': 'order_count'}, inplace=True)

# Rank the popular products
popular_products_by_month['rank'] = popular_products_by_month.groupby(['year', 'month'])['order_count'].rank(ascending=False, method='min')
popular_products_by_seller['rank'] = popular_products_by_seller.groupby(['seller_id'])['order_count'].rank(ascending=False, method='min')
popular_products_by_state['rank'] = popular_products_by_state.groupby(['customer_state'])['order_count'].rank(ascending=False, method='min')
popular_products_by_category['rank'] = popular_products_by_category.groupby(['product_category_name'])['order_count'].rank(ascending=False, method='min')


# In[37]:


popular_products_by_month


# In[38]:


popular_products_by_seller


# In[41]:


popular_products_by_state.head()


# In[42]:


popular_products_by_category.head()


# In[ ]:





# In[ ]:





# # 1.g : Popular categories by state, month

# In[ ]:





# In[43]:


data_odr_ptm4['order_purchase_timestamp'] = pd.to_datetime(data_odr_ptm4['order_purchase_timestamp'])
data_odr_ptm4['year'] = data_odr_ptm4['order_purchase_timestamp'].dt.year
data_odr_ptm4['month'] = data_odr_ptm4['order_purchase_timestamp'].dt.month



# Group the data by state, month, and product category and calculate total sales
popular_categories = data_odr_ptm4.groupby(['customer_state', 'year', 'month', 'product_category_name'])['payment_value'].sum().reset_index()

# Rank the categories within each state and month based on total sales
popular_categories['rank'] = popular_categories.groupby(['customer_state', 'year', 'month'])['payment_value'].rank(ascending=False, method='min')

# Filter and display the popular categories for specific states and months
# For example, to display popular categories in a specific state and month (e.g., 'Andhra Pradesh' in January 2018):
state = 'Andhra Pradesh'
year = 2018
month = 1
popular_categories_filtered = popular_categories[(popular_categories['customer_state'] == state) & (popular_categories['year'] == year) & (popular_categories['month'] == month)]

# Display the popular categories in the specified state and month
popular_categories_filtered


# In[44]:


popular_categories_filtered.sort_values("rank", ascending = True)


# In[ ]:





# # 1.h : List top 10 most expensive products sorted by price

# In[45]:


top_10_expensive_products = data_odr_itms.sort_values(by='price', ascending=False).head(10)

# Display the top 10 most expensive products
top_10_expensive_products


# In[ ]:





# In[ ]:





# # 2. Performing Customers/sellers Segmentation
# 2.a Divide the customers into groups based on the revenue generated 

# In[53]:


customer_revenue = data_odr_ptm4.groupby('customer_id')['payment_value'].sum().reset_index()

# Define segmentation thresholds (e.g., quartiles)
low_revenue_threshold = customer_revenue['payment_value'].quantile(0.25)
high_revenue_threshold = customer_revenue['payment_value'].quantile(0.75)

# Segment customers based on revenue
def segment_customer(revenue):
    if revenue <= low_revenue_threshold:
        return 'Low'
    elif revenue <= high_revenue_threshold:
        return 'Medium'
    else:
        return 'High'

customer_revenue['Segment'] = customer_revenue['payment_value'].apply(segment_customer)
customer_revenue.head()



# In[ ]:





# In[ ]:





# # 2.b : Divide the sellers into groups based on the revenue generated

# In[52]:


seller_revenue = data_odr_ptm4.groupby('seller_id')['payment_value'].sum().reset_index()

# Define segmentation thresholds (e.g., quartiles)
low_revenue_threshold = seller_revenue['payment_value'].quantile(0.25)
high_revenue_threshold = seller_revenue['payment_value'].quantile(0.75)

# Segment sellers based on revenue
def segment_seller(revenue):
    if revenue <= low_revenue_threshold:
        return 'Low'
    elif revenue <= high_revenue_threshold:
        return 'Medium'
    else:
        return 'High'

seller_revenue['Segment'] = seller_revenue['payment_value'].apply(segment_seller)

seller_revenue.head()


# In[ ]:





# In[ ]:





# # 3.0 Cross-Selling (Which products are selling together)
# Hint: We need to find which of the top 10 combinations of products are selling together in 
# each transaction. (combination of 2 or 3 buying together)
# 

# In[54]:


basket_sets = pd.crosstab(data4['order_id'], data4['product_category_name'])

# Calculate the co-occurrence matrix
product_co_occurrence = basket_sets.T.dot(basket_sets)

# Filter the matrix to exclude self-co-occurrence
product_co_occurrence.values[[range(product_co_occurrence.shape[0])], [range(product_co_occurrence.shape[1])]] = 0

# Find the product combinations that are frequently sold together
frequently_sold_together = product_co_occurrence.stack().sort_values(ascending=False)

# Display the top product combinations
frequently_sold_together.head(10)


# In[ ]:





# # 4. Payment Behaviour
# 4.a. How customers are paying?

# In[56]:


payment_counts = data_odr_pyt['payment_type'].value_counts()

# Create a bar chart to visualize the payment methods
plt.figure(figsize = (10, 8))
payment_counts.plot(kind='bar', color='skyblue')
plt.title('Payment Methods')
plt.xlabel('Payment Method')
plt.ylabel('Number of Customers')
plt.show()

# Display the payment method counts
print(payment_counts)


# # 4.b : Which payment channels are used by most customers?

# In[57]:


payment_channel_counts = data_odr_pyt['payment_type'].value_counts()

# Sort the payment channels by count in descending order
sorted_payment_channels = payment_channel_counts.sort_values(ascending=False)

# Create a bar chart to visualize the most used payment channels
plt.figure(figsize=(10, 6))
sorted_payment_channels.plot(kind='bar', color='skyblue')
plt.title('Most Used Payment Channels')
plt.xlabel('Payment Channel')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()

# Display the payment channels used by the most customers
sorted_payment_channels


# # 5.a Customer satisfaction towards category & product
#  Which categories (top 10) are maximum rated & minimum rated?

# In[58]:


category_avg_scores = data2.groupby('product_category_name')['review_score'].mean()

# Sort categories by average review score in descending order (maximum rated)
top_10_max_rated = category_avg_scores.sort_values(ascending=False).head(10)

# Sort categories by average review score in ascending order (minimum rated)
top_10_min_rated = category_avg_scores.sort_values().head(10)
print("Maximum rated product", top_10_max_rated)

print("Minimum Rated Product ", top_10_min_rated)


# # 5.b: Which products (top10) are maximum rated & minimum rated?

# In[59]:


product_avg_scores = data2.groupby('product_id')['review_score'].mean()

# Sort products by average review score in descending order (maximum rated)
top_10_max_rated = product_avg_scores.sort_values(ascending=False).head(10)

# Sort products by average review score in ascending order (minimum rated)
top_10_min_rated = product_avg_scores.sort_values().head(10)

top_10_max_rated


# In[60]:


# minimum rated
top_10_min_rated


# # 5.c: Average rating by location, seller, product, category, month etc.

# In[ ]:





# In[62]:



data5 = pd.merge(data4, data_prod, on = "product_id", how = "inner")


# In[ ]:





# In[63]:


data = data4.pivot_table(index = "geolocation_state", values = "review_score" , aggfunc = "mean")


# In[64]:


data.sort_values("review_score", ascending = False)


# In[71]:


data_ = data5.pivot_table(index = "product_category_name_y", values = "review_score",aggfunc = "mean" )


# In[72]:


data_


# In[ ]:




