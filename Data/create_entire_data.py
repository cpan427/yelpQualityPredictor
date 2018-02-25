import pandas as pd
import numpy as np

print ("Looking at the users")
# opening and cleaning the user set
users = pd.read_csv('yelp_user.csv')
userExcludeSet = {'name', 'useful'}

userHeaders = list(users)
for c in userHeaders:
	if (c in userExcludeSet):
		del users[c]

print ("Looking at the reviews")
# open and clean review set
reviews = pd.read_csv('yelp_review.csv')
reviewsExcludeSet = {'review_id', 'text'}
for c in reviewsExcludeSet:
	del reviews[c]

print ("Looking at the businesses")
# open the merged business set
biz = pd.read_csv('merged_business.csv')

print ("merge 1")
# inner join the businesses and the review
temp = pd.merge(reviews, biz, on= 'business_id', how='inner')

print ("merge 2")
# inner join the combined set with the user set
merged = pd.merge(temp, users, on='user_id', how='inner')

print ("Writing to file")
merged.to_csv('merged_raw_set.csv')