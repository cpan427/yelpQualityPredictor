import pandas as pd
import numpy as np

# opening and cleaning the business set
biz = pd.read_csv('yelp_business.csv')
bizIncludeSet = {'business_id', 'stars', 'review_count', 'is_open'}

bizHeaders = list(biz)
for c in bizHeaders:
	if (c not in bizIncludeSet):
		del biz[c]

# open and clean up the business attributes
attri = pd.read_csv('yelp_business_attributes.csv')
attriExcludeSet = {'ByAppointmentOnly'}

for c in attriExcludeSet:
	del attri[c]
headers = list(attri)

for c in headers:
	if (c != "business_id"):
		attri[c] = np.where(attri[c] == "True", 1, 0)

# doing an inner join
merged = pd.merge(biz, attri, on= 'business_id', how='left')


merged.to_csv('merged_business.csv')