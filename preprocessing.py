import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import chi2

df = pd.read_csv("dataset.csv")
df = df.drop("URL",axis=1)  #removed URL column
df = df.drop("WHOIS_STATEPRO",axis=1) #removed STATE for simplicity
df = df.drop("WHOIS_REGDATE",axis=1)
df = df.drop("WHOIS_UPDATED_DATE",axis=1)
df = df.drop("REMOTE_IPS",axis=1)
df = df.drop("DIST_REMOTE_TCP_PORT",axis=1)
df = df.drop("CONTENT_LENGTH",axis=1) # many values are NA 


#charset had UTF-8 and utf-8 as different values so replace UTF-8 with utf-8
df = df.replace(to_replace="UTF-8",value="utf-8")

#charset had ISO-8859-1 and iso-8859-1 so replace with one value(ISO-8859 also considered)
df = df.replace(to_replace="ISO-8859-1",value="iso-8859-1")
df = df.replace(to_replace="ISO-8859",value="iso-8859-1")

#charset = pd.DataFrame(data=charset,columns=['CHARSET'])
#print(charset.value_counts())

#print(df[df['CHARSET'] == "windows-1252"]['Type'])
#print(df[df['CHARSET'] == "windows-1251"]['Type'])

df = df.replace(".*Apache.*","apache",regex=True)
df = df.replace(".*nginx.*","nginx",regex=True)
df = df.replace(".*Microsoft.*","Microsoft-IIS",regex=True)
df = df.replace("mw.*codfw.*","mw.codfw.wmnet",regex=True)

servers = df.loc[:,'SERVER']

other = []
serverSet = set(servers)
for server in serverSet:
    current = df[df['SERVER'] == server]['Type']
    currentSet = set(current)
    if(len(currentSet) == 1 and currentSet.pop() == 0):
        other.append(server)

servers = servers.replace(to_replace=other,value="Other")
df = df.drop("SERVER",axis=1)
#malicious = df[df['Type'] == 1]
#maliciousServers = malicious['SERVER']
#print(len(set(maliciousServers)))

#get countries where Type is 0 (also less no of entries in the set)
#after getting the countries replace them with "OTH"
other = []
countries = df.loc[:,'WHOIS_COUNTRY']

countrySet = set(countries)
for country in countrySet:
    current = df[df['WHOIS_COUNTRY'] == country]['Type']
    currentSet = set(current)
    if(len(currentSet) == 1 and currentSet.pop() == 0):
        other.append(country)

other.append("None")
countries = countries.replace(to_replace=other,value="OTH")

#fix uppercase and lowercase values
countries = countries.replace(to_replace="us",value="US")
countries = countries.replace(to_replace='ru',value="RU")
df = df.drop("WHOIS_COUNTRY",axis=1)


df = pd.concat([df,servers,countries],axis=1)

discrete = df.loc[:,['CHARSET','WHOIS_COUNTRY','SERVER']]
encoder = OneHotEncoder()
array = encoder.fit_transform(discrete).toarray()
#print(array)
discrete_columns = np.append(encoder.categories_[0],encoder.categories_[1])
discrete_columns = np.append(discrete_columns,encoder.categories_[2])
#print(encoder.categories_)

discrete_df = pd.DataFrame(data=array,columns=discrete_columns)

#drop actual columns and insert one hot encoding of the columns
df = df.drop("CHARSET",axis=1)
df = df.drop("SERVER",axis=1)
df = df.drop("WHOIS_COUNTRY",axis=1)

output = df.loc[:,"Type"]
df = df.drop("Type",axis=1)

#final concat
df = pd.concat([df,discrete_df,output],axis=1)
df = df.fillna(0)
#checking the chi2 test for the discrete variables
#column = "WHOIS_COUNTRY"
#cross_tab = pd.crosstab(df[column],df['Type'])
#statistic,p,dof,b = chi2_contingency(cross_tab)
#critical_value = chi2.ppf(0.95,dof)
#print(statistic,critical_value)


#write cleaned data to new file
def split():
    df.to_csv("dataset-cleaned.csv",index=False)

    df_cleaned = pd.read_csv("dataset-cleaned.csv")

    positive_instances = df_cleaned[df_cleaned['Type'] == 1]
    negative_instances = df_cleaned[df_cleaned['Type'] == 0]

    yes_train,yes_test = train_test_split(positive_instances,test_size=0.2)
    no_train,no_test = train_test_split(negative_instances,test_size=0.2)

    train = yes_train.append(no_train)
    test = yes_test.append(no_test)
    train.to_csv("dataset-cleaned-train.csv",index=False)
    test.to_csv("dataset-cleaned-test.csv",index=False)

split()
