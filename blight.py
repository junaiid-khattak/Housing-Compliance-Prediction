import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv', encoding='ISO-8859-1', low_memory=False)
test_df = pd.read_csv('test.csv', encoding='ISO-8859-1')

train_df = train_df.dropna(subset=['compliance'])

# Label Extraction
y = train_df['compliance']
train_df.drop(['compliance'], inplace=True, axis=1)


def process_data(df):
    ticket_ids = df['ticket_id']
    # dropping columns that won't help in classification
    if {'violation_zip_code', 'ticket_id', 'country',
                       'violator_name', 'grafitti_status', 'inspector_name',
                       'violation_street_number', 'mailing_address_str_number', 'mailing_address_str_name',
                       'non_us_str_code', 'state', 'violation_street_name',
                       'payment_date'}.issubset(df.columns):
        X = df.drop(['violation_zip_code', 'ticket_id', 'country',
                     'violator_name', 'grafitti_status', 'inspector_name',
                     'violation_street_number', 'mailing_address_str_number', 'mailing_address_str_name',
                     'non_us_str_code', 'state', 'violation_street_name',
                     'payment_date', 'payment_status', 'compliance_detail',
                     'balance_due', 'payment_amount'], axis=1)
        X = X.drop(['ticket_issued_date', 'hearing_date', 'zip_code', 'collection_status',], axis=1)
    else:
        X = df.drop(['violation_zip_code', 'ticket_id', 'country',
                     'violator_name', 'grafitti_status', 'inspector_name',
                     'violation_street_number', 'mailing_address_str_number', 'mailing_address_str_name',
                     'non_us_str_code', 'state', 'violation_street_name'], axis=1)
        X = X.drop(['ticket_issued_date', 'hearing_date', 'zip_code'], axis=1)


    # Categorical Variables
    X['agency_name'] = pd.Categorical(X['agency_name'])
    X['city'] = pd.Categorical(X['city'])
    X['violation_code'] = pd.Categorical(X['violation_code'])
    X['violation_description'] = pd.Categorical(X['violation_description'])
    X['disposition'] = pd.Categorical(X['disposition'])

    X['agency_name'] = X['agency_name'].cat.codes
    X['city'] = X['city'].cat.codes
    X['violation_code'] = X['violation_code'].cat.codes
    X['violation_description'] = X['violation_description'].cat.codes
    X['disposition'] = X['disposition'].cat.codes

    # Filling Missing Data

    X = X.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x)
    return X, ticket_ids



X, train_ids = process_data(train_df)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("Training Scores ..")
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print(rfc.score(X_test, y_test))

lgr = LogisticRegression()
lgr.fit(X_train, y_train)
print(lgr.score(X_test, y_test))
