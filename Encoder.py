from sklearn.preprocessing import LabelEncoder
import pandas as pd
pd.options.mode.chained_assignment = None

def Encode(X_Train, X_Test):
    ColumnsBool = X_Train.dtypes == 'object'
    Columns = list(ColumnsBool[ColumnsBool].index)
    if X_Test.columns.size == 0:
        X_Train[Columns] = X_Train[Columns].fillna('nan')
        for column in Columns:
            encoder = LabelEncoder()
            Train = X_Train[column][X_Train[column] != 'nan']
            encoder.fit(Train)

            Store = X_Train[column].copy()

            Store[Store != 'nan'] = encoder.transform(Train)
            X_Train[column] = Store
        return X_Train

    else:
        X_Train[Columns] = X_Train[Columns].fillna('nan')
        X_Test[Columns] = X_Test[Columns].fillna('nan')
        for column in Columns:
            encoder = LabelEncoder()
            Train = X_Train[column][X_Train[column] != 'nan']
            Test = X_Test[column][X_Test[column] != 'nan']
            encoder.fit(Train)

            Store = X_Train[column].copy()
            TestStore = X_Test[column].copy()

            Store[Store != 'nan'] = encoder.transform(Train)
            TestStore[TestStore != 'nan'] = encoder.transform(Test)

            X_Train[column] = Store
            X_Test[column] = TestStore

        return X_Train, X_Test
