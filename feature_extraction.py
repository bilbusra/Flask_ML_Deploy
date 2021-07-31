import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import GradientBoostingClassifier

def convert_data(data, cols):
    data[['splitzone', 'irrlotcode']] = data[['splitzone', 'irrlotcode']].replace({'N': 0, 'Y': 1})

    df = data[data.yearbuilt > 0]

    df['block_1000'] = (df.block <= 1000).astype(int)
    df['council_0_6'] = (df.council <= 6).astype(int)
    df['council_45_51'] = ((df.council <= 51) & (df.council > 45)).astype(int)
    df['schooldist_0_4'] = ((df.schooldist <= 4) & (df.schooldist >= 0)).astype(int)
    df['schooldist_25_30'] = ((df.schooldist >= 25)).astype(int)
    df['zipcode_10034'] = ((df.zipcode <= 10034)).astype(int)
    df['policeprct_30'] = (df.policeprct <= 30).astype(int)
    df['yearalter1_0'] = (df.yearalter1 > 0).astype(int)
    df['heathcenterdistrict_22'] = (df.healthcenterdistrict < 22).astype(int)
    df['sanitboro_1'] = (df.sanitboro <= 1).astype(int)
    df['yearalter2_0'] = (df.yearalter2 > 0).astype(int)

    cat_results = pd.get_dummies(data=df, columns=['borough', 'landuse', 'proxcode', 'lottype', 'bsmtcode'],
                                 drop_first=True)
    index_values = cat_results['index']
    cat_results= cat_results.reindex(cat_results.columns.union(cols, sort=False), axis=1, fill_value=0)

    if 'index' not in cols:
        cols.append('index')
    X = cat_results[cols]
    X.dropna(inplace=True)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.drop(columns=['index']))

    res = pd.DataFrame(X_scaled, columns=cols.remove('index'))
    res[len(cols)] = index_values

    return res