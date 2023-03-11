import json
import dill
import os
from datetime import datetime
import pandas as pd


def predict():
    path = os.environ.get('PROJECT_PATH', '.')
    #path = 'C:/Users/almir/0_Skillbox/ds-intro/33_airflow/airflow_hw'

    last_model = os.listdir(path + '/data/models/')
    last_model.sort(reverse=True)

    with open(path + '/data/models/' + last_model[0], 'rb') as file:
        model = dill.load(file)

    df_list = list()
    for file_json in os.listdir(path + '/data/test/'):
        with open(path + '/data/test/' + file_json, 'rb') as file:
            df_list.append(json.load(file))

    df = pd.DataFrame(df_list)
    df['pred'] = model.predict(df)

    pred_file = f'{path}/data/predictions/preds{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df[['id', 'pred']].to_csv(pred_file, index=False)


if __name__ == '__main__':
    predict()
