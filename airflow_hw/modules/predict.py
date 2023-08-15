import os
import pandas as pd
import dill
import json


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = dill.load(file)
    return model


def make_predictions(model, data_folder):
    predictions_list = []

    for filename in os.listdir(data_folder):
        if filename.endswith('.json'):
            with open(os.path.join(data_folder, filename), 'r') as json_file:
                json_data = json.load(json_file)

            df = pd.DataFrame([json_data])
            predictions = model.predict(df)
            predictions_list.append(predictions[0])

    return predictions_list


def main():
    model_path = '/Users/user/airflow_hw/data/models/cars_pipe_202308141712.pkl'
    data_folder = '/Users/user/airflow_hw/data/test'

    model = load_model(model_path)
    predictions_list = make_predictions(model, data_folder)

    result_df = pd.DataFrame({'predictions': predictions_list})


    result_path = '/Users/user/airflow_hw/data/predictions/predictions.csv'

    result_df.to_csv(result_path, index=False)

    print(f'Предсказания для всех файлов сохранены в файл: {result_path}')


if __name__ == '__main__':
    main()