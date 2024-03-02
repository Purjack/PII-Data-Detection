import pandas as pd


def get_label_2_id():
    return {
        'O': 0,
        'B-NAME_STUDENT': 1,
        'I-NAME_STUDENT': 2,
        'B-EMAIL': 3,
        'I-EMAIL': 4,
        'B-USERNAME': 5,
        'I-USERNAME': 6,
        'B-ID_NUM': 7,
        'I-ID_NUM': 8,
        'B-PHONE_NUM': 9,
        'I-PHONE_NUM': 10,
        'B-URL_PERSONAL': 11,
        'I-URL_PERSONAL': 12,
        'B-STREET_ADDRESS': 13,
        'I-STREET_ADDRESS': 14
    }


def get_id_2_label():
    return {
        0: 'O',
        1: 'B-NAME_STUDENT',
        2: 'I-NAME_STUDENT',
        3: 'B-EMAIL',
        4: 'I-EMAIL',
        5: 'B-USERNAME',
        6: 'I-USERNAME',
        7: 'B-ID_NUM',
        8: 'I-ID_NUM',
        9: 'B-PHONE_NUM',
        10: 'I-PHONE_NUM',
        11: 'B-URL_PERSONAL',
        12: 'I-URL_PERSONAL',
        13: 'B-STREET_ADDRESS',
        14: 'I-STREET_ADDRESS'
    }


def load_data(
    root: str,
    extension: str
):
    if extension == "json":
        import json
        with open(f"{root}/train.json", 'r') as f:
            train_data = json.load(f)
        with open(f"{root}/test.json", 'r') as f:
            test_data = json.load(f)
    else:
        raise ValueError("Extension not found.")

    # Turn all data into DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    return train_df, test_df
