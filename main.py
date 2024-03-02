import utils
import preprocessing
import train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--kaggle", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":

    # 1. Load the data as dataframes
    if args.kaggle:
        train_df, test_df = utils.load_data(root="/kaggle/input/pii-detection-removal-from-educational-data", extension="json")
    else:
        train_df, test_df = utils.load_data(root="data", extension="json")

    # 2. Preprocess the data
    # 2.1. Tokenize in BERT acceptable format
    # 2.2. Load the data into a custom Dataset
    # 2.3. Wrap the data into a DataLoader
    train_dataloader = preprocessing.pipeline(dataframe=train_df,
                                              label_map=utils.get_label_2_id(),
                                              max_token_len=512,
                                              batch_size=64,
                                              isTrain=True,
                                              shuffle=True)
    test_dataloader = preprocessing.pipeline(dataframe=test_df,
                                             label_map=utils.get_label_2_id(),
                                             max_token_len=512,
                                             batch_size=64,
                                             isTrain=False,
                                             shuffle=False)

    # 3. Train BERT model
    train.pipeline(train_dataloader=train_dataloader,
                   test_dataloader=test_dataloader,
                   num_epochs=1)
