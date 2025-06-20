import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    下载和解压数据集
    """

    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    
    # 下载url指定路径的文件，然后把这个内容写入名为zip_path的文件中
    with urllib.request.urlopen(url) as response:
        # 打开一个名为zip_path的本地文件，用于二进制写入（"wb"）。
        with open(zip_path, "wb") as out_file:
            # 将远程response的内容写入out_file中
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved to {data_file_path}")

def create_balanced_dataset(df):
    """
    创建平衡数据集
    """

    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    # print(ham_subset.shape[0])
    balance_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balance_df

def random_split(df, train_frac, validation_frac):
    """
    随机拆分数据集
    """

    #  随机打乱数据
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    #  计算拆分索引
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    
    #  拆分数据集
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

if __name__ == "__main__":

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    # print(df)
    print(df["Label"].value_counts())
    # print(df[df["Label"] == "spam"].shape[0])
    # print(df[df["Label"] == "ham"].shape[0])

    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    """
    Label
    0    747
    1    747
    Name: count, dtype: int64
    """
    print(balanced_df["Label"].value_counts())

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # train_df:1045, validation_df:149, test_df:300
    print(f"train_df:{train_df.shape[0]}, validation_df:{validation_df.shape[0]}, test_df:{test_df.shape[0]}")

    # 保存数据集
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)