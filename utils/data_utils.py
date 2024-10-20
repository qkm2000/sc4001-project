from sklearn.model_selection import train_test_split
import pandas as pd


def import_malicious_phish(
    path="datasets\\malicious_phish.csv",
    fraction=0.03
):
    dataset = pd.read_csv(path)
    label_array = {
        "benign": 0,
        "phishing": 1,
        "defacement": 1,
        "malware": 1
    }
    dataset["type"] = dataset["type"].map(label_array)
    dataset.rename(columns={"type": "label"}, inplace=True)
    return final_processing(dataset, fraction)


def import_url_dataset(
    path="datasets\\URL dataset.csv",
    fraction=0.04
):
    dataset = pd.read_csv(path)
    label_array = {
        "legitimate": 0,
        "phishing": 1,
    }
    dataset["type"] = dataset["type"].map(label_array)
    dataset.rename(columns={"type": "label"}, inplace=True)
    return final_processing(dataset, fraction)


def import_phiusiil(
    path="datasets\\PhiUSIIL_Phishing_URL_Dataset.csv",
    fraction=0.1
):
    dataset = pd.read_csv(path)
    dataset = dataset[["URL", "label"]]
    dataset.rename(columns={"URL": "url"}, inplace=True)
    dataset["label"] = dataset["label"].apply(lambda x: 0 if x == 1 else 1)
    return final_processing(dataset, fraction)


def import_combined(
    path=[
        r"datasets\malicious_phish.csv",
        r"datasets\URL dataset.csv",
        r"datasets\PhiUSIIL_Phishing_URL_Dataset.csv",
    ],
    fraction=0.02
):
    mp_trn, mp_val, mp_tst = import_malicious_phish(path[0], 1)
    ud_trn, ud_val, ud_tst = import_url_dataset(path[1], 1)
    ph_trn, ph_val, ud_tst = import_phiusiil(path[2], 1)
    trn = pd.concat([mp_trn, ud_trn, ph_trn])
    val = pd.concat([mp_val, ud_val, ph_val])
    tst = pd.concat([mp_tst, ud_tst, ud_tst])
    dataset = pd.concat([trn, val, tst])
    return final_processing(dataset, fraction)


def final_processing(dataset, fraction):
    benign = dataset[dataset["label"] == 0].sample(
        frac=fraction,
        random_state=42
    )
    malicious = dataset[dataset["label"] == 1].sample(
        frac=fraction,
        random_state=42
    )

    # Split the benign data
    benign_train, benign_temp = train_test_split(
        benign,
        test_size=0.2,
        random_state=42
    )
    benign_val, benign_benchmark = train_test_split(
        benign_temp,
        test_size=0.5,
        random_state=42
    )

    # Split the malicious data
    malicious_train, malicious_temp = train_test_split(
        malicious,
        test_size=0.2,
        random_state=42
    )
    malicious_val, malicious_benchmark = train_test_split(
        malicious_temp,
        test_size=0.5,
        random_state=42
    )

    # Combine the train, validation, and benchmark datasets
    train = pd.concat([benign_train, malicious_train])
    validation = pd.concat([benign_val, malicious_val])
    benchmark = pd.concat([benign_benchmark, malicious_benchmark])

    train.reset_index(drop=True, inplace=True)
    validation.reset_index(drop=True, inplace=True)
    benchmark.reset_index(drop=True, inplace=True)

    return train, validation, benchmark
