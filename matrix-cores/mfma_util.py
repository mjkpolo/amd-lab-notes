#!/usr/bin/env python

import fire
import pandas as pd
from pathlib import Path


def get_pivoted(csv_fn):
    rocprof_df = pd.read_csv(csv_fn)

    unique_counters = list(
        c for c in rocprof_df.columns.to_list() if
        c != "Counter_Name" and c != "Counter_Value"
    )

    pivoted_df = rocprof_df.pivot_table(
        index=unique_counters,
        columns="Counter_Name",
        values="Counter_Value",
    ).sort_values("Start_Timestamp", ascending=True).reset_index()
    return pivoted_df


def derive_mfma_util(df):
    # WARN only for MI300x, otherwise change n_xcd and n_cu
    n_xcd = 8
    n_cu = 304
    df["MFMA Util"] = (
        100 * df.get("SQ_VALU_MFMA_BUSY_CYCLES") /
        (n_cu * df.get("GRBM_GUI_ACTIVE") / n_xcd * 4)
    )


def main(counter: str):
    counter_dir_path = Path(counter)
    if counter_dir_path.is_dir():
        csvs = tuple(counter_dir_path.rglob("*_counter_collection.csv"))
        assert len(csvs) == 1
        csv_fn = csvs[0]
    else:
        csv_fn = counter
    df = get_pivoted(csv_fn)
    derive_mfma_util(df)
    mfma_util = df["MFMA Util"].values
    print("MFMA Util (%):")
    print(mfma_util)


if __name__ == "__main__":
    fire.Fire(main)
