#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


def main():

    def get_csv(dir):
        mfma = tuple(Path(dir).rglob("*_counter_collection.csv"))
        assert len(mfma) == 1
        return mfma[0]

    full_mfma = get_csv("FULL_MFMA")
    startup_mfma = get_csv("STARTUP_MFMA")

    dp_flops = 4096*4096*14336*2

    full_df = pd.read_csv(full_mfma)
    startup_df = pd.read_csv(startup_mfma)

    def get_counter_value(df, counter):
        return df[df['Counter_Name'] == counter]['Counter_Value'].values

    def get_mfma_util(df, startup_cycles=None):
        GRBM_GUI_ACTIVE = get_counter_value(df, 'GRBM_GUI_ACTIVE')
        if startup_cycles is not None:
            GRBM_GUI_ACTIVE -= startup_cycles

        SQ_VALU_MFMA_BUSY_CYCLES = get_counter_value(
            df, 'SQ_VALU_MFMA_BUSY_CYCLES')

        n_xcd = 8
        n_cu = 304
        return 100 * SQ_VALU_MFMA_BUSY_CYCLES / (GRBM_GUI_ACTIVE / n_xcd * n_cu * 4)

    startup_cycles = get_counter_value(startup_df, 'GRBM_GUI_ACTIVE')

    print('unadjusted mfma util %:', get_mfma_util(full_df))
    print('no startup mfma util %:', get_mfma_util(full_df, startup_cycles))

    actual_flops = get_counter_value(full_df, 'SQ_INSTS_VALU_MFMA_MOPS_F16')*512
    print(f'% diff to f_fc_dp flops: {(dp_flops-actual_flops)/dp_flops*100}')


if __name__ == "__main__":
    exit(main())
