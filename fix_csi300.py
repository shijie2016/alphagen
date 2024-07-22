#%%
import pandas as pd

dataroot = "~/.qlib/qlib_data/cn_data_rolling/instruments"

# Read instruments
instruments = pd.read_csv(f"{dataroot}/all.txt", sep="\t", header=None)
instruments.columns = ["instrument_id", "start_date", "end_date"]

csi300_instruments = pd.read_csv(f"{dataroot}/csi300.txt", sep="\t", header=None)
csi300_instruments.columns = ["instrument_id", "start_date", "end_date"]
#%%
# Merge instruments and csi300 instruments
# 仅保留 csi300_instruments 中的 instrument_id
csi300_ids = csi300_instruments['instrument_id']

# 合并数据，并使用 instruments 中的日期信息
merged_data = pd.merge(csi300_ids, instruments, on='instrument_id', how='inner')

# 去重，确保每个 instrument_id 只有一条记录
unique_merged_data = merged_data.drop_duplicates(subset='instrument_id')
#%%
# 保存为txt文件，保持与原始文件一致的格式
output_path = f"{dataroot}/csi300_filtered.txt"
unique_merged_data.to_csv(output_path, sep="\t", header=False, index=False)
