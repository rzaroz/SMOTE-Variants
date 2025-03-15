import os
import time
import psutil
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN

start_time = time.time()
src = "../../Main Dataset/wustl-ehms-2020_with_attacks_categories.csv"

df = pd.read_csv(src)
df = df.drop(["Dir", "Flgs", "SrcAddr", "DstAddr", "Sport", "Dport", "SrcBytes", "DstBytes", "SrcGap", "DstGap", "SrcGap", "Trans", "TotPkts", "SrcMac", "DstMac"], axis=1)

# Preprocessing
X = df.drop(["Attack Category", "Label"], axis=1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Attack Category"])

sm = ADASYN(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

final_y = []
for i_ in y_res:
    final_y.append(label_encoder.classes_[i_])

X_res["Attack Category"] = np.array(final_y)

X_res.to_csv("report/ADASYN-wustl-ehms-2020.csv", index=False)

print("ADASYN - Usages")
end_time = time.time()
execution_time = end_time - start_time
print(f"Time: {execution_time} second")

process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / 1024 / 1024  # تبدیل به مگابایت
print(f"Ram usage: {memory_usage} mb")


def measure_cpu_usage():
    process = psutil.Process(os.getpid())
    start_cpu = process.cpu_percent()

    end_cpu = process.cpu_percent()
    return end_cpu - start_cpu

cpu_readings = []
for i in range(10):
    cpu_usage = measure_cpu_usage()
    cpu_readings.append(cpu_usage)


avg_cpu = sum(cpu_readings) / len(cpu_readings)
print(f"CPU usage: {avg_cpu}")