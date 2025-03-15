import os
import time
import psutil
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

start_time = time.time()

df_src = "../Main Dataset/UNSW_NB15_training-set.csv"
df = pd.read_csv(df_src)

# Preprocessing
object_features = ["proto", "service", "state"]
X = df.drop(["id", "label", "attack_cat"], axis=1)

for obj in object_features:
    X[obj] = LabelEncoder().fit_transform(X[obj])
print("All object features changes to numeric!")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["attack_cat"])

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

final_y = []
for i_ in y_res:
    final_y.append(label_encoder.classes_[i_])

X_res["attack_cat"] = np.array(final_y)

# X_res.to_csv("report/SMOTE-UNSW_NB15.csv", index=False)


print("SMOTE - Usages")
end_time = time.time()
execution_time = end_time - start_time
print(f"Time: {execution_time} second")

process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / 1024 / 1024  # تبدیل به مگابایت
print(f"Ram usage: {memory_usage} mb")

cpu_usage = process.cpu_percent(interval=1.0)
print(f"CPU usage: {cpu_usage}%")