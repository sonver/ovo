import os
import shutil
import random

# 源目录
female_dir = r"E:\Dataset\2025-11-大午褐\1122-4000\1_female"
male_dir = r"E:\Dataset\2025-11-大午褐\1122-4000\2_male"

# 输出根目录
base_dir = "hy_txt_snv_gauss_1122_dawuhe_no_clear"
train_val_male_dir = os.path.join(base_dir, "train_val", "male")
train_val_female_dir = os.path.join(base_dir, "train_val", "female")
test_male_dir = os.path.join(base_dir, "test", "male")
test_female_dir = os.path.join(base_dir, "test", "female")

# 创建目标文件夹
for d in [train_val_male_dir, train_val_female_dir, test_male_dir, test_female_dir]:
    os.makedirs(d, exist_ok=True)

# 固定随机种子
random.seed(42)

# 收集 .txt 文件
female_files = [os.path.join(female_dir, f) for f in os.listdir(female_dir) if f.endswith(".txt")]
male_files = [os.path.join(male_dir, f) for f in os.listdir(male_dir) if f.endswith(".txt")]

# 打乱
random.shuffle(female_files)
random.shuffle(male_files)

# 划分：前200为test，其余为train_val
test_female = female_files[:0]
train_val_female = female_files[0:]

test_male = male_files[:0]
train_val_male = male_files[0:]

# 拷贝 female
for f in test_female:
    shutil.copy(f, os.path.join(test_female_dir, os.path.basename(f)))
for f in train_val_female:
    shutil.copy(f, os.path.join(train_val_female_dir, os.path.basename(f)))

# 拷贝 male
for f in test_male:
    shutil.copy(f, os.path.join(test_male_dir, os.path.basename(f)))
for f in train_val_male:
    shutil.copy(f, os.path.join(train_val_male_dir, os.path.basename(f)))

# 输出统计
print(f"✅ Female: test={len(test_female)}, train_val={len(train_val_female)}")
print(f"✅ Male: test={len(test_male)}, train_val={len(train_val_male)}")
