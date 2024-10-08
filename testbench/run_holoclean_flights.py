import sys
sys.path.append('../')
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *

# 1. 设置 HoloClean 会话
hc = holoclean.HoloClean(
    db_name='holo',
    db_user='holocleanuser',
    db_password='abcd1234',
    db_host='localhost',
    domain_thresh_1=0,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    epochs=10,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=True,
    timeout=3*60000,
    feature_norm=False,
    weight_norm=False,
    print_fw=True
).session

# 2. 加载数据和约束
hc.load_data('flights', 'D:\\algorithm paper\\data_cleaning\\Datasets\\flights\\8.69%.csv')
hc.load_dcs('path_to_your_constraints/flights_constraints.txt')  # 约束文件

# 3. 检测错误
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)

# 4. 修复错误
hc.setup_domain()
featurizers = [
    InitAttrFeaturizer(),
    OccurAttrFeaturizer(),
    FreqFeaturizer(),
    ConstraintFeaturizer(),
]

hc.repair_errors(featurizers)

# 5. 评估结果（如果有正确的数据）
# hc.evaluate(fpath='path_to_corrected_data/flights_clean.csv',
#             tid_col='tuple_id',
#             attr_col='attribute',
#             val_col='correct_val')
