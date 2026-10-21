import sklearn
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

print("All 5 algorithm families imported successfully!")
print("XGBoost version:", xgb.__version__)
print("LightGBM version:", lgb.__version__)
print("CatBoost version:", cb.__version__)
print("Scikit-Learn version:", sklearn.__version__)

