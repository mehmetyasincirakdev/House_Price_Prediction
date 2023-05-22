import pandas
import numpy
import matplotlib.pyplot as plot
import seaborn
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree   import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

warnings.simplefilter(action="ignore",category=FutureWarning)
warnings.simplefilter("ignore",category=ConvergenceWarning)

pandas.set_option("display.max_columns",None)
pandas.set_option("display.width",None)
pandas.set_option("display.float_format",lambda x:"%.3f"%x)

# Test ve train datasetlerimizi okutalım.
train=pandas.read_csv("Datasets/train.csv")
test=pandas.read_csv("Datasets/test.csv")

# Train ve test datasetlerini birleştirelim.
dataframe=train.append(test,ignore_index=False).reset_index()
dataframe=dataframe.drop("index",axis=1)
dataframe.head()

# Genel Resim
def check_dataframe(dataFrame):
    print("########## Shape ########## ")
    print(dataframe.shape)
    print("########## Types ########## ")
    print(dataframe.dtypes)
    print("########## Head ########## ")
    print(dataframe.head(5))
    print("########## Tail ########## ")
    print(dataframe.tail(5))
    print("########## NA ########## ")
    print(dataframe.isnull().sum())
    print("########## Quantiles ########## ")
    print(dataframe.quantile([0,0.05,0.50,0.95,0.99,1]).T)
check_dataframe(dataframe)

# Numeric ve kategorik değişkenlerin yakalanması

def grab_col_names(dataFrame,cat_th=10,car_th=20):
    """

    :param dataFrame:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and  dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]


    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataFrame.shape[0]}")
    print(f"Variables: {dataFrame.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(dataframe)