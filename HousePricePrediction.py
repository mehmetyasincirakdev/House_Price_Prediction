import warnings

import matplotlib.pyplot as plot
import numpy
import pandas
import seaborn
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pandas.set_option("display.max_columns", None)
pandas.set_option("display.width", None)
pandas.set_option("display.float_format", lambda x: "%.3f" % x)

# Test ve train datasetlerimizi okutalım.
train = pandas.read_csv("Datasets/train.csv")
test = pandas.read_csv("Datasets/test.csv")

# Train ve test datasetlerini birleştirelim.
dataframe = train.append(test, ignore_index=False).reset_index()
dataframe = dataframe.drop("index", axis=1)
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
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_dataframe(dataframe)


# Numeric ve kategorik değişkenlerin yakalanması

def grab_col_names(dataFrame, cat_th=10, car_th=20):
    """

    :param dataFrame:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
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


# Kategorik Değişken Analizi

def cat_summary(dataFrame, col_name, plot=False):
    print(pandas.DataFrame({col_name: dataFrame[col_name].value_counts(), "Ratio": 100 * dataFrame[col_name].value_counts() / len(dataFrame)}))
    if plot:
        seaborn.countplot(x=dataFrame[col_name], data=dataFrame)
        plot.show(block=True)


for col in cat_cols:
    cat_summary(dataframe, col)


def num_summary(dataFrame, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataFrame[numerical_col].describe(quantiles).T)

    if plot:
        dataFrame[numerical_col].hist(bins=50)
        plot.xlabel(numerical_col)
        plot.tittle(numerical_col)
        plot.show(block=True)
        print("#################################################")


for col in num_cols:
    num_summary(dataframe, col)


def target_summary_with_cat(dataFrame, target, categorical_col):
    print(pandas.DataFrame({"TARGET_MEAN": dataFrame.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(dataframe, "SalePrice", col)

dataframe["SalePrice"].hist(bins=100)
plot.show(block=True)

numpy.log1p(dataframe["SalePrice"]).hist(bins=50)
plot.show(block=True)

correlation = dataframe[num_cols].corr()
correlation

seaborn.set(rc={"figure.figsize": (12, 12)})
seaborn.heatmap(correlation, cmap="RdBu")
plot.show(block=True)


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    correlation = dataframe.corr()
    cor_matrix = correlation.abs()
    upper_triangle_matrix = cor_matrix.where(numpy.triu(numpy.ones(cor_matrix.shape), k=1).astype(numpy.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as seaborn
        import matplotlib.pyplot as plot
        seaborn.set(rc={'figure.figsize': (15, 15)})
        seaborn.heatmap(correlation, cmap="RdBu")
        plot.show(block=True)
    return drop_list


high_correlated_cols(dataframe, plot=False)


# Feature Engineering
#####################
# Aykırı Analizi

# Aykırı değerlerin baskılanması

def outlier_thresholds(dataFrame, variable, low_quantile=0.10, upper_quantile=0.90):
    quantile_one = dataFrame[variable].quantile(low_quantile)
    quantile_three = dataFrame[variable].quantile(upper_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataFrame, col_name):
    low_limit, up_limit = outlier_thresholds(dataFrame, col_name)
    if dataFrame[(dataFrame[col_name] > up_limit) | (dataFrame[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(dataframe, col))


def replace_with_thresholds(dataFrame, variable):
    low_limit, up_limit = outlier_thresholds(dataFrame, variable)
    dataFrame.loc[(dataFrame[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataFrame[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(dataframe, col)


def missing_values_table(dataFrame, na_name=False):
    na_columns = [col for col in dataFrame.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataFrame[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataFrame[na_columns].isnull().sum() / dataFrame.shape[0] * 100).sort_values(ascending=False)
    missing_dataframe = pandas.concat([n_miss, numpy.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_dataframe, end="\n")

    if na_name:
        return na_columns


missing_values_table(dataframe)

dataframe["Alley"].value_counts()
dataframe["BsmtQual"].value_counts()

no_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
           "Fence", "MiscFeature"]

for col in no_cols:
    dataframe[col].fillna("No", inlace=True)
missing_values_table(dataframe)


def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")
    return data


dataframe = quick_missing_imp(dataframe, num_method="median", cat_length=17)


# Rare analizi yapınız ve rare encoder uygulayınız.
######################################

# Kategorik kolonların dağılımının incelenmesi

def rare_analyser(dataFrame, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pandas.DataFrame({"COUNT": dataFrame[col].value_counts(),
                                "RATIO": dataFrame[col].value_counts() / len(dataFrame),
                                "TARGET_MEAN": dataFrame.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(dataframe, "SalePrice", cat_cols)


# Nadir sınıfların tespit edilmesi
def rare_encoder(dataFrame, rare_perc):
    temp_dataframe = dataFrame.copy()

    rare_columns = [col for col in temp_dataframe.columns if temp_dataframe[col].dtypes == 'O'
                    and (temp_dataframe[col].value_counts() / len(temp_dataframe) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_dataframe[var].value_counts() / len(temp_dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        temp_dataframe[var] = numpy.where(temp_dataframe[var].isin(rare_labels), 'Rare', temp_dataframe[var])

    return temp_dataframe


rare_encoder(dataframe, 0.01)

# yeni değişkenler oluşturunuz ve oluşturduğunuz yeni değişkenlerin başına 'NEW' ekleyiniz.
######################################


dataframe["NEW_1st*GrLiv"] = dataframe["1stFlrSF"] * dataframe["GrLivArea"]

dataframe["NEW_Garage*GrLiv"] = (dataframe["GarageArea"] * dataframe["GrLivArea"])

dataframe["TotalQual"] = dataframe[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                                    "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis=1)  # 42

# Total Floor
dataframe["NEW_TotalFlrSF"] = dataframe["1stFlrSF"] + dataframe["2ndataframelrSF"]  # 32

# Total Finished Basement Area
dataframe["NEW_TotalBsmtFin"] = dataframe.BsmtFinSF1 + dataframe.BsmtFinSF2  # 56

# Porch Area
dataframe["NEW_PorchArea"] = dataframe.OpenumpyorchSF + dataframe.EnclosedPorch + dataframe.Screenumpyorch + dataframe["3Ssnumpyorch"] + dataframe.WoodDeckSF  # 93

# Total House Area
dataframe["NEW_TotalHouseArea"] = dataframe.NEW_TotalFlrSF + dataframe.TotalBsmtSF  # 156

dataframe["NEW_TotalSqFeet"] = dataframe.GrLivArea + dataframe.TotalBsmtSF  # 35

# Lot Ratio
dataframe["NEW_LotRatio"] = dataframe.GrLivArea / dataframe.LotArea  # 64

dataframe["NEW_RatioArea"] = dataframe.NEW_TotalHouseArea / dataframe.LotArea  # 57

dataframe["NEW_GarageLotRatio"] = dataframe.GarageArea / dataframe.LotArea  # 69

# MasVnrArea
dataframe["NEW_MasVnrRatio"] = dataframe.MasVnrArea / dataframe.NEW_TotalHouseArea  # 36

# Dif Area
dataframe["NEW_DifArea"] = (dataframe.LotArea - dataframe["1stFlrSF"] - dataframe.GarageArea - dataframe.NEW_PorchArea - dataframe.WoodDeckSF)  # 73

dataframe["NEW_OverallGrade"] = dataframe["OverallQual"] * dataframe["OverallCond"]  # 61

dataframe["NEW_Restoration"] = dataframe.YearRemodAdd - dataframe.YearBuilt  # 31

dataframe["NEW_HouseAge"] = dataframe.YrSold - dataframe.YearBuilt  # 73

dataframe["NEW_RestorationAge"] = dataframe.YrSold - dataframe.YearRemodAdd  # 40

dataframe["NEW_GarageAge"] = dataframe.GarageYrBlt - dataframe.YearBuilt  # 17

dataframe["NEW_GarageRestorationAge"] = numpy.abs(dataframe.GarageYrBlt - dataframe.YearRemodAdd)  # 30

dataframe["NEW_GarageSold"] = dataframe.YrSold - dataframe.GarageYrBlt  # 48

drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope", "Heating", "PoolQC", "MiscFeature", "Neighborhood"]

# drop_list'teki değişkenlerin düşürülmesi
dataframe.drop(drop_list, axis=1, inumpylace=True)

# Label Encoding & One-Hot Encoding işlemlerini uygulayınız.
##################

cat_cols, cat_but_car, num_cols = grab_col_names(dataframe)


def label_encoder(dataFrame, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataFrame[binary_col])
    return dataFrame


binary_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and len(dataframe[col].unique()) == 2]

for col in binary_cols:
    label_encoder(dataframe, col)


def one_hot_encoder(dataFrame, categorical_cols, drop_first=False):
    dataframe = pandas.get_dummies(dataFrame, columns=categorical_cols, drop_first=drop_first)
    return dataframe


dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)

# MODELLEME
##################################

##################################
# GÖREV 3: Model kurma
##################################

#  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
train_dataframe = dataframe[dataframe['SalePrice'].notnull()]
test_dataframe = dataframe[dataframe['SalePrice'].isnull()]

y = train_dataframe['SalePrice']  # numpy.log1p(dataframe['SalePrice'])
X = train_dataframe.drop(["Id", "SalePrice"], axis=1)

# Train verisi ile model kurup, model başarısını değerlendiriniz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          # ("Ridge", Ridge()),
          # ("Lasso", Lasso()),
          # ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          # ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
# ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = numpy.mean(numpy.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# BONUS : Log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz.
# Not: Log'un tersini (inverse) almayı unutmayınız.
##################

# Log dönüşümünün gerçekleştirilmesi


train_dataframe = dataframe[dataframe['SalePrice'].notnull()]
test_dataframe = dataframe[dataframe['SalePrice'].isnull()]

y = numpy.log1p(train_dataframe['SalePrice'])
X = train_dataframe.drop(["Id", "SalePrice"], axis=1)

# Verinin eğitim ve tet verisi olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# lgbm_tuned = LGBMRegressor(**lgbm_gs_best.best_params_).fit(X_train, y_train)

lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

y_pred
# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması
new_y = numpy.expm1(y_pred)
new_y
new_y_test = numpy.expm1(y_test)
new_y_test

numpy.sqrt(mean_squared_error(new_y_test, new_y))

# RMSE : 22118.413146021652


##################
# hiperparametre optimizasyonlarını gerçekleştiriniz.
##################


lgbm_model = LGBMRegressor(random_state=46)

rmse = numpy.mean(numpy.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500]
               # "colsample_bytree": [0.5, 0.7, 1]
               }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = numpy.mean(numpy.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))


################################################################
# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
################################################################

# feature importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pandas.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plot.figure(figsize=(10, 10))
    seaborn.set(font_scale=1)
    seaborn.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plot.title("Features")
    plot.tight_layout()
    plot.show()
    if save:
        plot.savefig("importances.png")


model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)

########################################
# test dataframeindeki boş olan salePrice değişkenlerini tahminleyiniz ve
# Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturunuz. (Id, SalePrice)
########################################

model = LGBMRegressor()
model.fit(X, y)
predictions = model.predict(test_dataframe.drop(["Id", "SalePrice"], axis=1))

dictionary = {"Id": test_dataframe.index, "SalePrice": predictions}
dataframeSubmission = pandas.DataFrame(dictionary)
dataframeSubmission.to_csv("housePricePredictions.csv", index=False)
