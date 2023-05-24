import warnings

import matplotlib.pyplot as plot
import numpy
import pandas
import seaborn
from sklearn.exceptions import ConvergenceWarning

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
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(correlation, cmap="RdBu")
        plt.show(block=True)
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


def missing_values_table(dataFrame, na_name=False):
    na_columns = [col for col in dataFrame.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataFrame[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataFrame[na_columns].isnull().sum() / dataFrame.shape[0] * 100).sort_values(ascending=False)

    missing_df = pandas.concat([n_miss, numpy.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(dataframe)

dataframe["Alley"].value_counts()
dataframe["BsmtQual"].value_counts()

no_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
           "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

# Kolonlardaki boşlukların "No" ifadesi ile doldurulması
for col in no_cols:
    dataframe[col].fillna("No", inplace=True)

missing_values_table(dataframe)


# Bu fonsksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar

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


df = quick_missing_imp(dataframe, num_method="median", cat_length=17)


# Rare analizi yapınız ve rare encoder uygulayınız.
######################################

# Kategorik kolonların dağılımının incelenmesi

def rare_analyser(dataFrame, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataFrame[col].value_counts()))
        print(pandas.DataFrame({"COUNT": dataFrame[col].value_counts(),
                                "RATIO": dataFrame[col].value_counts() / len(dataFrame),
                                "TARGET_MEAN": dataFrame.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(dataframe, "SalePrice", cat_cols)


# Nadir sınıfların tespit edilmesi
def rare_encoder(dataFrame, rare_perc):
    temp_dataFrame = dataFrame.copy()

    rare_columns = [col for col in temp_dataFrame.columns if temp_dataFrame[col].dtypes == 'O'
                    and (temp_dataFrame[col].value_counts() / len(temp_dataFrame) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_dataFrame[var].value_counts() / len(temp_dataFrame)
        rare_labels = tmp[tmp < rare_perc].index
        temp_dataFrame[var] = np.where(temp_dataFrame[var].isin(rare_labels), 'Rare', temp_dataFrame[var])

    return temp_dataFrame


rare_encoder(df, 0.01)
