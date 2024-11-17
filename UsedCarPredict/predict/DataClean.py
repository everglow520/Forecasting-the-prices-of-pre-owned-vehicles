import pandas as pd
import numpy as np
import joblib
import re
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#年份转换为年
def parse_date(date_str):
    try:
        # 尝试解析格式 '%Y年'
        return pd.to_datetime(date_str, format='%Y年')
    except ValueError:
        try:
            # 尝试解析格式 '%Y'
            return pd.to_datetime(date_str, format='%Y')
        except ValueError:
            # 如果解析失败，返回 NaT（Not a Time）
            return pd.NaT

#处理第二列删除第一个空格后的值
def remove_text_after_first_space(text):
    if pd.isna(text):  # 如果是缺失值，返回缺失值
        return text
    if isinstance(text, str):
        return text.split(' ', 1)[0]  # 仅处理字符串类型的值
    return text  # 对于非字符串类型，保持原样

def convert_mileage(mileage_str):
    if '万公里' in mileage_str:
        return float(mileage_str.replace('万公里', '')) * 10000
    elif '公里' in mileage_str:
        return float(mileage_str.replace('公里', ''))
    else:
        return float(mileage_str)

def extract_numbers(x):
    if isinstance(x, str):
        match = re.search(r'[-+]?\d*\.?\d+|\d+', x)
        return float(match.group()) if match else None
    return x


df = pd.read_csv('D:\Course\Python\CDS532\data.csv')

# 设置品牌数量的阈值
threshold = 40
# 计算每个品牌的数量
brand_counts = df['brand'].value_counts()
# 保留品牌数量大于或等于阈值的品牌
brands_to_keep = brand_counts[brand_counts >= threshold].index
# 过滤数据，只保留品牌数量足够的行
df = df[df['brand'].isin(brands_to_keep)]

# 逐行解析 'start_time' 列
df['start_time'] = df['start_time'].apply(parse_date)
# 提取年份
df['year'] = df['start_time'].dt.year
# 获取当前年份
current_year = datetime.now().year
# 计算年份差
df['years_difference'] = current_year - df['year']
#删去无用行
df = df.drop(columns=['year'])
df = df.drop(columns=['start_time'])
df = df.drop(columns=['tag'])
#删去第二列前面的空格
df = df.applymap(lambda x: x[1:] if isinstance(x, str) and x.startswith(' ') else x)
# 确保列中的数据都是字符串类型，处理缺失值
df['title'] = df['title'].apply(lambda x: str(x) if pd.notna(x) else x)
# 应用函数到指定列
df['title'] = df['title'].apply(remove_text_after_first_space)

#公里转换为米
df['distance'] = df['distance'].apply(convert_mileage)
#价格单位转换
df['price'] = df['price'] * 10000
#发动机排量去掉最后的字母
df['volumn'] = df['volumn'].apply(extract_numbers)
# df.to_csv('处理后的文件.csv', index=False)

#绘制品牌直方图
brand_counts = df['brand'].value_counts()
print(brand_counts)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(25, 6))
brand_counts.plot(kind='bar')
plt.title('Brand Counts')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#相关性热力图
# plt.figure(figsize=(8, 8))
# sns.heatmap(df.corr(), annot=True)
# plt.show()



##建模
categorical_features = ['brand','gear']
numerical_features = ['distance', 'volumn', 'years_difference']

df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df.fillna(df.mean(numeric_only=True))

oneHotEncoder = OneHotEncoder(drop='first')
categorical_scaler = oneHotEncoder.fit_transform(df[categorical_features]).toarray()
# print(a)

standardScaler = StandardScaler()

numerical_scaler = standardScaler.fit_transform(df[numerical_features])
# print(numerical_scaler)

x = np.hstack([categorical_scaler, numerical_scaler])
# print(x)

y = df['price'].to_numpy()
# print(y)

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2)

random_model = RandomForestRegressor(n_estimators=300, random_state=42,n_jobs=-1)
random_model.fit(x_train, y_train)

y_pred = random_model.predict(x_test)
# print(y_pred)

test1 = random_model.score(x_train, y_train) #随机森林
test2 = random_model.score(x_test, y_test)
print("RandomForest Train Score:", test1)
print("RandomForest Test Score:", test2)

lr_model = LinearRegression() #线性回归
lr_model.fit(x_train, y_train)
test3 = lr_model.score(x_train, y_train)
test4 = lr_model.score(x_test, y_test)
print("LinearRegression Train Score:", test3)
print("LinearRegression Test Score:", test4)

tree_model = DecisionTreeRegressor(random_state=42) #决策树
tree_model.fit(x_train, y_train)
test5 = tree_model.score(x_train, y_train)
test6 = tree_model.score(x_test, y_test)
print("Decision Tree Train Score:", test5)
print("Decision Tree Test Score:", test6)

adaboost_model = AdaBoostRegressor() #AdaBoost
adaboost_model.fit(x_train, y_train)
test7 = adaboost_model.score(x_train, y_train)
test8 = adaboost_model.score(x_test, y_test)
print("AdaBoost Train Score:", test7)
print("AdaBoost Test Score:", test8)

gbm_model = GradientBoostingRegressor() #梯度提升器
gbm_model.fit(x_train, y_train)
test9 = gbm_model.score(x_train, y_train)
test10 = gbm_model.score(x_test, y_test)
print("Gradient Boosting Train Score:", test9)
print("Gradient Boosting Test Score:", test10)



joblib_file = "random_forest_model.joblib"
joblib_file2 = "lr_model.joblib"
joblib_file3 = "oneHotEncoder.joblib"
joblib_file4 = "standardScaler.joblib"
joblib.dump(random_model, joblib_file)
joblib.dump(lr_model, joblib_file2)
joblib.dump(oneHotEncoder, joblib_file3)
joblib.dump(standardScaler, joblib_file4)
print(f"模型已保存到 {joblib_file}")


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices1')
plt.title('Actual vs Predicted Prices1')
plt.show()


