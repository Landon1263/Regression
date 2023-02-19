import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import statsmodels.api as sm

#1. 收集数据
data = pd.read_csv('words.csv')

# 2. 数据预处理
data['percent'] = data['Number in hard mode']/data['Number of  reported results']
data = pd.get_dummies(data, columns=['Is Single Syllable', 'Is Repeated']) # 将布尔变量转换为数值变量
X = data[['Word frequency', 'Is Single Syllable_True', 'Concreteness', 'Letter frequency', 'Is Repeated_True']]
y = data['percent']

# 使用2阶多项式
poly = PolynomialFeatures(degree=8, include_bias=False)
X_poly = poly.fit_transform(X)

# 标准化自变量
X_poly = (X_poly - X_poly.mean(axis=0)) / X_poly.std(axis=0)

# 3. 拟合模型并进行正则化
model = Ridge(alpha=0.01)
alphas = np.logspace(-10, 0, 200) # 尝试不同的 alpha 值
coefficients = np.zeros((len(alphas), X_poly.shape[1])) # 存储每个 alpha 值对应的系数
for i, alpha in enumerate(alphas):
    model.set_params(alpha=alpha)
    model.fit(X_poly, y)
    coefficients[i] = model.coef_

# 4. 模型评估
R_squared = model.score(X_poly, y) # 计算拟合优度
residuals = y - model.predict(X_poly) # 计算残差
print(f"R square = {R_squared}")

# 5. 结论分析
X_poly = sm.add_constant(X_poly) # 添加常数列，方便计算 p 值
model2 = sm.OLS(y, X_poly).fit_regularized(alpha=0.01, L1_wt=0) # 使用 statsmodels 计算回归系数的 p 值并进行正则化
coef_df = pd.DataFrame({'coef': model.coef_, 'pval': model2.params[1:]})
coef_df.index = poly.get_feature_names(X.columns)
coef_df = coef_df[coef_df['pval'] < 0.05] # 只保留显著的自变量
print(coef_df)





# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# import statsmodels.api as sm
#
# # 1. 收集数据
# data = pd.read_csv('words.csv')
#
# # 2. 数据预处理
# data['percent'] = data['Number in hard mode']/data['Number of  reported results']
# data = pd.get_dummies(data, columns=['Is Single Syllable', 'Is Repeated']) # 将布尔变量转换为数值变量
# X = data[['Word frequency', 'Is Single Syllable_True', 'Concreteness', 'Letter frequency', 'Is Repeated_True']]
# y = data['percent']
#
# # 使用2阶多项式
# poly = PolynomialFeatures(degree=6, include_bias=False)
# X_poly = poly.fit_transform(X)
#
# # 标准化自变量
# X_poly = (X_poly - X_poly.mean(axis=0)) / X_poly.std(axis=0)
#
# # 3. 拟合模型
# model = LinearRegression()
# model.fit(X_poly, y)
#
# # 4. 模型评估
# R_squared = model.score(X_poly, y) # 计算拟合优度
# residuals = y - model.predict(X_poly) # 计算残差
# print(f"R square = {R_squared}")
#
# # 5. 结论分析
# X_poly = sm.add_constant(X_poly) # 添加常数列，方便计算 p 值
# model2 = sm.OLS(y, X_poly).fit() # 使用 statsmodels 计算回归系数的 p 值
# coef_df = pd.DataFrame({'coef': model.coef_, 'pval': model2.pvalues[1:]})
# coef_df.index = poly.get_feature_names(X.columns)
# coef_df = coef_df[coef_df['pval'] < 0.05] # 只保留显著的自变量
# print(coef_df)








