
# BUSINESS ANALYTICS PROJECT - SUPERMARKET SALES
# Dataset: Supermarket_Sales_Business_Analytics.csv
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import statsmodels.formula.api as smf
 
df = pd.read_csv("C:/Users/info/Downloads/Supermarket_Sales_Business_Analytics.csv", parse_dates=["Date"])
 sa
# 1. EDA
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
for c in df.columns:
    print(c, df[c].nunique())
 
# 2. Descriptive statistics
cols=["Quantity","Sales","Discount","Net_Sales","Profit"]
for c in cols:
    s=df[c]
    print(c, "Mean:",s.mean(), "Median:",s.median(),
          "Mode:",s.mode().iloc[0], "Min:",s.min(), "Max:",s.max(),
          "Range:",s.max()-s.min(), "Variance:",s.var(),
          "Std:",s.std())
 
# 3. Visualizations
plt.hist(df["Sales"], bins=20); plt.title("Distribution of Gross Sales"); plt.show()
plt.hist(df["Profit"], bins=20); plt.title("Distribution of Profit"); plt.show()
plt.boxplot(df["Sales"]); plt.title("Box Plot of Gross Sales"); plt.show()
plt.scatter(df["Quantity"], df["Sales"]); plt.title("Quantity vs Sales"); plt.show()
 
# 4. Correlation
numeric=df[["Quantity","Unit_Price","Discount","Sales","Net_Sales","Profit"]]
print(numeric.corr())
plt.imshow(numeric.corr(),aspect="auto"); plt.colorbar(); plt.show()
 
# 5. Regression: Quantity -> Sales
X=df[["Quantity"]]; y=df["Sales"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
model=LinearRegression().fit(X_train,y_train)
pred=model.predict(X_test)
print("Coefficient:",model.coef_[0])
print("Intercept:",model.intercept_)
print("R2:",model.score(X_test,y_test))
print("MSE:",mean_squared_error(y_test,pred))
print("RMSE:",mean_squared_error(y_test,pred)**0.5)
print("Prediction for Quantity=5:",model.predict(pd.DataFrame({"Quantity":[5]}))[0])
 
# 6. Probability
print("P(Member):",(df["Customer_Type"]=="Member").mean())
print("P(Female):",(df["Gender"]=="Female").mean())
print("P(Profit > 500):",(df["Profit"]>500).mean())
 
# 7. Sampling / CLT
rng=np.random.default_rng(42)
for n in [30,60,100]:
    means=[rng.choice(df["Net_Sales"].to_numpy(),n,replace=False).mean() for _ in range(1000)]
    print(n, np.mean(means), np.std(means,ddof=1),
          df["Net_Sales"].std(ddof=1)/np.sqrt(n))
 
# 8. Confidence intervals
s=df["Net_Sales"]; se=s.std(ddof=1)/np.sqrt(len(s))
for conf in [.90,.95,.99]:
    t=stats.t.ppf((1+conf)/2,len(s)-1)
    print(conf, s.mean()-t*se, s.mean()+t*se)
 
# 9. Bootstrap
boot=[]
for i in range(1000):
    sample=rng.choice(s.to_numpy(),len(s),replace=True)
    boot.append(np.median(sample))
print("Bootstrap 95% CI:",np.percentile(boot,[2.5,97.5]))
 
# 10. Hypothesis test
result=stats.ttest_1samp(s,1500)
print(result)
 
# 11. A/B analytical scenario: low vs high discount
md=df["Discount"].median()
A=df.loc[df["Discount"]<=md,"Net_Sales"]
B=df.loc[df["Discount"]>md,"Net_Sales"]
print("A mean:",A.mean(),"B mean:",B.mean())
print(stats.ttest_ind(A,B,equal_var=False))
 
# 12. Time series + forecasting
monthly=df.set_index("Date").resample("MS")["Net_Sales"].sum()
train=monthly.iloc[:-3]; test=monthly.iloc[-3:]
hw=ExponentialSmoothing(train,trend="add",seasonal=None).fit()
hwf=hw.forecast(3)
ar=ARIMA(train,order=(1,1,1)).fit()
arf=ar.forecast(3)
for name,p in [("Holt",hwf),("ARIMA",arf)]:
    print(name,"MAE",mean_absolute_error(test,p),
          "MSE",mean_squared_error(test,p),
          "RMSE",mean_squared_error(test,p)**0.5)
 
# 13. Optimization scenario
# Resource assumptions are scenario inputs because the dataset has no labor/machine columns.
# Products selected from highest observed average transaction profit.
p=df.groupby("Product_Line")["Profit"].mean().sort_values(ascending=False)
p1,p2=p.index[:2]
profit1,profit2=p.iloc[0],p.iloc[1]
# Maximize profit = profit1*x1 + profit2*x2
# subject to 2*x1+x2 <= 100, x1+2*x2 <= 80, x1,x2 >= 0
# Optimal corner: x1=40, x2=20
print("Optimal:",p1,40,p2,20)
print("Maximum expected profit:",40*profit1+20*profit2)
 
# 14. Simulation
# Uses empirical Quantity distribution and observed profit per unit.
profit_per_unit=df["Profit"].sum()/df["Quantity"].sum()
for inventory in [4,6,8]:
    values=[]
    for i in range(5000):
        demand=rng.choice(df["Quantity"].to_numpy())
        sold=min(demand,inventory)
        shortage=max(demand-inventory,0)
        leftover=max(inventory-demand,0)
        profit=sold*profit_per_unit-leftover*20-shortage*10
        values.append((profit,shortage,leftover,sold/demand))
    a=np.array(values)
    print(inventory,a[:,0].mean(),a[:,1].mean(),a[:,2].mean(),a[:,3].mean())
 
# 15. ANOVA
groups=[g["Profit"].to_numpy() for _,g in df.groupby("Branch")]
print(stats.f_oneway(*groups))
 
# 16. Factorial experiment
df["Discount_Level"]=np.where(df["Discount"]<=df["Discount"].median(),"Low","High")
m=smf.ols("Net_Sales ~ C(Discount_Level)*C(Customer_Type)",data=df).fit()
print(sm.stats.anova_lm(m,typ=2))
print(df.groupby(["Discount_Level","Customer_Type"])["Net_Sales"].mean())

