import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
sns.set(style = 'darkgrid')

df = pd.read_csv('Training_DataSet.csv')

col_names = df.columns

#df.info()

#returns the unique values of the category
uni = df['SellerCity'].unique()
#print(uni)

n = df.head()
#print(n)
      
#check for duplicates
dupes = df[df.duplicated()]
#print(dupes)
#there are no duplicates (check this again later after converting to dummy and whatnot)

#check about missing values
missing = df.isnull().sum()
#print(missing)

#gather a list of quant columns
quant_list = ['ListingID','SellerRating', 'SellerRevCnt', 'SellerZip', 'VehListdays','VehMileage', 'VehYear','Dealer_Listing_Price']
#fill in missing values with the median
for item in quant_list:
    df[item] = df[item].fillna(df[item].median())

#find out which features are categorical
types = df.dtypes
#print(types)

#split off the target variables
n = len(df.columns)
X = df[df.columns[0:(n-2)]]
y_price = df[df.columns[n-1]]
y_trim = df[df.columns[n-2]]

#get dummies for all categorical variables

X = pd.get_dummies(X, columns = ['SellerCity','SellerListSrc','SellerName','SellerState','VehBodystyle','VehColorExt','VehColorInt','VehDriveTrain','VehEngine','VehFeats','VehFuel','VehHistory','VehMake','VehModel','VehPriceLabel','VehSellerNotes','VehType','VehTransmission'], drop_first = True, dummy_na = True)
#print(df)

#check to see that every object is now int
new_types = X.dtypes
#print(new_types)

#the first time I ran this there was still one object so the next few lines were to find out which one I missed. It is now fixed.
#print(df.info())
#print(df.select_dtypes('O').info())
#col_names = df.columns

#fnd out which columns are bool
#print(df.info())
#print(df.select_dtypes('bool').info())

X['SellerIsPriv'] = X['SellerIsPriv'].astype(int)
X['VehCertified'] = X['VehCertified'].astype(int)

#now check to see that all the data is in a good form
#print(X.info())
col_names = X.columns
#print(col_names)

#I think its all in a form i can operate on now!

#split into X and y_trim and y_price
#n = len(X.columns)
#X = X[X.columns[0:(n-2)]]
#y_price = df[df.columns[n-1]]
#y_trim = df[df.columns[n-2]]


#print("first y_price: ", y_price)
##APPLY FEATURE SCALING AND MEAN NORM

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#I will focus on price for now
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_price, test_size = .02)
#print("y_train: ", y_train)
#print(y_train[23])
#print("y_train: ", y_train)

#apply gradient descent to find a model
def gradient_des(X,y, alpha = .001, epochs = 30, lambd = 1000000 ):

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    X = np.concatenate((np.ones((m,1)), X), axis = 1)
    W = np.random.randn(n+1, )

    cost_hist = []

    #start the iteration
    for current_iteration in range(epochs):
        #print("W: ", W)
        y_estimated = X.dot(W)

        #print("y_est length: ", np.shape(y_estimated))

        #print("y_estimated: ", y_estimated)
        #print("y :", y)

        error = y_estimated - y
        #print("error: ", error)

        #print("est: ", y_estimated, "y :", y)
        
        #cost = (1/2*m) * np.sum(error**2)
        #ADDING REGULARIZATION TERM
        cost = ((1/2*m) * np.sum(error**2)) + (lambd/(2*m))* np.sum(W**2)

        #print("cost: ", cost)

        gradient = (1/m) * X.T.dot(error)
        
        #print("Gradient: ", gradient)

        W = W - alpha * gradient

        #print("weight: ", W)

        if current_iteration % 10 == 0:
            print("cost: " , cost , "iteration: " , current_iteration)

        cost_hist.append(cost)

    error = y_estimated- y
    mse_f = np.mean(error**2)
    mae_f = np.mean(abs(error))
    rmse_f = np.sqrt(mse_f)
    r2_f = 1-(sum(error**2)/sum((y-np.mean(y))**2))

    print("Training set Results by manual calculation:")
    print("MAE:",mae_f)
    print("MSE:", mse_f)
    print("RMSE:", rmse_f)
    print("R-Squared:", r2_f)

    r2 = metrics.r2_score(y,y_estimated)
    print("R-Squared:", r2)

    return W, cost_hist


#use the model to make a prediction and compare the the actual values
def compare_test(X_test, y_test, W):

    m_test = np.shape(X_test)[0]
    X_test = np.concatenate((np.ones((m_test,1)), X_test), axis = 1)

    y_estimated = X_test.dot(W)


    #print("est:", y_estimated)
    #print("actual: ", y_test)
    
    error = y_estimated- y_test
    mse_f = np.mean(error**2)
    mae_f = np.mean(abs(error))
    rmse_f = np.sqrt(mse_f)
    r2_f = 1-(sum(error**2)/sum((y_test-np.mean(y_test))**2))

    print("Results by manual calculation:")
    print("MAE:",mae_f)
    print("MSE:", mse_f)
    print("RMSE:", rmse_f)
    print("R-Squared:", r2_f)

    r2 = metrics.r2_score(y_test,y_estimated)
    print("R-Squared:", r2)

#do the mean squared error instead

    return error


def main():
    #make sure indexing looks okay, no weird extra index column


    weight, cost_hist = gradient_des(X_train, y_train, alpha = .1, epochs = 50)

    plt.plot(np.arange(len(cost_hist)), cost_hist)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function J")
    plt.title("Gradient Descent")
    plt.show()
    #print(cost_hist)
    test_error = compare_test(X_test, y_test, weight)
    #print("error: ", test_error)

#print(y_train)
if __name__ == '__main__':
    main()



##do to:
    # XXX split into test and train
    #if not, investigate what X and Y look like and if we need to remove labels


