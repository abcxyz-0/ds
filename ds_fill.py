import os

# String holders for code
Data_Wrangling_1 = """
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
temp = sns.load_dataset('titanic')
data = pd.DataFrame(temp)
data
data.isna().sum()
data.describe
data=data.drop(['deck','embark_town','embarked'] , axis= 1)
median = data['age'].median()
data['age'].fillna(median , inplace = True )

data_format = data['alone'].astype(int)
data_format
data_normal = pd.get_dummies(data['alive'], drop_first= True) # drop_first = after converting columns we drop the first column
data_normal
data_new = pd.concat([data,data_format,data_normal],axis=1)
data_new= data_new.drop(['alive','alone'],axis=1)
data_new
"""
Data_Wrangling_2 ="""
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import plotly.express as px
import seaborn as sns
temp = pd.read_csv('exam.csv')
data = pd.DataFrame(temp)
data
data.isna().sum()
data.corr()
data.describe()
sns.boxplot(data) 
data['Discussion'].describe()

lower , upper = data['Discussion'].quantile([0.05,0.90])
outliers = data[(data['Discussion'] > upper) | (data['Discussion']<lower)]
outliers.shape
outliers_remove = data[(data['Discussion'] < upper) & (data['Discussion'] > lower)]
outliers_remove.shape

sns.scatterplot(data=data, x="raisedhands", y="Discussion")
plt.show() # scale not done

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data["raisedhands"] = scaler.fit_transform(data[["raisedhands"]])
sns.scatterplot(data=data, x="raisedhands", y="Discussion")
plt.show()  #scale is done 01
"""
descriptive_stat = """ 
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
temp = sns.load_dataset('iris')
iris_data  = pd.DataFrame(temp)
iris_data
iris_data.describe()
print('UNIQUE Species := ', iris_data['species'].unique())
fig = px.box(iris_data , x='sepal_length',y='sepal_width')
fig.show()
group_data = pd.get_dummies(iris_data['species'],drop_first=True) # Normalization of data
group_data
iris_data = pd.concat([iris_data,group_data],axis=1)
iris_data

summary_stats = iris_data.groupby('species').agg({
    'sepal_length': ['mean', 'median', 'min', 'max', 'std'],
    'sepal_width': ['mean', 'median', 'min', 'max', 'std'],
    'petal_length': ['mean', 'median', 'min', 'max', 'std'],
    'petal_width': ['mean', 'median', 'min', 'max','std']
})
print(summary_stats)

x = px.scatter(iris_data, x="sepal_length", y="sepal_width", color="species")
x.show()
"""
linear_reg_boston_DA_I = """ 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston
import pandas as pd  
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
boston_dataset = load_boston()
boston_dataset.keys()
boston_dataset
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()
px.density_heatmap(boston.corr())
X = boston.drop('TAX',axis=1)
Y = boston['TAX']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

y_train_predict = lin_model.predict(X_train)
#for train data
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

#  for testing set
y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

"""

logistic_reg_DA_II = """ 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix , accuracy_score  , precision_score,recall_score , f1_score , classification_report

temp = pd.read_csv("Social_Network_Ads.csv")
data = pd.DataFrame(temp)
data
data = data.drop('Gender', axis=1)

X = data.drop('Purchased',axis=1)
y = data['Purchased']

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size= 0.2)
logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(x_train,y_train)

#step --> predict train data
y_pred_train = logistic_model.predict(x_train)
#step --> predict test data
y_pred_test = logistic_model.predict(x_test)

confusion_matrix(y_test,y_pred_test)

print(classification_report(y_test,y_pred_test))

accuracy = accuracy_score(y_test, y_pred_test)
error_rate  = 1 -accuracy
print("accuracy := ",accuracy,"----"," error rate := ",error_rate)
"""

naviebayes_classifi_DA_III   = """ 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix ,accuracy_score, classification_report
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import plotly.express as px
temp = sns.load_dataset('iris')
data = pd.DataFrame(temp)
data
data.isna().sum()
X = data.drop('species',axis=1)
y = data['species']
x_train , x_test , y_train ,y_test = train_test_split(X, y , random_state= 8 , test_size= 0.2)
bayes_class = GaussianNB()
bayes_class.fit(x_train,y_train)

#Step := for train data prediction
y_pred_train = bayes_class.predict(x_train)
#Step := for test data prediction
y_pred_test = bayes_class.predict(x_test)

print(classification_report(y_test,y_pred_test))
accuracy = accuracy_score(y_test , y_pred_test)
error = 1-accuracy
print('accuracy := ',accuracy," --- error := ",error)
c_matrix = confusion_matrix(y_test,y_pred_test)
print(c_matrix)
px.density_heatmap(c_matrix)

"""

text_analysis = """ 
import nltk
document = "This is an feet of an example document for tokenization. , This is an example document for POS tagging ,stemming"
#tokenization
from nltk.tokenize import word_tokenize
token = word_tokenize(document)
token
#POS tag
from nltk.tag import pos_tag
pos_tagging = pos_tag(token)
pos_tagging
#stop words removals
from nltk.corpus import  stopwords
stopword_removal = set(stopwords.words('english'))
filtered_tokens = []
for i in token:
    if i.lower() not in stopword_removal:
        filtered_tokens.append(i)
filtered_tokens
#steamming and lemmatization

"""

data_visualization_I= """ 
import numpy as np 
import pandas as pd 
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
temp = sns.load_dataset('titanic')
data = pd.DataFrame(temp)
data
sns.barplot(x='sex',y='age',data=data)
px.histogram(data,x='sex',y='survived')
px.histogram(data,y='fare',x='class')

"""

data_visualization_II = """ 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns
temp = sns.load_dataset('titanic')
data = pd.DataFrame(temp)
data
px.box(data,y='age',x='sex')
sns.catplot(x='sex',hue='survived',kind='count',data=data)

# Statistics for 'sex' column
sex_stats = data['sex'].value_counts()
# Statistics for 'age' column
age_stats = data['age'].describe()
print("Observations for 'sex'")
print(sex_stats)
print("Observations for 'age':")
print(age_stats)
"""

data_visualization_III = """ 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px
import pandas as pd 
temp = sns.load_dataset('iris')
data = pd.DataFrame(temp)
data.dtypes
sns.histplot(data)
px.histogram(data , y=['sepal_length','petal_length'],x='species')
px.histogram(data , y=['petal_width','sepal_width'],x='species')
px.box(data,y=['petal_width','sepal_width'],x='species')
px.box(data, y=['petal_width','sepal_width'],x='species')

data.hist()
data.boxplot()
q1 = data['sepal_length'].quantile(0.25)
q3 = data['sepal_length'].quantile(0.75)
iqr = q3 -q1 

upper_bound = q1 - (1.5 * iqr) 
lower_bound = q3 + (1.5 * iqr)
print(upper_bound , " ----- ", lower_bound)

data.describe()
"""

scala = """ 
object Hello { 
def main(args: Array[String]) = { 
println("Hello, world") 
} 
} 
"""


masterDict = {
    'Data_Wrangling_1' : Data_Wrangling_1,
    'Data_Wrangling_2': Data_Wrangling_2,
    'descriptive_stat': descriptive_stat,
    'linear_reg_boston_DA_I': linear_reg_boston_DA_I,
    'logistic_reg_DA_II': logistic_reg_DA_II,
    'naviebayes_classifi_DA_III': naviebayes_classifi_DA_III,
    'text_analysis':text_analysis,
    'data_visualization_I':data_visualization_I,
    'data_visualization_II':data_visualization_II,
    'data_visualization_III':data_visualization_III,
    'scala':scala
}

class Writer:
    def __init__(self, filename):
        self.filename = os.path.join(os.getcwd(), filename)
        self.masterDict = masterDict
        self.questions = list(masterDict.keys())

    def getCode(self, input_string):
        input_string = self.masterDict[input_string]
        with open(self.filename, 'w') as file:
            file.write(input_string)
        print(f'##############################################')

if __name__ == '__main__':
    write = Writer('output.txt')
    # print(write.questions)
    write.getCode('descision_region_perceptron')
