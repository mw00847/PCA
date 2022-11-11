import pandas
from sklearn.decomposition import PCA
import numpy
import matplotlib.pyplot as plot


#df is the wavenumbers
#naming the variables and standards into pandas dataframes

df=pd.read_csv('standard1.csv',usecols=[0])

df1=pd.read_csv('standard2.csv',usecols=[1])
df2=pd.read_csv('25%.csv',usecols=[1])
df3=pd.read_csv('50%.csv',usecols=[1])
df4=pd.read_csv('75%.csv',usecols=[1])
df5=pd.read_csv('variable1.csv',usecols=[1])
df6=pd.read_csv('variable2.csv',usecols=[1])
df7=pd.read_csv('a.csv',usecols=[1])
df8=pd.read_csv('b.csv',usecols=[1])
df9=pd.read_csv('c.csv',usecols=[1])
df10=pd.read_csv('d.csv',usecols=[1])
df11=pd.read_csv('e.csv',usecols=[1])
df12=pd.read_csv('f.csv',usecols=[1])
df13=pd.read_csv('g.csv',usecols=[1])



#concatenation of the variables and standards into one dataframe

df=pd.concat([df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13],axis=1)
df.columns=['2','3','4','5','6','7','8','9','10','11','12','13']


#normalize the data before PCA
df_normalized=(df - df.mean()) / df.std()



#setting the number of components for PCA
pca = PCA(n_components=df.shape[1])
pca.fit(df_normalized)




# Reformat and view results
loadings = pd.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(df_normalized.columns))],
index=df.columns)
print(loadings)

#plot the variance
plot.plot(pca.explained_variance_ratio_)
plot.ylabel('Explained Variance')
plot.xlabel('Components')
plot.show()


#compare the principal components
plt.scatter(loadings.PC0,loadings.PC1)
plt.show()

#how to label the scatter plot as each original csv

