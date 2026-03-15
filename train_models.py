import pandas as pd

df = pd.read_csv("Delivery_Workers_Wage_Strike_Dataset.csv")

print(df.head())

print("\nDataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMean values")
print(df.mean(numeric_only=True))

print("\nVariance")
print(df.var(numeric_only=True))


###Data Preprocessing###
print("\nData Preprocessing")
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["City"] = le.fit_transform(df["City"])
df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.day

df = df.drop(columns=["WorkerID","Date"])


###Feature and Target Split###
print("\nFeature and Target Split")
X = df.drop("NetIncome", axis=1)
y = df["NetIncome"]


###Train-Test Split###
print("\nTrain-Test Split")
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


###Regression Analysis###
print("\nRegression Analysis")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

reg = LinearRegression()

reg.fit(X_train,y_train)

pred = reg.predict(X_test)

print("Regression Score:", r2_score(y_test,pred))



###PCA (Dimensionality Reduction)###
print("\nPCA (Dimensionality Reduction)")
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X)

print("Explained Variance:",pca.explained_variance_ratio_)



###MDS###
print("\nMDS (Multidimensional Scaling)")
from sklearn.manifold import MDS

mds = MDS(n_components=2)

X_mds = mds.fit_transform(X)


###Linear Discriminant###
print("\nLinear Discriminant Analysis")
df["ProfitClass"] = (df["NetIncome"]>0).astype(int)

X = df.drop(["NetIncome","ProfitClass"],axis=1)
y = df["ProfitClass"]

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

lda.fit(X_train,y_train)



###Logistic Discrimination###
print("\nLogistic Discrimination")
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(X_train,y_train)



###Kernel Machine (SVM)###
print("\nKernel Machine (SVM)")
from sklearn.svm import SVC

svm = SVC(kernel="rbf")

svm.fit(X_train,y_train)



###MLP Classifier###
print("\nMLP Classifier")
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,50),max_iter=500)

mlp.fit(X_train,y_train)



###Ensemble Learning###
print("\nEnsemble Learning")
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)



###K-Means Clustering###
print("\nK-Means Clustering")
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

df["Cluster"] = kmeans.fit_predict(X)



###Hierarchical Clustering###
print("\nHierarchical Clustering")
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=3)

df["HC_Cluster"] = hc.fit_predict(X)



###Model Comparison###
print("\nModel Comparison")
from sklearn.metrics import accuracy_score

models = {
"Logistic":log,
"SVM":svm,
"MLP":mlp,
"RandomForest":rf
}

for name,model in models.items():

    pred = model.predict(X_test)

    print(name,accuracy_score(y_test,pred))