import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# قراءة بيانات الطلاب
df_Student = pd.read_csv("Student Mental health.csv")

# عرض البيانات الأولية
print(df_Student.head())
print(df_Student.info())
print(df_Student.isna().sum())
print(df_Student.Age.value_counts())
print(df_Student.Timestamp.value_counts())
print(df_Student.describe())

# التعامل مع القيم المفقودة
# بالنسبة للأعمدة الرقمية، قم بملء القيم المفقودة بالوسيط
for column in ['Age']:
    median_val = df_Student[column].median()
    df_Student[column].fillna(median_val, inplace=True)

# إزالة الصفوف المكررة
print(df_Student.duplicated().sum())
df_Student.drop_duplicates(inplace=True)

# رسم توزيع البيانات
sns.distplot(df_Student['Age'])
plt.show()

sns.set_theme(style="darkgrid")
sns.countplot(y="Age", data=df_Student, palette="flare")
plt.ylabel('Age num')
plt.xlabel('Total')
plt.show()

sns.set_theme(style="darkgrid")
sns.countplot(x="Timestamp", data=df_Student, palette="rocket")
plt.xlabel('Time (Y=year, D=data)')
plt.ylabel('Total')
plt.show()

# رسم توزيع الأعمار حسب الوقت
pd.crosstab(df_Student.Age, df_Student.Timestamp).plot(kind="bar", figsize=(12, 5), color=['#003f5c', '#ffa600', '#58508d', '#bc5090', '#ff6361'])
plt.title('Age distribution based on time')
plt.xlabel('Age')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()

# تجهيز البيانات لتدريب النموذج
# تحويل القيم الفئوية إلى أرقام
le = LabelEncoder()
for column in df_Student.select_dtypes(include=[object]).columns:
    df_Student[column] = le.fit_transform(df_Student[column])

# تحديد المتغيرات المستقلة والتابعة
X = df_Student.drop(["Age"], axis=1)
y = df_Student["Age"]

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df_Student.to_csv("Clean_data.csv",index=False)
df_Student = pd.read_csv('Clean_data.csv')
# حساب مصفوفة الارتباط وعرضها
corr_matrix = df_Student.corr()
print(corr_matrix)

# رسم مصفوفة الارتباط
plt.figure(figsize=(6, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title('Correlation Heatmap for Specific Columns')
plt.show()
# تدريب نموذج الانحدار اللوجستي
LRclassifier = LogisticRegression(solver='liblinear', max_iter=300)
LRclassifier.fit(X_train, y_train)
y_pred = LRclassifier.predict(X_test)
LRAcc = accuracy_score(y_test, y_pred)
print('Logistic Regression accuracy is: {:.2f}%'.format(LRAcc * 100))

# عرض مصفوفة الالتباس
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# حفظ النموذج
model_filename = 'model_of_data.joblib'
joblib.dump(LRclassifier, model_filename)
print(f"Model saved as {model_filename}")

# تدريب نموذج الجيران الأقرب
KNclassifier = KNeighborsClassifier(n_neighbors=3)
KNclassifier.fit(X_train, y_train)
y_pred = KNclassifier.predict(X_test)
KNAcc = accuracy_score(y_test, y_pred)
print('K Neighbors accuracy is: {:.2f}%'.format(KNAcc * 100))

# إيجاد أفضل قيمة لـ K
scoreListknn = []
for i in range(1, 50):
    KNclassifier = KNeighborsClassifier(n_neighbors=i)
    KNclassifier.fit(X_train, y_train)
    scoreListknn.append(KNclassifier.score(X_test, y_test))
    
plt.plot(range(1, 50), scoreListknn)
plt.xticks(np.arange(1, 50, 1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

KNAccMax = max(scoreListknn)
print("KNN Acc Max {:.2f}%".format(KNAccMax * 100))

# تدريب نموذج الـ SVM
SVCclassifier = SVC(kernel='linear', max_iter=50)
SVCclassifier.fit(X_train, y_train)
y_pred = SVCclassifier.predict(X_test)
SVCAcc = accuracy_score(y_test, y_pred)
print('SVC accuracy is: {:.2f}%'.format(SVCAcc * 100))

# تدريب نموذج شجرة القرار
DTclassifier = DecisionTreeClassifier(max_leaf_nodes=7)
DTclassifier.fit(X_train, y_train)
y_pred = DTclassifier.predict(X_test)
DTAcc = accuracy_score(y_test, y_pred)
print('Decision Tree accuracy is: {:.2f}%'.format(DTAcc * 100))

# إيجاد أفضل عدد للأوراق
scoreListDT = []
for i in range(2, 50):
    DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
    DTclassifier.fit(X_train, y_train)
    scoreListDT.append(DTclassifier.score(X_test, y_test))
    
plt.plot(range(2, 50), scoreListDT)
plt.xticks(np.arange(2, 50, 5))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.show()

DTAccMax = max(scoreListDT)
print("DT Acc Max {:.2f}%".format(DTAccMax * 100))

# تدريب نموذج Random Forest
RFclassifier = RandomForestClassifier(max_leaf_nodes=5)
RFclassifier.fit(X_train, y_train)
y_pred = RFclassifier.predict(X_test)
RFAcc = accuracy_score(y_test, y_pred)
print('Random Forest accuracy is: {:.2f}%'.format(RFAcc * 100))

# إنشاء DataFrame للمقارنة
compare = pd.DataFrame({
    'Model': ['Logistic Regression', 'K Neighbors', 'K Neighbors Max', 'SVM', 'Decision Tree', 'Decision Tree Max', 'Random Forest'],
    'Accuracy': [LRAcc * 100, KNAcc * 100, KNAccMax * 100, SVCAcc * 100, DTAcc * 100, DTAccMax * 100, RFAcc * 100]
})

# عرض النتائج بعد الترتيب
print(compare.sort_values(by='Accuracy', ascending=False))



