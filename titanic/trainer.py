import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ctx = 'C:/Users/ezen/PycharmProjects/test1/titanic/data/'
train = pd.read_csv(ctx+'train.csv')
test = pd.read_csv(ctx+'test.csv')
#df = pd.DataFrame(train)
#print(df.columns)

"""
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
       
PassengerId 고객아이디
Survived 생존여부   Survived    0 = No, 1 = Yes
pclass 승선권 클래스  Ticket class    1 = 1st, 2 = 2nd, 3 = 3rd
Name 이름
Sex 성별   Sex    
Age 나이   Age in years    
Sibsp 동반한 형제자매, 배우자 수   # of siblings / spouses aboard the Titanic    
Parch 동반한 부모, 자식 수   # of parents / children aboard the Titanic    
Ticket 티켓 번호   Ticket number    
Fare 티켓의 요금   Passenger fare    
Cabin 객실번호   Cabin number    
Embarked 승선한 항구명   Port of Embarkation
  C = Cherbourg 쉐부로, Q = Queenstown 퀸스타운, S = Southampton 사우스햄톤
"""

# f, ax = plt.subplots(1, 2, figsize=(18, 8))
# train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
#
# ax[0].set_title('Survived')
# ax[0].set_ylabel('')
#
# sns.countplot('Survived', data=train, ax=ax[1])
# ax[1].set_title('Survived')
# plt.show()
# 생존률 38.4%, 사망률 61.6%

"""
데이터는 훈련데이터(train.csv), 목적데이터(test.csv) 두가지로 제공됩니다.
목적데이터에는 위 항목에서는 Survived 정보가 빠져있습니다.
그것은 답이기 때문입니다.
"""
############
# 성별
############
# f, ax = plt.subplots(1, 2, figsize=(18, 8))
# train['Survived'][train['Sex'] == 'male'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
# train['Survived'][train['Sex'] == 'female'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)
#
# ax[0].set_title('Survived(Male)')
# ax[1].set_title('Survived(Female)')

# plt.show()
# 남성 생존률 18.9% 사망률 81.1%
# 여성 생존률 74.2% 사망률 25.8%

############
# 승선권 Pclass
############
df_1 = [train['Sex'],train['Survived']]
df_2 = train['Pclass']
df = pd.crosstab(df_1, df_2, margins=True)
# print(df.head())
"""
Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891
"""

# f, ax = plt.subplots(2, 2, figsize=(20, 15))
# sns.countplot('Embarked', data=train, ax=ax[0,0])
# ax[0,0].set_title('No. Of Passengers Boarded')
# sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0,1])
# ax[0,1].set_title('Mail - Female Embarked')
# sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1,0])
# ax[1,0].set_title('Pclass vs Survived')
# sns.countplot('Pclass', data=train, ax=ax[1,1])
# ax[1,1].set_title('Embarked vs Pclass')

# plt.show()
"""
위 데이터를 보면 절반 이상의 승객이 ‘Southampton’에서 배를 탔으며, 여기에서 탑승한 승객의
 70% 가량이 남성이었습니다. 현재까지 검토한 내용으로는 남성의 사망률이 여성보다 훨씬 높았기에
  자연스럽게 ‘Southampton’에서 탑승한 승객의 사망률이 높게 나왔습니다.
또한 ‘Cherbourg’에서 탑승한 승객들은 1등 객실 승객의 비중 및 생존률이
 높은 것으로 보아서 이 동네는 부자동네라는 것을 예상할 수 있습니다.
"""

# 결측치 제거

# train.info()
"""
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
"""
# print(train.isnull().sum())
"""
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
"""

def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['survived', 'dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.show()

# bar_chart('Sex')
# bar_chart('Pclass')
# bar_chart('SibSp')
# bar_chart('Parch')
# bar_chart('Embarked')

# S, Q 에 탑승한 사람이 더 많이 사망했고 C 는 덜 사망했다
"""
지금까지 데이터를 분석한 결과 Pclass, Sex, Age, SibSp, Parch, Embarked 컬럼의 경우 실제로 Survived에 영향을 미치는 것을 알 수 있습니다. 이를 바탕으로 데이터를 Feature Engineering을 진행할 것입니다
"""
"""
Feature Engineering은 머신러닝 알고리즘을 작동하기 위해 데이터에 대해 특징을 만들어 내는 과정입니다. 간단히 정리하면 모델의 성능을 높이기 위해 모델에 입력할 데이터를 만들기 위해 주어진 초기 데이터로부터 특정을 가공하고 생서하는 전체 과정을 의미합니다.
"""
"""
위 정보에서 얻을 수 있는 사실은 아래와 같습니다.
1. Age의 약 20프의 데이터가 Null로 되어있다.
2. Cabin의 대부분 값은 Null이다.
3. Name, Sex, Ticket, Cabin, Embarked는 숫자가 아닌 문자 값이다.
   - 연관성 없는 데이터는 삭제하거나 숫자로 바꿀 예정입니다.
     (머신러닝은 숫자를 인식하기 때문입니다.)
그리고 이를 바탕으로 이렇게 데이터를 가공해 보겠습니다.
1. Cabin과 Ticket 두 값은 삭제한다.(값이 비어있고 연관성이 없다는 판단하에)
2. Embarked, Name, Sex 값은 숫자로 변경할 것 입니다.
3. Age의 Null 데이터를 채워 넣을 것입니다.
4. Age의 값의 범위를 줄일 것입니다.(큰 범위는 머신러닝 분석시 좋지 않습니다.)
5. Fare의 값도 범위를 줄일 것입니다.
지금부터 위와 같이 데이터 가공을 시작해 보도록 하겠습니다. 저의 방법이 정확하고 맞지는 않다고 봅니다. 여러 자료를 참고해서 처음 시도해보는 머신러닝 과제이기 때문에 배운다는 생각으로 정리하고 있다는 것을 미리 알아 두셨으면 좋겠습니다.
"""

# Cabin, Ticket 값 삭제
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
# print(train.head())
# print(test.head())
"""
PassengerId  Survived  Pclass    ...    Parch     Fare  Embarked
0            1         0       3    ...        0   7.2500         S
1            2         1       1    ...        0  71.2833         C
2            3         1       3    ...        0   7.9250         S
3            4         1       1    ...        0  53.1000         S
4            5         0       3    ...        0   8.0500         S
[5 rows x 10 columns]
"""
# Embarked 값 가공
"""
Embarked의 각각 값의 갯수(S, C, Q)부터 살펴보겠습니다. 앞서 Embarked는 2개의 값이 Null이라는 것을 알 수 있었습니다.
"""
s_city = train[train["Embarked"]=='S'].shape[0] #shape[0] -> 스칼라
# print("S :", s_city) # S : 644
c_city = train[train["Embarked"]=='C'].shape[0]
# print("C :", c_city) # C : 168
q_city = train[train["Embarked"]=='Q'].shape[0]
# print("Q :", q_city) # Q : 77

"""
대부분의 값이 S라는 것을 알 수 있습니다. 그렇기 때문에 비어있는 두 값도 S로 채워도 무방할 듯합니다.
"""
train = train.fillna({"Embarked":"S"})
"""
그 다음은 각 값(S, C, Q)을 숫자로 변경해 주겠습니다. 앞서 머신러닝은 숫자를 인식하고 문자는 인식하지 않는다고 말씀드렸었습니다. 아래와 같이 1, 2, 3으로 가공을 완료 하였습니다.
"""
city_mapping = {"S":1, "C":2, "Q":3}
train['Embarked'] = train['Embarked'].map(city_mapping)
test['Embarked'] = test['Embarked'].map(city_mapping)
# print(train.head())
# print(test.head())
'''
   PassengerId  Survived  Pclass    ...    Parch     Fare  Embarked
0            1         0       3    ...        0   7.2500         1
1            2         1       1    ...        0  71.2833         2
2            3         1       3    ...        0   7.9250         1
3            4         1       1    ...        0  53.1000         1
4            5         0       3    ...        0   8.0500         1
[5 rows x 10 columns]
'''

# Name 값 가공하기
'''
Name의 경우는 간단하게 문자열 파싱을 통해 가공해 보았습니다. 각각 이름은 남자와 여자 등을 뜻하는 Mr, Mrs가 있습니다. 이 부분을 가공해서 어떤 데이터가 있는지 보았습니다. 그리고 성별과 비교하여 Mr, MrS, Miss 데이터가 비슷한지 보았습니다.
'''
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
# print(pd.crosstab(train['Title'], train['Sex']))
'''
Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
'''
'''
대부분 데이터가 정확했고 이를 바탕으로 비슷한 글자는 바꾸고 최대한 줄여서 정리해 보겠습니다. 
Mr, Mrs, Miss, Royal, Rare, Master 6개로 줄어보았고 이를 바탕으로 각 생존률의 평균을 살펴 보았습니다. 
Royal은 모두가 생존했네요.
'''

for dataset in combine:
    dataset['Title'] \
        = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'], 'Rare')
    dataset['Title'] \
        = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
    dataset['Title'] \
        = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] \
        = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] \
        = dataset['Title'].replace('Mme','Mrs')
# print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())
'''
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4    Rare  0.250000
5   Royal  1.000000
'''
#이 데이터를 바탕으로 1부터 6까지로 매핑을 하여 숫자로 변경하였습니다. 아래에 map() 함수는 데이터 가공에서 가장 많이 쓰이는 것 같습니다.
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) # fillna
# print(train.head())
"""
   PassengerId  Survived  Pclass  ...       Fare Embarked  Title
0            1         0       3  ...     7.2500        1      1
1            2         1       1  ...    71.2833        2      3
2            3         1       3  ...     7.9250        1      2
3            4         1       1  ...    53.1000        1      3
4            5         0       3  ...     8.0500        1      1
[5 rows x 11 columns]
"""

# 이제 train 함수의 Name과 PassengerId 삭제하겠습니다.
train = train.drop(['Name','PassengerId'], axis = 1)
test = test.drop(['Name'], axis = 1)
combine = [train, test]
# print(train.head())
"""
  Survived  Pclass     Sex   Age  SibSp  Parch     Fare  Embarked  Title
0         0       3    male  22.0      1      0   7.2500         1      1
1         1       1  female  38.0      1      0  71.2833         2      3
2         1       3  female  26.0      0      0   7.9250         1      2
3         1       1  female  35.0      1      0  53.1000         1      3
4         0       3    male  35.0      0      0   8.0500         1      1
"""

"""
이와 동시에 Sex 값도 map()함수를 사용하여 숫자로 변경을 했습니다. 간단하게 male은 0, 그리고 female은 1로 변경했습니다.
"""
sex_mapping = {"male":0, "female":1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
# print(train.head())
"""
Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Embarked  Title
0         0       3    0  22.0      1      0   7.2500         1      1
1         1       1    1  38.0      1      0  71.2833         2      3
2         1       3    1  26.0      0      0   7.9250         1      2
3         1       1    1  35.0      1      0  53.1000         1      3
4         0       3    0  35.0      0      0   8.0500         1      1
"""

# Age 값 가공하기
"""
개인적으로 Age 값 부분이 제일 어려웠던거 같습니다. Null 값도 워낙 많았고 간단하게 평균값으로 넣어도 될까라는 것입니다.
먼저 Null 값을 -0.5 채워 넣은 후 pandas의 cut() 함수를 사용해서 AgeGroup을 만들어 보겠습니다.
cut() 함수는 각 구간의 값을 특정 값으로 정의해주는 함수입니다.
"""
train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels = labels)
# print(train.head())
"""
Survived  Pclass  Sex     ...       Embarked  Title     AgeGroup
0         0       3    0     ...              1      1      Student
1         1       1    1     ...              2      3        Adult
2         1       3    1     ...              1      2  Young Adult
3         1       1    1     ...              1      3  Young Adult
4         0       3    0     ...              1      1  Young Adult
[5 rows x 10 columns]
"""

# bar_chart('AgeGroup')
"""
위와 같이 만들었지만 Null 값의 Unknown의 대부분이 사망한 것으로 나왔고 많은 부분을 차지하고 있었습니다.
이 값을 앞서 Title에 따라 연령을 추측해서 넣어보겠습니다. 사실 이부분이 이해가 많이가지 않았는데
하나의 데이터 가공 방법 인거 같습니다.
"""
age_title_mapping = {1: "Young Adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult", 6:"Adult"}
# age_title_mapping = {0: 'Unknown', 1: "Young Adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult", 6:"Adult"}
for x in range(len(train['AgeGroup'])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]
for x in range(len(test['AgeGroup'])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]
# print(train.head())
"""
!!! KeyError: 'Mr'
   Survived  Pclass  Sex     ...       Embarked  Title     AgeGroup
0         0       3    0     ...              1      1      Student
1         1       1    1     ...              2      3        Adult
2         1       3    1     ...              1      2  Young Adult
3         1       1    1     ...              1      3  Young Adult
4         0       3    0     ...              1      1  Young Adult
[5 rows x 10 columns]
"""

#이제 AgeGroup을 숫자로 바꿔보겠습니다. 그 후 Age를 삭제하여 줍니다.
age_mapping = {'Baby' : 1, 'Child':2, 'Teenager':3, 'Student': 4, 'Young Adult':5,'Adult':6, 'Senior':7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
# print(train.head())

# 이제 어느덧 마지막 Fare만 남았네요. Fare는 간단하게 qcut 사용해 보겠습니다. 4개의 범위를 나눠서 1, 2, 3, 4로 바꾸었습니다.
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = {1,2,3,4})
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = {1,2,3,4})
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
# print(train.head())
"""
   Survived  Pclass  Sex  SibSp    ...     Embarked  Title  AgeGroup  FareBand
0         0       3    0      1    ...            1      1         4         1
1         1       1    1      1    ...            2      3         6         4
2         1       3    1      0    ...            1      2         5         2
3         1       1    1      1    ...            1      3         5         4
4         0       3    0      0    ...            1      1         5         2
[5 rows x 9 columns]
"""
"""
이렇게 모든 데이터 가공이 끝났습니다. 물론 잘한 데이터 가공은 아닙니다.
다른 분들이 한 자료를 보면 SibSp, Parch 컬럼도 활용하여 데이터를 만들어 내는 것을 봤지만 아직 제 실력으론
부족하기 때문에 여기까지 데이터 가공 즉 Feature Engieneering을 마무리 해보겠습니다.
이제 적절한 머신러닝 모델을 선택하여 예측만 하면 마무리됩니다.
쉽지 않은 과정이였고 처음하다보니 많이 부족한 자료라고 생각합니다.
머신러닝 공부를 하면서 뭔가 한가지 프로젝트를 진행해 보고 싶었고 하나를 하면 더 공부하고자 하는
의욕이 생기지 않을까라는 생각이 들었습니다.
그래서 타이타닉 프로젝트를 진행하였고 어느덧 중반을 지나가고 있습니다.
"""

# *****
# 데이터 모델링
# *****

"""
앞서 Data Feature Engineering을 통해 데이터 전처리를 진행하였습니다. 데이터 전처리 과정을 통해 데이터를 
컴퓨터가 이해하기 쉽도록 만들었습니다.
이제 가장 높은 정확도를 보이는 모델을 찾은 다음 Test Set의 Survived 즉 생존 여부를 예측하는 과정을 진행하겠습니다. 
먼저 Train Set을 X와 Y값로 나누어 줍니다. 아래에서 Train_data가 X이고 target이 Y라고 볼 수 있습니다
"""
train_data = train.drop('Survived', axis = 1)
target = train['Survived']
# print(train_data.shape)
# print(target.shape)
"""
((891, 8), (891,))
"""
"""
그런 다음 Train 데이터의 정보 즉 info() 함수를 보면 아래와 같이 null 값이 없는 것을 알 수 있습니다.
"""
# print(train.info)
"""
<bound method DataFrame.info of      Survived  Pclass  Sex  SibSp    ...     Embarked  Title  AgeGroup  FareBand
0           0       3    0      1    ...            1      1         4         1
1           1       1    1      1    ...            2      3         6         4
2           1       3    1      0    ...            1      2         5         2
3           1       1    1      1    ...            1      3         5         4
--- 이하 생략 ---
"""