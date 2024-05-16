from dataclasses import dataclass
from icecream import ic
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

from app.api.context.domains import DataSets
from app.api.context.models import Models

class TitanicModel(object):

    model = Models()
    dataset = DataSets()

    def preprocess(self, train_frame, test_frame):
                print(f'프로세스 시작')
                this = self.dataset # 파이썬에는 this 없음
                this.feature = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

                this.train = self.new_model(train_frame)
                this.test = self.new_model(test_frame)

                self.df_info(this)

                this.id = this.test['PassengerId']
                this.label = this.train['Survived']

                this = self.drop_feature_train(this, 'Survived')
                this = self.drop_feature(this, 'Ticket', 'Parch', 'Cabin', 'SibSp')

                # 아래는 template-method 패턴
                this = self.extract_title_from_name(this)

                # this = self.remove_duplicate_title(this)
                this = self.title_nominal(this)
                
                this = self.drop_feature(this, 'Name')

                self.df_info(this)

                this = self.sex_nominal(this)
                # this = self.drop_feature(this, 'Sex')

                this = self.embarked_nominal(this)
                this = self.age_ratio(this)
                # this = self.drop_feature(this, 'Age')

                this = self.pclass_ordinal(this)
                this = self.fare_ordinal(this)
                # this = self.drop_feature(this, 'Fare')

                self.df_info(this)

                k_fold = self.create_k_fold()
                accuracy = self.get_accuracy(this, k_fold)
                ic(accuracy)
                
                return this
    
    def df_info(self, this):
            print('='*50)
            print(f'1. Train 의 type 은 {type(this.train)} 이다.')
            print(f'2. Train 의 column 은 {this.train.columns} 이다.')
            print(f'3. Train 의 상위 1개의 데이터는 {this.train.head()} 이다.')
            print(f'4. Train 의 null 의 갯수는 {this.train.isnull().sum()} 이다.')
            print(f'5. Test 의 type 은 {type(this.test)} 이다.')
            print(f'6. Test 의 column 은 {this.test.columns} 이다.')
            print(f'7. Test 의 상위 1개의 데이터는 {this.test.head()} 이다.')
            print(f'8. Test 의 null 의 갯수는 {this.test.isnull().sum()} 이다.')
                
    def new_model(self, payload) -> object:
            this = self.model
            this.context = './app/api/titanic/data/' # main.py기준
            this.fname = payload
            return pd.read_csv(this.context + this.fname)
    
    @staticmethod
    def create_train(this) ->str:
            return this.train.drop('Survived', axis=1) # 0:행, 1:열
    
    @staticmethod
    def create_label(this) ->str:
            return this.train['Survived']
    
    # @staticmethod
    # def drop_feature(this, *feature) -> object:
    #         for i in [this.train, this.test]:
    #                 for j in feature:
    #                         i.drop(j, axis=1, inplace=True)

    #         [i.drop(j, axis=1, inplace=True) for j in feature for i in [this.train, this.test]]

    #         return this


    @staticmethod
    def drop_feature(this, *feature) -> pd.DataFrame:
            # for i, j in itertools.product(feature, [this.train, this.test]):
            #         j.drop(i, axis=1, inplace=True)
            [i.drop(j, axis=1, inplace=True) for j in feature for i in [this.train, this.test]]

            return this
    
    @staticmethod
    def drop_feature_train(this, *feature) -> object:
            [this.train.drop(j, axis=1, inplace=True) for j in feature]

            return this
            
    @staticmethod
    def pclass_ordinal(this) -> pd.DataFrame:
            #this.train['Pclass'] = this.train['Pclass'].map({1: 1, 2: 2, 3: 3})
            # this.train['Pclass'] = this.train['Pclass'].map({1: 1})
            return this

    
    @staticmethod
    def sex_nominal(this) -> pd.DataFrame:
            # gender_mapping = {'male': 0, 'female': 1}
            # for these in [this.train, this.test]:
            #         these['Gender'] = these['Sex'].map(gender_mapping)

            this.train['Sex'] = this.train['Sex'].map({'male': 1, 'female': 2})
            this.test['Sex'] = this.test['Sex'].map({'male': 1, 'female': 2})
            return this

    @staticmethod
    def parch_ratio(this) -> pd.DataFrame:
            return this
    
    @staticmethod
    def fare_ordinal(this) -> pd.DataFrame:
            return this

    @staticmethod
    def embarked_nominal(this) -> pd.DataFrame:
            this.train['Embarked'] = this.train['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})
            this.test['Embarked'] = this.test['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})
            return this

    @staticmethod
    def extract_title_from_name(this) -> pd.DataFrame:
            for these in [this.train, this.test]:
                    these['Title'] = these['Name'].str.extract('([A-Za-z]+)\.')
            #[]를 하나로 봐야함. [A-Za-z] = 대문자 or 소문자 // + 하나 이상 // {2,} 둘 이상, {2,4} 둘 이상 넷 미만 // 마지막에 .(.외의 기호는\ 빼고 쓰면됨)
            return this

    @staticmethod
    def remove_duplicate_title(this) -> pd.DataFrame:
            a = []
            for these in [this.train, this.test]:
                    a += list(set(these['Title']))
            a = list(set(a))
            print(a)
            '''
            ['Mr', 'Sir', 'Major', 'Don', 'Rev', 'Countess', 'Lady', 'Jonkheer', 'Dr',
            'Miss', 'Col', 'Ms', 'Dona', 'Mlle', 'Mme', 'Mrs', 'Master', 'Capt']
            Royal : ['Countess', 'Lady', 'Sir']
            Rare : ['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Mme' ]
            Mr : ['Mlle']
            Ms : ['Miss']
            Master
            Mrs
            '''


            return this
    
    @staticmethod
    def title_nominal(this) -> pd.DataFrame:

            for these in [this.train, this.test]:
                    these['Title'] = these['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
                    # 'Countess', 'Lady', 'Sir'에 해당하는 값을 'Royal'로 바꿈
                    these['Title'] = these['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Mme'], 'Rare')
                    these['Title'] = these['Title'].replace(['Mlle'], 'Mr')
                    these['Title'] = these['Title'].replace(['Miss'], 'Ms')
                    # Master 는 변화없음
                    # Mrs 는 변화없음
                    these['Title'] = these['Title'].fillna(0)
                    these['Title'] = these['Title'].map({'Mr': 1, 'Ms': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6})

            return this

    
            
    @staticmethod
    def age_ratio(this) -> pd.DataFrame:
            train = this.train
            test = this.test
            age_mapping = {'Unknown':0 , 'Baby': 1, 'Child': 2, 'Teenager' : 3, 'Student': 4,
                    'Young Adult': 5, 'Adult':6,  'Senior': 7}
            train['Age'] = train['Age'].fillna(-0.5)
            test['Age'] = test['Age'].fillna(-0.5) # 왜 NaN 값에 -0.5 를 할당할까요 ?
            bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf] # 이것을 이해해보세요
            labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

            for these in train, test:
                    # pd.cut()을 사용하시오.
                    these['Age'] = pd.cut(these['Age'], bins, labels=labels)
                    these['AgeGroup'] = these['Age'].map(age_mapping)
            return this
    
    @staticmethod
    def create_k_fold() -> object:
            return KFold(n_splits=10, shuffle=True, random_state=0)
    
    @staticmethod
    def learning(self, train_fname, test_fname) -> object:
            this = self.preprocess(train_fname, test_fname)
            print(f'학습 시작')
            k_fold = self.create_k_fold()
            accuracy = self.get_accuracy(this, k_fold)
            ic(f'사이킷런 알고리즘 정확도: {accuracy}')
            return accuracy

    @staticmethod
    def get_accuracy(this, k_fold) -> object:
            score = cross_val_score(RandomForestClassifier(), this.train, this.label,
                                    cv=k_fold, n_jobs=1, scoring='accuracy')
            return round(np.mean(score)*100, 2)