import itertools
from app.api.titanic.model.titanic_model import TitanicModel
import pandas as pd


class TitanicService:

        model = TitanicModel()

        def process(self):
                print(f'프로세스 시작')
                this = self.model # 파이썬에는 this 없음
                this.train = self.new_model('train.csv')
                this.test = self.new_model('test.csv')
                self.df_info(this)

                this.id = this.test['PassengerId']
                
                this = self.drop_feature(this, 'Ticket', 'Name')

                self.df_info(this)

                
                this = self.create_train(this)
                
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
        def drop_feature(this, *feature) -> object:
                # for i, j in itertools.product(feature, [this.train, this.test]):
                #         j.drop(i, axis=1, inplace=True)

                [i.drop(j, axis=1, inplace=True) for j in feature for i in [this.train, this.test]]

                return this
        
        @staticmethod
        def df_info(this):
                [print(i.head(3)) for i in [this.train, this.test]]
                # [print(f'{i.info()}') for i in  [this.train, this.test]]

