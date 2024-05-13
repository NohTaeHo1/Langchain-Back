from dataclasses import dataclass

@dataclass
class TitanicModel:
    def _init_(self): #_ : 감춰진 파일명 = hidden, _init_ 생성자
        self.train = None
        self.test = None
        self.context = None
        self.fname = ''
        self.id = 0
        self.label = 0
        #pass # null의 의미 // 답을 label로 보통 적음

