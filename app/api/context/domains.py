from dataclasses import dataclass

@dataclass
class DataSets:
    def _init_(self): #_ : 감춰진 파일명 = hidden, _init_ 생성자
        _context : str
        _fname : str
        _train : object
        _test : object
        _id : str
        _label : str
        _feature : object

        @property
        def context(self) -> str: return self._context

        @context.setter
        def context(self, context): self._context = context

        @property
        def fname(self) -> str : return self._frame

        @fname.setter
        def fname(self, fname) : self._fname = fname

        @property
        def train(self) -> str : return self._train

        @train.setter
        def train(self, train) : self._train = train

        @property
        def test(self) -> str : return self._test

        @test.setter
        def test(self, test) : self._test = test

        @property
        def id(self) -> str : return self._id

        @id.setter
        def id(self, id) : self._id = id

        @property
        def label(self) -> str : return self._label

        @label.setter
        def label(self, label) : self._label = label



        
        #pass # null의 의미 // 답을 label로 보통 적음

