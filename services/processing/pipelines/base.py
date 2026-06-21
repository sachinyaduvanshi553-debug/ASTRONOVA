from abc import ABC, abstractmethod
import pandas as pd

class BasePipeline(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'BasePipeline':
        pass
        
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
