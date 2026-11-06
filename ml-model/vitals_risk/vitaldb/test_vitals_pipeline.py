import numpy as np
import pandas as pd
from vitals_pipeline import make_windows,heuristic_label,align_case,read_track

def test_window_and_label():
    t=np.arange(0,121,5);df=pd.DataFrame({'time':t,'hr':75.,'spo2':98.,'temperature':37.})
    w=make_windows(df,1)
    assert len(w)>0
    assert heuristic_label(w.iloc[0])=='low'

def test_low_oxygen():
    t=np.arange(0,121,5);df=pd.DataFrame({'time':t,'hr':75.,'spo2':85.,'temperature':37.})
    assert heuristic_label(make_windows(df,1).iloc[0])=='high'

def test_missing_signal():
    t=np.arange(0,121,5);df=pd.DataFrame({'time':t,'hr':75.,'spo2':np.nan,'temperature':37.})
    assert heuristic_label(make_windows(df,1).iloc[0])=='insufficient_data'
