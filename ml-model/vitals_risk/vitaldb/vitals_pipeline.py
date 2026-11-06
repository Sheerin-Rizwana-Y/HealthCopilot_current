"""Phase 2 research prototype. Labels are heuristic, NOT clinical ground truth."""
import argparse, gzip, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import joblib

SIGNALS = ('hr','spo2','temperature')
LIMITS = {'hr':(30,220),'spo2':(50,100),'temperature':(30,43)}
FEATURES = [f'{s}_{stat}' for s in SIGNALS for stat in ('median','min','max','std','slope','valid_fraction')]

def read_track(path):
    with open(path,'rb') as f: compressed=f.read(2)==b'\x1f\x8b'
    df=pd.read_csv(path,compression='gzip' if compressed else None)
    if df.shape[1]<2: raise ValueError(f'{path}: expected time and value columns')
    # VitalDB track files commonly have Time, Value columns.
    names={str(c).strip().lower():c for c in df.columns}
    tcol=next((names[k] for k in ('time','timestamp','t','seconds') if k in names),df.columns[0])
    vcol=next((names[k] for k in ('value','val','v') if k in names and names[k]!=tcol),next(c for c in df.columns if c!=tcol))
    x=pd.DataFrame({'time':pd.to_numeric(df[tcol],errors='coerce'),'value':pd.to_numeric(df[vcol],errors='coerce')}).dropna(subset=['time'])
    return x.sort_values('time').drop_duplicates('time',keep='last')

def align_case(files, step=5, max_gap=15):
    tracks={}
    for s,p in files.items():
        df=read_track(p); low,high=LIMITS[s]
        df.loc[~df.value.between(low,high),'value']=np.nan
        tracks[s]=df
    start=max(d.time.min() for d in tracks.values()); end=min(d.time.max() for d in tracks.values())
    if not np.isfinite(start+end) or end-start<step: return pd.DataFrame()
    times=np.arange(np.ceil(start/step)*step,end+0.00001,step)
    aligned=pd.DataFrame({'time':times})
    for s,df in tracks.items():
        valid=df.dropna(subset=['value']).rename(columns={'time':'source_time','value':s})
        aligned=pd.merge_asof(aligned.sort_values('time'),valid.sort_values('source_time'),left_on='time',right_on='source_time',direction='nearest',tolerance=max_gap)
        aligned=aligned.drop(columns='source_time')
    return aligned

def make_windows(df,caseid,window=60,step=30,min_valid=.6):
    rows=[]
    if df.empty:return pd.DataFrame()
    for start in np.arange(df.time.min(),df.time.max()-window+0.0001,step):
        part=df[(df.time>=start)&(df.time<start+window)]
        if len(part)<2:continue
        row={'caseid':caseid,'start_sec':float(start)}
        for s in SIGNALS:
            a=part[s].dropna().to_numpy(dtype=float)
            row[f'{s}_valid_fraction']=float(len(a)/len(part))
            row[f'{s}_median']=float(np.median(a)) if len(a) else np.nan
            row[f'{s}_min']=float(np.min(a)) if len(a) else np.nan
            row[f'{s}_max']=float(np.max(a)) if len(a) else np.nan
            row[f'{s}_std']=float(np.std(a)) if len(a) else np.nan
            row[f'{s}_slope']=float((a[-1]-a[0])/(max(len(a)-1,1))) if len(a) else np.nan
        row['quality_ok']=all(row[f'{s}_valid_fraction']>=min_valid for s in SIGNALS)
        rows.append(row)
    return pd.DataFrame(rows)

def heuristic_label(row):
    # Demonstration thresholds only; NOT validated clinical diagnoses.
    if not row.quality_ok:return 'insufficient_data'
    hr,spo2,temp=(row[f'{s}_median'] for s in SIGNALS)
    if pd.isna(hr) or pd.isna(spo2) or pd.isna(temp):return 'insufficient_data'
    if spo2<90 or hr<45 or hr>130 or temp<35 or temp>=39:return 'high'
    if spo2<94 or hr<55 or hr>110 or temp<36 or temp>=38:return 'moderate'
    return 'low'

def build(raw,output,window=60,step=30):
    output.mkdir(parents=True,exist_ok=True)
    groups={}
    for p in raw.glob('case_*_*.csv*'):
        m=re.fullmatch(r'case_(\d+)_(hr|spo2|temperature)\.csv(?:\.gz)?',p.name)
        if m:groups.setdefault(int(m[1]),{})[m[2]]=p
    frames=[]; quality=[]
    for caseid,files in sorted(groups.items()):
        if set(files)!=set(SIGNALS):
            quality.append({'caseid':caseid,'status':'missing_tracks','tracks':','.join(files) });continue
        try:
            aligned=align_case(files); w=make_windows(aligned,caseid,window,step)
            quality.append({'caseid':caseid,'status':'ok' if len(w) else 'no_windows','aligned_rows':len(aligned),'windows':len(w),'valid_windows':int(w.quality_ok.sum()) if len(w) else 0})
            if len(w):frames.append(w)
        except Exception as e:quality.append({'caseid':caseid,'status':'error','detail':str(e)})
    pd.DataFrame(quality).to_csv(output/'quality_report.csv',index=False)
    if not frames:raise ValueError('No usable windows. See quality_report.csv')
    data=pd.concat(frames,ignore_index=True);data['risk_label']=data.apply(heuristic_label,axis=1)
    data.to_csv(output/'windows.csv',index=False)
    return data

def train(data,output):
    usable=data[data.risk_label!='insufficient_data'].copy()
    counts=usable.groupby('risk_label').size().to_dict()
    groups=usable.caseid.nunique()
    if groups<3:raise ValueError(f'Need >=3 distinct cases for grouped holdout; found {groups}. Features exported; training stopped to avoid leakage.')
    if usable.risk_label.nunique()<2:raise ValueError('Need >=2 risk classes; inspect label distribution and obtain more cases.')
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42)
    tr,te=next(split.split(usable,usable.risk_label,groups=usable.caseid))
    a,b=usable.iloc[tr],usable.iloc[te]
    if a.risk_label.nunique()<2:raise ValueError('Training cases contain fewer than two classes. Download more cases.')
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=False)),('rf',RandomForestClassifier(n_estimators=200,random_state=42,class_weight='balanced_subsample',min_samples_leaf=2,n_jobs=-1))])
    model.fit(a[FEATURES],a.risk_label)
    pred=model.predict(b[FEATURES]); report=classification_report(b.risk_label,pred,output_dict=True,zero_division=0)
    report['train_cases']=sorted(map(int,a.caseid.unique()));report['test_cases']=sorted(map(int,b.caseid.unique()))
    report['train_labels']=a.risk_label.value_counts().to_dict();report['test_labels']=b.risk_label.value_counts().to_dict()
    report['confusion_labels']=list(model.classes_);report['confusion_matrix']=confusion_matrix(b.risk_label,pred,labels=model.classes_).tolist()
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    joblib.dump({'model':model,'features':FEATURES,'labels':list(model.classes_),'version':'vitals-v1-heuristic','warning':'NOT clinically validated'},output/'vitals_v1.joblib')
    return report

def predict(model_path,windows_path,output_path):
    artifact=joblib.load(model_path) # Load ONLY trusted locally-created joblib files.
    df=pd.read_csv(windows_path);out=df[['caseid','start_sec']].copy()
    valid=df[FEATURES].notna().any(axis=1) & df.quality_ok.astype(bool)
    out['prediction']='insufficient_data'
    if valid.any():
        out.loc[valid,'prediction']=artifact['model'].predict(df.loc[valid,artifact['features']])
        probabilities=artifact['model'].predict_proba(df.loc[valid,artifact['features']])
        for j,label in enumerate(artifact['model'].classes_):out.loc[valid,'prob_'+label]=probabilities[:,j]
    out.to_csv(output_path,index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','all','predict']);p.add_argument('--raw',type=Path,default=Path('datasets/vitals/raw'));p.add_argument('--out',type=Path,default=Path('datasets/vitals/processed'));p.add_argument('--window',type=int,default=60);p.add_argument('--step',type=int,default=30);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.command in ('prepare','all'):data=build(a.raw,a.out,a.window,a.step);print('Window labels:',data.risk_label.value_counts().to_dict())
    if a.command in ('train','all'):
        data=pd.read_csv(a.out/'windows.csv');print(json.dumps(train(data,a.out),indent=2))
    if a.command=='predict':predict(a.out/'vitals_v1.joblib',a.out/'windows.csv',a.out/'predictions.csv');print('Saved predictions.csv')
if __name__=='__main__':main()
