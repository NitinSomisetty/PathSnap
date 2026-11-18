from pathlib import Path
import logging, os
from functools import lru_cache
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBClassifier
from pipeline_updated import Config, FEATURE_NAMES, RoadIndex, align_reference, build_candidate_features, clean_track, read_graph, read_reference
from route_decoder import decode

ROOT = Path(__file__).resolve().parent
DEFAULT_EVALUATION_ROOT = ROOT / 'evaluation_trips'
DEFAULT_DEVELOPMENT_ROOT = ROOT / 'Student_folder' / 'Student folder' / 'student_test'
DATA_ROOT = Path(os.environ['PATHSNAP_EVALUATION_DIR']).resolve() if os.getenv('PATHSNAP_EVALUATION_DIR') else (DEFAULT_EVALUATION_ROOT if DEFAULT_EVALUATION_ROOT.is_dir() else DEFAULT_DEVELOPMENT_ROOT).resolve()
MODEL_PATH = Path(os.getenv('PATHSNAP_MODEL_PATH', ROOT / 'artifacts' / 'xgboost_model.json'))
app = FastAPI(title='PathSnap')
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('PATHSNAP_CORS_ORIGINS', '*').split(','), allow_methods=['GET','POST'], allow_headers=['*'])

def trips():
    if not DATA_ROOT.is_dir(): return {}
    return {p.name:p for p in DATA_ROOT.iterdir() if p.is_dir() and all((p / f'{p.name}.{e}').is_file() for e in ('track','nodes','arcs'))}

@lru_cache(maxsize=1)
def classifier():
    loaded = XGBClassifier()
    loaded.load_model(str(MODEL_PATH))
    if loaded.n_features_in_ != len(FEATURE_NAMES): raise ValueError('Model feature count mismatch')
    return loaded

@app.get('/health')
def health():
    return {'status':'ready' if MODEL_PATH.is_file() and len(trips()) > 0 else 'setup_required', 'trip_count':len(trips()), 'data_mode':'evaluation' if DATA_ROOT == DEFAULT_EVALUATION_ROOT.resolve() or os.getenv('PATHSNAP_EVALUATION_DIR') else 'development'}

@app.get('/trips')
def list_trips():
    return [{'trip_id':k} for k in sorted(trips())]

@app.post('/trips/{trip_id}/predict')
def predict(trip_id: str):
    trip = trips().get(trip_id)
    if trip is None: raise HTTPException(404, 'Unknown trip ID')
    try:
        model = classifier()
        clean, audit = clean_track(trip)
        nodes, arcs = read_graph(trip)
        road = RoadIndex(nodes, arcs, Config())
        candidates, obs = build_candidate_features(clean, road)
        if not len(candidates['X']): raise ValueError('No road candidates found')
        prob = model.predict_proba(candidates['X'])[:,1]
        chosen = decode(candidates['observation_index'], candidates['arc_id'], prob, obs, arcs)
        points = [{'observation_id':int(row.observation_id), 'latitude':float(row.latitude), 'longitude':float(row.longitude), 'timestamp_s':float(row.timestamp_s), 'arc_id':int(chosen[i]) if chosen[i]>=0 else None} for i,row in obs.iterrows()]
        route=[]
        for arc in chosen:
            if arc>=0 and (not route or route[-1]['arc_id'] != int(arc)):
                a,b=arcs[arc]; route.append({'arc_id':int(arc), 'coordinates':[[float(nodes[a,1]),float(nodes[a,0])],[float(nodes[b,1]),float(nodes[b,0])]]})
        matched=chosen>=0
        metrics={'matched_observations':int(matched.sum()), 'total_observations':len(chosen), 'coverage':float(matched.mean()), 'reference':None}
        if (trip / f'{trip.name}.route').is_file():
            labels=align_reference(clean,road,read_reference(trip,len(arcs)))
            usable=labels.label_usable.to_numpy(dtype=bool)&matched
            metrics['reference']={'type':'weak_monotone_route_alignment', 'usable_observations':int(usable.sum()), 'pointwise_accuracy':float(np.mean(chosen[usable]==labels.reference_arc_id.to_numpy()[usable])) if usable.any() else None}
        logging.info('trip=%s observations=%d matched=%d',trip_id,len(chosen),int(matched.sum()))
        return {'trip_id':trip_id,'observations':points,'route':route,'metrics':metrics,'audit':audit}
    except (ValueError,FileNotFoundError) as exc:
        raise HTTPException(422,str(exc)) from exc