"""Evaluator UI: streamlit run frontend.py --server.address 0.0.0.0"""
import os
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title='PathSnap Evaluator',layout='wide')
st.title('PathSnap · Route reconstruction')
api=st.sidebar.text_input('API base URL',os.getenv('PATHSNAP_API_URL','http://localhost:8000')).rstrip('/')
try:
    health=requests.get(f'{api}/health',timeout=10); health.raise_for_status()
    response=requests.get(f'{api}/trips',timeout=10); response.raise_for_status()
    ids=[item['trip_id'] for item in response.json()]
except requests.RequestException as exc:
    st.error(f'Cannot reach the API: {exc}'); st.stop()
if health.json()['status']!='ready': st.warning('API setup incomplete: check the model and trip directory on the API laptop.')
if health.json().get('data_mode') == 'development': st.info('Development mode: showing the ten student-test trips. Set PATHSNAP_EVALUATION_DIR on the API laptop for reserved evaluation trips.')
if not ids: st.info('No evaluation trips available.'); st.stop()
trip_id=st.selectbox('Evaluation trip',ids)
if st.button('Reconstruct route',type='primary'):
    with st.spinner('Running model and decoder…'):
        try:
            result=requests.post(f'{api}/trips/{trip_id}/predict',timeout=180)
            result.raise_for_status(); st.session_state.result=result.json()
        except requests.RequestException as exc: st.error(f'Prediction failed: {exc}')
result=st.session_state.get('result')
if result and result['trip_id']==trip_id:
    metrics=result['metrics']; ref=metrics.get('reference')
    a,b,c=st.columns(3)
    a.metric('Matched observations',f"{metrics['matched_observations']:,} / {metrics['total_observations']:,}")
    b.metric('Coverage',f"{metrics['coverage']:.1%}")
    c.metric('Weak reference accuracy',f"{ref['pointwise_accuracy']:.1%}" if ref and ref['pointwise_accuracy'] is not None else 'Unavailable')
    if ref: st.caption(f"Accuracy uses {ref['usable_observations']:,} usable points aligned to the supplied route; these are weak labels.")
    points=result['observations']; center=[points[0]['latitude'],points[0]['longitude']]
    m=folium.Map(location=center,zoom_start=14,tiles='OpenStreetMap')
    folium.PolyLine([[p['latitude'],p['longitude']] for p in points],color='#2780c2',weight=2,opacity=.5,tooltip='Raw GPS').add_to(m)
    for segment in result['route']:
        folium.PolyLine(segment['coordinates'],color='#13a66b',weight=5,tooltip=f"Arc {segment['arc_id']}").add_to(m)
    st_folium(m,width=None,height=600)
    st.caption('Blue: GPS observations · Green: decoded road segments')



