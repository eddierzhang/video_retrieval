"""Streamlit upload flow for local video search."""
import json
from pathlib import Path

import streamlit as st

import local_search
from ui_resources import timestamp


def render():
    st.subheader('Upload and search — no API key')
    st.caption('Processing runs on this computer. First use needs internet to download free model weights.')
    st.info('Visual search ranks sampled frames. It does not verify events or provide the advanced temporal reasoning of the OpenRouter pipeline.')
    upload = st.file_uploader('Upload a video', type=['mp4', 'mov', 'mkv', 'avi', 'webm'])
    interval = st.select_slider('Sample one frame every (seconds)', options=[1, 2, 3, 5, 10], value=3)
    speech = st.checkbox('Also transcribe speech locally', value=False)
    st.caption('Smaller sampling intervals improve coverage but take longer. Speech search uses matching words in the transcript.')
    if st.button('Process uploaded video', type='primary', disabled=upload is None):
        st.session_state.pop('local_results', None)
        st.session_state.pop('local_video', None)
        try:
            with st.status('Processing uploaded video…', expanded=True) as status:
                bar = st.progress(0.0)
                def progress(fraction, message):
                    bar.progress(float(fraction), text=message)
                source = local_search.save_upload(upload)
                data = local_search.process_video(source, interval, speech, progress)
                data['display_name'] = upload.name
                st.session_state['local_video'] = data
                status.update(label='Video ready', state='complete', expanded=False)
        except Exception as exc:
            st.error(f'Processing failed: {exc}')
            st.caption('Check the video format and available disk space. On first use, model downloads need internet.')

    data = st.session_state.get('local_video')
    if data is None:
        st.caption('Upload a video and click Process uploaded video to begin.')
        return
    st.subheader(f"Ready: {data.get('display_name', Path(data['source']).name)}")
    st.caption(f"{timestamp(data['duration'])} · {len(data['times'])} sampled frames")
    if data.get('transcribed') and not data.get('has_audio', True):
        st.info('This video has no audio track. Visual search is available.')
    with st.expander('Preview processed video'):
        st.video(data['source'])
    with st.form('local_search'):
        query = st.text_input('Describe a scene or enter spoken words')
        mode = st.radio('Search in', ['Visual', 'Speech'] if data.get('segments') else ['Visual'], horizontal=True)
        top_k = st.slider('Number of results', 1, 20, 5)
        submit = st.form_submit_button('Search locally', type='primary')
    if submit:
        st.session_state.pop('local_results', None)
        try:
            with st.spinner('Searching locally…'):
                matches = local_search.search(data, query, mode, top_k)
            st.session_state['local_results'] = {'query': query, 'mode': mode, 'matches': matches}
        except Exception as exc:
            st.error(f'Search failed: {exc}')
    result = st.session_state.get('local_results')
    if result is not None:
        st.subheader(f"{len(result['matches'])} ranked moments")
        st.write(result['query'])
        st.caption('Scores are similarity rankings, not confidence or proof that an event occurred.')
        st.download_button('Download local results', json.dumps(result, indent=2),
                           file_name='local_results.json', mime='application/json')
        if not result['matches']:
            st.info('No transcript words matched. Try words actually spoken in the video.')
        for row in result['matches']:
            with st.container(border=True):
                st.write(f"{timestamp(row['start'])} – {timestamp(row['end'])}")
                st.caption(f"Similarity: {row['score']:.3f}")
                if row.get('text'):
                    st.write(row['text'])
                st.video(data['source'], start_time=int(row['start']), end_time=max(int(row['start']) + 1, int(row['end'])))
