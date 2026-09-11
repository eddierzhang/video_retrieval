"""Upload and search through the original architecture with local model calls."""
import json
from pathlib import Path
import shutil
from uuid import uuid4

import streamlit as st
from local_search import save_upload, ROOT
from ui_resources import timestamp
from video_retrieval.local_backend import LocalModels, check_runtime, use_local
from video_retrieval.local_indexing import prepare_video


def render():
    st.subheader('Original architecture · local models')
    st.caption('Hierarchical indexing → multimodal retrieval → evidence fusion → recursive search → two verification passes → boundary refinement → clips')
    st.info('All inference runs locally. Processing uses a local vision-language model and can take substantially longer than simple frame search. Models see sampled frames and a speech transcript.')
    with st.expander('Local model settings and setup'):
        st.code('.\\start-local.ps1', language='powershell')
        st.caption('Keep Ollama running in another terminal. Defaults: qwen2.5:7b for planning and qwen2.5vl:3b for vision. No API key needed.')
        planner = st.text_input('Planning model', 'qwen2.5:7b')
        vision = st.text_input('Scene / first verification / refinement / OCR model', 'qwen2.5vl:3b')
        verifier = st.text_input('Second verification model', 'qwen2.5vl:3b')
        frames = st.slider('Maximum frames per model call', 4, 24, 12)
    models = LocalModels(planner, vision, verifier, frames)
    if st.button('Check local runtime'):
        try:
            check_runtime(models)
            st.success('Local runtime and selected models are ready.')
        except Exception as exc:
            st.error(str(exc))
    upload = st.file_uploader('Upload a video for the original pipeline', type=['mp4', 'mov', 'mkv', 'avi', 'webm'])
    if st.button('Process with original architecture', type='primary', disabled=upload is None):
        st.session_state.pop('architecture_pipeline', None)
        st.session_state.pop('architecture_results', None)
        try:
            if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
                raise RuntimeError('FFmpeg and ffprobe must be installed and on PATH.')
            check_runtime(models)
            with st.status('Building local indexes…', expanded=True) as status:
                detail = st.empty()
                pipeline = prepare_video(save_upload(upload), models, detail.write)
                st.session_state['architecture_pipeline'] = pipeline
                st.session_state['architecture_name'] = upload.name
                status.update(label='Original pipeline ready', state='complete', expanded=False)
        except Exception as exc:
            st.error(f'Processing failed: {exc}')
    pipeline = st.session_state.get('architecture_pipeline')
    if pipeline is None:
        return
    if pipeline.resources.local_models != models:
        st.warning('Model settings changed. Process the video again to use matching indexes.')
        return
    st.subheader(f"Ready: {st.session_state.get('architecture_name', 'video')}")
    resources = pipeline.resources
    st.caption(f"{resources.video_index.ntotal} fine chunks · {len(resources.metadata_records)} scene records · {len(resources.transcript_metadata)} speech windows")
    with st.form('architecture_search'):
        query = st.text_input('Find an event, action, spoken phrase, or visible text')
        submit = st.form_submit_button('Search with original pipeline', type='primary')
    if submit:
        st.session_state.pop('architecture_results', None)
        if not query.strip():
            st.warning('Enter a query.')
        else:
            try:
                with st.status('Retrieving and verifying locally…', expanded=True) as status:
                    detail = st.empty()
                    # Call original function in a local context to report each model stage.
                    from video_retrieval.pipeline import retrieve_video
                    with use_local(models, detail.write):
                        result = retrieve_video(query.strip(), resources, output_root=str(ROOT / 'results' / uuid4().hex))
                    result['model_backend'] = models.signature()
                    st.session_state['architecture_results'] = result
                    status.update(label='Search complete', state='complete', expanded=False)
            except Exception as exc:
                st.error(f'Search failed: {exc}')
    result = st.session_state.get('architecture_results')
    if result:
        st.subheader(f"{len(result.get('matches', []))} verified moments")
        st.write(result.get('query', ''))
        st.download_button('Download pipeline results', json.dumps(result, indent=2, default=str),
                           file_name='pipeline_results.json', mime='application/json')
        if not result.get('matches'):
            st.info('No matches passed the original verification stages.')
        for row in result.get('matches', []):
            with st.container(border=True):
                st.write(f"{timestamp(row['start'])} – {timestamp(row['end'])}")
                st.write(row.get('description', 'Matching occurrence'))
                if row.get('clip_path') and Path(row['clip_path']).is_file():
                    st.video(row['clip_path'])
                with st.expander('Evidence and confidence'):
                    st.json(row)
        with st.expander('Plan, diagnostics, and model configuration'):
            st.json({k: v for k, v in result.items() if k != 'matches'})
