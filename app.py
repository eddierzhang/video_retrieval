"""Run locally with: python -m streamlit run app.py"""
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import streamlit as st

from ui_resources import ROOT, discover_manifests, index_defaults, load_pipeline, read_manifest, timestamp


st.set_page_config(page_title='Video Search', page_icon='🎬', layout='wide')
st.title('Video Search')
st.caption('Find moments in your indexed videos using natural language.')

with st.sidebar:
    st.header('Video library')
    manifests = discover_manifests()
    selected = st.selectbox('Indexed video', [str(p) for p in manifests] + ['Custom manifest'],
                            format_func=lambda p: Path(p).parent.name if p != 'Custom manifest' else p)
    manifest_path = st.text_input('Manifest path', '') if selected == 'Custom manifest' else selected
    if not manifest_path:
        st.info('Index a video in model_completed.ipynb, then select its manifest.json here.')
        st.stop()
    try:
        manifest = read_manifest(manifest_path)
    except (OSError, ValueError, TypeError) as exc:
        st.error(f'Cannot read manifest: {exc}')
        st.stop()
    video_path = st.text_input('Source video path', manifest['video']['path'], key=f'video:{manifest_path}',
                              help='Use the same original video that these indexes were built from.')
    with st.expander('Index locations'):
        defaults = index_defaults(manifest_path)
        visual_dir = st.text_input('Visual index folder', defaults[0], key=f'visual:{manifest_path}')
        metadata_dir = st.text_input('Metadata index folder', defaults[1], key=f'metadata:{manifest_path}')
        transcript_dir = st.text_input('Transcript index folder', defaults[2], key=f'transcript:{manifest_path}')
    if st.button('Reload indexes'):
        st.session_state.pop('pipeline', None)
        st.session_state.pop('result', None)
    st.caption('API key: configured' if os.environ.get('OPENROUTER_API_KEY') else 'Set OPENROUTER_API_KEY before starting the app.')

identity = (manifest_path, video_path, visual_dir, metadata_dir, transcript_dir)
if st.session_state.get('identity') != identity:
    st.session_state.pop('pipeline', None)
    st.session_state.pop('result', None)
    st.session_state['identity'] = identity

st.subheader(manifest['video'].get('filename', Path(video_path).name))
st.caption(f"Duration: {timestamp(manifest['video']['duration'])}")
source = Path(video_path).expanduser()
if not source.is_file():
    st.warning('The source video was not found. Update its path in the sidebar.')
else:
    with st.expander('Preview source video'):
        st.video(str(source))

with st.form('search'):
    query = st.text_input('What are you looking for?', placeholder='Find all footage of the man in the black shirt.')
    with st.expander('Search settings'):
        pro = st.checkbox('Use Pro verification', value=True)
        refine = st.checkbox('Refine clip boundaries', value=True)
        confidence = st.slider('Minimum Pro confidence', 0.0, 1.0, 0.25, 0.05)
    submitted = st.form_submit_button('Search video', type='primary')

if submitted:
    st.session_state.pop('result', None)
    if not query.strip():
        st.warning('Enter a search query.')
    elif not os.environ.get('OPENROUTER_API_KEY'):
        st.error('Set OPENROUTER_API_KEY in your terminal, then restart the app.')
    elif not source.is_file():
        st.error('Select an existing source video first.')
    elif not shutil.which('ffmpeg'):
        st.error('FFmpeg must be installed and available on PATH.')
    else:
        try:
            with st.status('Searching video…', expanded=True) as status:
                if 'pipeline' not in st.session_state:
                    st.write('Loading saved visual, metadata, and transcript indexes…')
                    st.session_state['pipeline'] = load_pipeline(*identity)
                st.write('Finding candidates, verifying matches, and extracting clips. This may take several minutes.')
                result = st.session_state['pipeline'].retrieve(
                    query.strip(), run_pro_verification=pro, refine_boundaries=refine,
                    pro_min_confidence=confidence,
                    output_root=str(ROOT / 'final_results' / 'ui' / uuid4().hex),
                )
                st.session_state['result'] = result
                status.update(label='Search complete', state='complete', expanded=False)
        except Exception as exc:
            # API exceptions can include request details; never render credentials.
            message = str(exc).replace(os.environ.get('OPENROUTER_API_KEY', ''), '[redacted]')
            st.error(f'Search failed: {message}')
            st.info('Check your index locations, API account, and FFmpeg installation, then retry.')

result = st.session_state.get('result')
if result is not None:
    matches = result.get('matches', [])
    st.subheader(f'{len(matches)} matching moments')
    st.write(result.get('query', ''))
    st.download_button('Download results JSON', json.dumps(result, indent=2, default=str),
                       file_name='search_results.json', mime='application/json')
    if not matches:
        st.info('No matches passed verification. Try a different description or lower the confidence threshold.')
    for i, match in enumerate(matches, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {timestamp(match.get('start', 0))} – {timestamp(match.get('end', 0))}**")
            if match.get('confidence') is not None:
                st.caption(f"Confidence: {float(match['confidence']):.0%}")
            st.write(match.get('description') or match.get('reason') or 'Matching moment')
            clip = Path(match.get('clip_path') or '__missing__')
            if clip.is_file():
                st.video(str(clip))
            elif source.is_file():
                st.video(str(source), start_time=max(0, int(match.get('start', 0))))
                st.caption('Showing the original video from this timestamp; extracted clip is unavailable.')
            with st.expander('Match details'):
                st.json(match)
    with st.expander('Query plan and diagnostics'):
        st.json({k: v for k, v in result.items() if k != 'matches'})
