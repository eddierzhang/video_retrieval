// Thin client for the local Moments API. Mutations carry X-Moments, which the server requires.

const APP_HEADER = { 'X-Moments': '1' };

async function request(method, url, body) {
  const options = { method, headers: method === 'GET' ? {} : { ...APP_HEADER } };
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  let data = null;
  try { data = await response.json(); } catch { /* empty or non-JSON body */ }
  if (!response.ok) throw new Error(data?.error || `Request failed (${response.status})`);
  return data;
}

const videoUrl = (id) => `/api/videos/${encodeURIComponent(id)}`;

export const api = {
  system: () => request('GET', '/api/system'),
  saveSettings: (settings) => request('PUT', '/api/settings', settings),
  videos: () => request('GET', '/api/videos'),
  video: (id) => request('GET', videoUrl(id)),
  deleteVideo: (id) => request('DELETE', videoUrl(id)),
  reindex: (id) => request('POST', `${videoUrl(id)}/index`),
  transcript: (id) => request('GET', `${videoUrl(id)}/transcript`),
  createSearch: (id, body) => request('POST', `${videoUrl(id)}/searches`, body),
  search: (id, searchId) => request('GET', `${videoUrl(id)}/searches/${encodeURIComponent(searchId)}`),
  deleteSearch: (id, searchId) => request('DELETE', `${videoUrl(id)}/searches/${encodeURIComponent(searchId)}`),
  searchFeedback: (id, searchId, body) => request('POST', `${videoUrl(id)}/searches/${encodeURIComponent(searchId)}/feedback`, body),
  addMissed: (id, searchId, body) => request('POST', `${videoUrl(id)}/searches/${encodeURIComponent(searchId)}/missed`, body),
  refineSearch: (id, searchId) => request('POST', `${videoUrl(id)}/searches/${encodeURIComponent(searchId)}/refine`),
  cancelJob: (jobId) => request('POST', `/api/jobs/${encodeURIComponent(jobId)}/cancel`),
};

// XHR rather than fetch so the sidebar can show upload progress.
export function uploadVideo(file, onProgress) {
  const xhr = new XMLHttpRequest();
  const promise = new Promise((resolve, reject) => {
    xhr.open('POST', '/api/videos');
    xhr.setRequestHeader('X-Moments', '1');
    xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name));
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    xhr.addEventListener('load', () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error(data?.error || `Upload failed (${xhr.status})`));
    });
    xhr.addEventListener('error', () => reject(new Error('Upload failed: the server could not be reached.')));
    xhr.addEventListener('abort', () => reject(new Error('Upload cancelled.')));
    xhr.send(file);
  });
  return { promise, abort: () => xhr.abort() };
}
