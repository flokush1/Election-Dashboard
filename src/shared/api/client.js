export class ApiError extends Error {
  constructor(message, { status = 0, payload = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

const defaultHeaders = {
  Accept: 'application/json'
};

export async function apiRequest(path, options = {}) {
  const {
    method = 'GET',
    body,
    headers = {},
    timeoutMs = 30000,
    signal
  } = options;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  const abort = () => controller.abort();
  if (signal) {
    if (signal.aborted) abort();
    else signal.addEventListener('abort', abort, { once: true });
  }

  try {
    const isFormData = typeof FormData !== 'undefined' && body instanceof FormData;
    const response = await fetch(path, {
      method,
      body: body && !isFormData && typeof body !== 'string' ? JSON.stringify(body) : body,
      headers: {
        ...defaultHeaders,
        ...(body && !isFormData ? { 'Content-Type': 'application/json' } : {}),
        ...headers
      },
      signal: controller.signal
    });

    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json')
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      const message = payload?.error || payload?.message || `Request failed (${response.status})`;
      throw new ApiError(message, { status: response.status, payload });
    }

    return payload;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new ApiError('Request timed out', { status: 408 });
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export const apiGet = (path, options) => apiRequest(path, { ...options, method: 'GET' });
export const apiPost = (path, body, options) => apiRequest(path, { ...options, method: 'POST', body });
