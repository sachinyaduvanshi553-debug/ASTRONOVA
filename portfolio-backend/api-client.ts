const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api';

export const getSessionId = (): string => {
  if (typeof window === 'undefined') return 'server';
  let sessionId = sessionStorage.getItem('portfolio_session_id');
  if (!sessionId) {
    sessionId = 'sess_' + Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
    sessionStorage.setItem('portfolio_session_id', sessionId);
  }
  return sessionId;
};

export async function submitContactForm(data: { name: string; email: string; message: string }) {
  const response = await fetch(`${API_BASE_URL}/contact`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-session-id': getSessionId(),
    },
    body: JSON.stringify(data),
  });

  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Failed to send message');
  return result;
}

export async function trackEvent(params: {
  eventType: 'PAGE_VIEW' | 'SECTION_VIEW' | 'PROJECT_CLICK' | 'RESUME_DOWNLOAD' | 'LINK_CLICK' | 'AI_CHAT_OPEN' | 'CONTACT_SUBMIT' | 'SKILL_HOVER' | 'CUSTOM';
  section?: string;
  element?: string;
  metadata?: Record<string, unknown>;
  dwellTimeMs?: number;
}) {
  try {
    await fetch(`${API_BASE_URL}/analytics/event`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId: getSessionId(), ...params }),
    });
  } catch (error) {
    console.debug('Analytics event failed:', error);
  }
}

export async function streamAIChat(
  message: string,
  history: Array<{ role: 'user' | 'assistant'; content: string }>,
  onChunk: (chunk: string) => void,
  onDone: () => void,
  onError: (err: string) => void
) {
  try {
    const response = await fetch(`${API_BASE_URL}/ai/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId: getSessionId(), message, history }),
    });

    if (!response.ok || !response.body) throw new Error('Failed to connect to AI assistant');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.replace('data: ', '').trim();
          if (!jsonStr) continue;
          try {
            const data = JSON.parse(jsonStr);
            if (data.type === 'chunk') onChunk(data.content);
            else if (data.type === 'done') onDone();
            else if (data.type === 'error') onError(data.message);
          } catch (e) {
            console.error('SSE JSON parse error:', e);
          }
        }
      }
    }
  } catch (err: unknown) {
    onError(err instanceof Error ? err.message : 'AI error');
  }
}
