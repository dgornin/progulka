/*
 * The one door to the server: JSON calls, the NDJSON stream of agent events, the session.
 *
 * The session is an HttpOnly cookie the page cannot read, so "who am I" is always
 * GET /api/me. The server enforces access on every call; the pages only redirect to
 * the login form for convenience. Every failure becomes an ApiError carrying the
 * server's Russian explanation, so pages can show it as is.
 */

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const OFFLINE = "Нет связи с сервером. Проверьте интернет и попробуйте ещё раз.";

const FALLBACK = {
  401: "Нужно войти",
  403: "Недостаточно прав",
  404: "Не найдено",
  409: "Конфликт: действие уже выполняется",
  429: "Слишком много запросов, попробуйте чуть позже",
};

/** FastAPI explains a refusal in {"detail": ...}: a string or a list of validation errors. */
function explain(status, text) {
  try {
    const { detail } = JSON.parse(text);
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((problem) => problem.msg).join("; ");
  } catch {
    // not JSON: a proxy's error page or plain text, replaced by a generic message below
  }
  return FALLBACK[status] ?? (status >= 500 ? "Сервер не смог ответить, попробуйте позже" : `Ошибка ${status}`);
}

/** Call a JSON endpoint; resolves to the parsed body (null for 204), throws ApiError. */
export async function api(path, { method = "GET", body } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, OFFLINE);
  }
  const text = await response.text();
  if (!response.ok) throw new ApiError(response.status, explain(response.status, text));
  return text ? JSON.parse(text) : null;
}

function parseLine(line) {
  try {
    return JSON.parse(line);
  } catch {
    const shown = line.length > 120 ? `${line.slice(0, 119)}…` : line;
    return { type: "warning", data: { message: `Непонятная строка от сервера: ${shown}` } };
  }
}

/**
 * POST a JSON body and call onEvent({type, data}) for every NDJSON line as soon as it
 * arrives. A refusal before the stream (401, 404, 409, 429) throws ApiError; an abort
 * through `signal` rethrows the AbortError untouched, so a caller can tell "stopped"
 * from "failed".
 */
export async function streamEvents(path, body, onEvent, { signal } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError(0, OFFLINE);
  }
  if (!response.ok || !response.body) {
    throw new ApiError(response.status, explain(response.status, await response.text()));
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new ApiError(0, "Связь с сервером прервалась во время работы агента");
    }
    buffer += decoder.decode(chunk.value, { stream: !chunk.done });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // the tail may be half a line; it waits for the next chunk
    for (const line of lines) if (line.trim()) onEvent(parseLine(line));
    if (chunk.done) break;
  }
  if (buffer.trim()) onEvent(parseLine(buffer));
}

/** The signed-in user, or null for a visitor (any other failure is thrown). */
export async function currentUser() {
  try {
    return (await api("/api/me")).user;
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) return null;
    throw error;
  }
}

/** The signed-in user; a visitor is sent to the login form and this never resolves. */
export async function requireUser() {
  const user = await currentUser();
  if (user) return user;
  const next = encodeURIComponent(location.pathname + location.search);
  location.assign(`/login?next=${next}`);
  return new Promise(() => {});
}

export async function logout() {
  await api("/api/auth/logout", { method: "POST" });
  location.assign("/");
}
