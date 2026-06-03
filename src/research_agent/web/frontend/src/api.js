// Thin fetch wrapper. Cookies are sent automatically same-origin; `include`
// keeps it working behind the dev proxy too. A 401 means "not signed in".

async function get(path) {
  const res = await fetch(path, { credentials: "include" });
  if (res.status === 401) throw new Error("unauthorized");
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

export const api = {
  me: () => get("/api/me"),
  projects: () => get("/api/projects"),
  project: (id) => get(`/api/projects/${id}`),
  file: (relPath) => get(`/api/file?path=${encodeURIComponent(relPath)}`),
  tasks: () => get("/api/tasks"),
  task: (id) => get(`/api/tasks/${id}`),
};
