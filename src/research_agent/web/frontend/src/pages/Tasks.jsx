import React, { useEffect, useState } from "react";
import { api } from "../api.js";

const STATUS_CLASS = {
  pending: "s-pending",
  running: "s-running",
  done: "s-done",
  failed: "s-failed",
  cancelled: "s-cancelled",
};

export default function Tasks() {
  const [tasks, setTasks] = useState([]);
  const [live, setLive] = useState(false);
  const [detail, setDetail] = useState(null);

  // Live feed via Server-Sent Events; falls back to one fetch if SSE drops.
  useEffect(() => {
    const es = new EventSource("/api/tasks/stream", { withCredentials: true });
    es.addEventListener("tasks", (e) => {
      setLive(true);
      try {
        setTasks(JSON.parse(e.data));
      } catch {
        /* ignore */
      }
    });
    es.onerror = () => setLive(false);
    api.tasks().then(setTasks).catch(() => {});
    return () => es.close();
  }, []);

  async function open(id) {
    setDetail({ id, loading: true });
    try {
      setDetail(await api.task(id));
    } catch (e) {
      setDetail({ id, error: String(e) });
    }
  }

  return (
    <div>
      <h2>
        Tasks <span className={live ? "live on" : "live"}>{live ? "● live" : "○ offline"}</span>
      </h2>
      <table className="tasks">
        <thead>
          <tr><th>#</th><th>Project</th><th>Agent</th><th>Status</th><th>Input</th><th>When</th></tr>
        </thead>
        <tbody>
          {tasks.map((t) => (
            <tr key={t.id} onClick={() => open(t.id)} className="clickable">
              <td>{t.id}</td>
              <td className="muted">{t.project_name || "—"}</td>
              <td className="mono">{t.agent}</td>
              <td><span className={`badge ${STATUS_CLASS[t.status] || ""}`}>{t.status}</span></td>
              <td className="truncate">{t.input}</td>
              <td className="muted small">{t.created_at ? new Date(t.created_at).toLocaleTimeString() : ""}</td>
            </tr>
          ))}
          {tasks.length === 0 && (
            <tr><td colSpan="6" className="muted">No tasks yet.</td></tr>
          )}
        </tbody>
      </table>

      {detail && (
        <div className="modal" onClick={() => setDetail(null)}>
          <div className="modal-body" onClick={(e) => e.stopPropagation()}>
            <button className="close" onClick={() => setDetail(null)}>×</button>
            <h3>Task #{detail.id}</h3>
            {detail.loading && <p>Loading…</p>}
            {detail.error && <p className="error">{detail.error}</p>}
            {detail.status && (
              <>
                <p><b>{detail.agent}</b> · <span className={`badge ${STATUS_CLASS[detail.status] || ""}`}>{detail.status}</span></p>
                <h4>Input</h4>
                <pre className="filebody">{detail.input}</pre>
                {detail.result && (<><h4>Result</h4><pre className="filebody">{detail.result}</pre></>)}
                {detail.error && (<><h4>Error</h4><pre className="filebody">{detail.error}</pre></>)}
                <h4>Trace ({(detail.trace || []).length} steps)</h4>
                <pre className="filebody small">{JSON.stringify(detail.trace || [], null, 2)}</pre>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
