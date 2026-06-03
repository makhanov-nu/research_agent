import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";

export default function Projects() {
  const [projects, setProjects] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.projects().then(setProjects).catch((e) => setError(String(e)));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!projects) return <p>Loading projects…</p>;
  if (projects.length === 0)
    return <p className="muted">No projects yet — start a chat with the agent.</p>;

  return (
    <div>
      <h2>Projects</h2>
      <div className="grid">
        {projects.map((p) => (
          <Link key={p.id} to={`/projects/${p.id}`} className="card project">
            <div className="project-name">{p.name}</div>
            <div className="muted mono">{p.slug}</div>
            <div className="muted small">
              {p.created_at ? new Date(p.created_at).toLocaleString() : ""}
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
