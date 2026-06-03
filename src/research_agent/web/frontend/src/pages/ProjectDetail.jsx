import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { api } from "../api.js";

const KIND_LABELS = {
  lit_review: "Literature review",
  council: "Council proposal",
  methodology: "Methodology",
  experiments: "Experiments",
  paper: "Paper",
};

export default function ProjectDetail() {
  const { id } = useParams();
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [openFile, setOpenFile] = useState(null); // { rel, content }

  useEffect(() => {
    api.project(id).then(setData).catch((e) => setError(String(e)));
  }, [id]);

  async function view(rel) {
    setOpenFile({ rel, content: "Loading…" });
    try {
      const content = await api.file(rel);
      setOpenFile({ rel, content: typeof content === "string" ? content : JSON.stringify(content, null, 2) });
    } catch (e) {
      setOpenFile({ rel, content: `Error: ${e}` });
    }
  }

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p>Loading project…</p>;

  const filesByKind = {};
  for (const f of data.files) {
    const parts = f.rel.split("/"); // projects/<slug>/<kind>/...
    const kind = parts[2] || "other";
    (filesByKind[kind] ||= []).push(f);
  }

  return (
    <div>
      <Link to="/" className="back">← Projects</Link>
      <h2>{data.project.name}</h2>
      <div className="muted mono">{data.project.slug}</div>

      <div className="split">
        <div className="files">
          {Object.keys(KIND_LABELS).map((kind) =>
            filesByKind[kind] ? (
              <div key={kind} className="kind">
                <h3>{KIND_LABELS[kind]}</h3>
                <ul>
                  {filesByKind[kind].map((f) => (
                    <li key={f.rel}>
                      <button className="linkbtn" onClick={() => view(f.rel)}>
                        {f.rel.split("/").slice(3).join("/")}
                      </button>
                      <span className="muted small"> · {fmtSize(f.size)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null
          )}
          {data.files.length === 0 && <p className="muted">No files yet.</p>}
        </div>

        <div className="reader">
          {openFile ? (
            <>
              <div className="reader-head">
                <span className="mono small">{openFile.rel}</span>
                <a href={`/api/file?path=${encodeURIComponent(openFile.rel)}`} target="_blank" rel="noreferrer">open raw</a>
              </div>
              <pre className="filebody">{openFile.content}</pre>
            </>
          ) : (
            <p className="muted">Select a file to read it here.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function fmtSize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
