import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { api } from "../api.js";

const KIND_LABELS = {
  uploads: "Uploads",
  lit_review: "Literature review",
  council: "Council proposal",
  methodology: "Methodology",
  experiments: "Experiments",
  paper: "Paper",
};

const TEXT_EXTS = new Set(["txt", "md", "tex", "bib", "json", "py", "log", "csv", "rst", "yaml", "yml"]);
const IMG_EXTS = new Set(["png", "jpg", "jpeg", "gif", "webp", "svg", "bmp", "tiff", "tif"]);

function fileExt(name) {
  const parts = name.split(".");
  return parts.length > 1 ? parts.pop().toLowerCase() : "";
}

export default function ProjectDetail() {
  const { id } = useParams();
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [openFile, setOpenFile] = useState(null); // { rel, name, content, isImage }

  useEffect(() => {
    api.project(id).then(setData).catch((e) => setError(String(e)));
  }, [id]);

  async function view(f) {
    const ext = fileExt(f.name);
    if (IMG_EXTS.has(ext)) {
      setOpenFile({ rel: f.rel, name: f.name, content: null, isImage: true });
    } else {
      setOpenFile({ rel: f.rel, name: f.name, content: "Loading…", isImage: false });
      try {
        const content = await api.file(f.rel);
        setOpenFile({ rel: f.rel, name: f.name, content: typeof content === "string" ? content : JSON.stringify(content, null, 2), isImage: false });
      } catch (e) {
        setOpenFile({ rel: f.rel, name: f.name, content: `Error: ${e}`, isImage: false });
      }
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

  // Ordered sections: known kinds first, then any unknown ones
  const knownKinds = Object.keys(KIND_LABELS);
  const unknownKinds = Object.keys(filesByKind).filter((k) => !KIND_LABELS[k]);
  const allKinds = [...knownKinds, ...unknownKinds];

  return (
    <div>
      <Link to="/" className="back">← Projects</Link>
      <h2>{data.project.name}</h2>
      <div className="muted mono">{data.project.slug}</div>

      <div className="split">
        <div className="files">
          {allKinds.map((kind) =>
            filesByKind[kind] ? (
              <div key={kind} className="kind">
                <h3>{KIND_LABELS[kind] || kind}</h3>
                <ul>
                  {filesByKind[kind].map((f) => {
                    const ext = fileExt(f.name);
                    const isImg = IMG_EXTS.has(ext);
                    const isText = TEXT_EXTS.has(ext);
                    const label = f.rel.split("/").slice(3).join("/");
                    return (
                      <li key={f.rel}>
                        {isImg || isText ? (
                          <button className="linkbtn" onClick={() => view(f)}>
                            {label}
                          </button>
                        ) : (
                          <a href={`/api/file?path=${encodeURIComponent(f.rel)}`} download={f.name}>
                            {label}
                          </a>
                        )}
                        <span className="muted small"> · {fmtSize(f.size)}</span>
                      </li>
                    );
                  })}
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
              {openFile.isImage ? (
                <img
                  src={`/api/file?path=${encodeURIComponent(openFile.rel)}`}
                  alt={openFile.name}
                  style={{ maxWidth: "100%", maxHeight: "70vh", objectFit: "contain" }}
                />
              ) : (
                <pre className="filebody">{openFile.content}</pre>
              )}
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
