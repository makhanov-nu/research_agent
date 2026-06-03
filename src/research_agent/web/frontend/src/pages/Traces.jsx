import React from "react";

// The Phoenix trace UI, reverse-proxied at /phoenix behind the same auth.
// Embedded here so traces live inside the app; "open full" escapes the iframe.
export default function Traces() {
  return (
    <div className="traces">
      <div className="traces-head">
        <h2>Traces</h2>
        <a href="/phoenix/" target="_blank" rel="noreferrer">open full ↗</a>
      </div>
      <iframe className="phoenix-frame" src="/phoenix/" title="Phoenix traces" />
      <p className="muted small">
        Powered by a local Arize Phoenix instance. If this is blank, Phoenix
        isn't running (`docker compose up -d phoenix`) or `PHOENIX_ENABLED` is off.
      </p>
    </div>
  );
}
