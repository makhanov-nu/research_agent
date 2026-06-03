import React, { useEffect, useState } from "react";
import { Routes, Route, Link, Navigate } from "react-router-dom";
import { api } from "./api.js";
import Projects from "./pages/Projects.jsx";
import ProjectDetail from "./pages/ProjectDetail.jsx";
import Tasks from "./pages/Tasks.jsx";

export default function App() {
  const [user, setUser] = useState(undefined); // undefined = loading

  useEffect(() => {
    api.me().then(setUser).catch(() => setUser(null));
  }, []);

  if (user === undefined) return <div className="center">Loading…</div>;
  if (user === null) return <Login />;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">🔬 Research Agent</div>
        <nav>
          <Link to="/">Projects</Link>
          <Link to="/tasks">Tasks</Link>
        </nav>
        <div className="user">
          {user.email} · <a href="/auth/logout">Sign out</a>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Projects />} />
          <Route path="/projects/:id" element={<ProjectDetail />} />
          <Route path="/tasks" element={<Tasks />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  );
}

function Login() {
  return (
    <div className="center">
      <div className="card login">
        <h1>🔬 Research Agent</h1>
        <p>Sign in to view your projects and live tasks.</p>
        <a className="btn" href="/auth/login">Sign in with WorkOS</a>
      </div>
    </div>
  );
}
