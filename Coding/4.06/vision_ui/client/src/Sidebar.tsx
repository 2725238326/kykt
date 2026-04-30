import { WorkspaceTab } from "./App";

interface SidebarProps {
  activeWorkspace: WorkspaceTab;
  onWorkspaceChange: (tab: WorkspaceTab) => void;
  summary: {
    total: number;
    running: number;
  };
}

export function Sidebar({ activeWorkspace, onWorkspaceChange, summary }: SidebarProps) {
  const navItems: Array<{ key: WorkspaceTab; label: string; icon: string }> = [
    { key: "queue", label: "工作队列", icon: "📋" },
    { key: "create", label: "新建任务", icon: "➕" },
    { key: "samples", label: "样例矩阵", icon: "📊" },
    { key: "development", label: "研发加速", icon: "🚀" },
    { key: "system", label: "系统配置", icon: "⚙️" },
  ];

  return (
    <aside className="workspace-sidebar">
      <div className="sidebar-header">
        <div className="brand-mark">K</div>
        <strong>KYKT Vision</strong>
      </div>
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <button
            key={item.key}
            className={`nav-item ${activeWorkspace === item.key ? "active" : ""}`}
            onClick={() => onWorkspaceChange(item.key)}
          >
            <span>{item.icon}</span>
            {item.label}
            {item.key === "queue" && summary.running > 0 && (
              <span className="section-pill" style={{ marginLeft: "auto" }}>{summary.running}</span>
            )}
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className="dense-text">
          <p>Local Desktop Workbench</p>
          <small className="muted-text">v0.1.0-alpha</small>
        </div>
      </div>
    </aside>
  );
}
