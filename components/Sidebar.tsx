import Link from "next/link";
import { Activity, Camera, Database, Gauge, HeartPulse, Home, Settings } from "lucide-react";

const items = [
  ["/", "Overview", Home],
  ["/analyze", "Analyze", Camera],
  ["/anemia", "Anemia / FAMACHA", HeartPulse],
  ["/tracking", "Live IDs", Activity],
  ["/records", "Herd Records", Database],
  ["/system", "System", Settings],
];

export function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark"><Gauge size={21} /></div>
        <div>
          <div className="brand-title">GoatVision</div>
          <div className="brand-sub">Smart goat monitoring</div>
        </div>
      </div>

      <div className="sidebar-label">WORKSPACE</div>
      <nav className="nav">
        {items.map(([href, label, Icon]) => (
          <Link key={href} href={href as string} className="nav-link">
            <Icon size={17} strokeWidth={1.9} />
            <span>{label}</span>
          </Link>
        ))}
      </nav>

      <div className="sidebar-divider" />

      <div className="sidebar-callout">
        <div className="mini-orb" />
        <div>
          <b>Vision engine</b>
          <span>Weight + FAMACHA screening</span>
        </div>
      </div>

      <div className="sidebar-footer">
        <span className="status-dot" /> Detection · view · weight · anemia
      </div>
    </aside>
  );
}
