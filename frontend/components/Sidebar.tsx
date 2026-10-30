 "use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Camera,
  HeartPulse,
  Database,
  Settings,
  ScanSearch,
} from "lucide-react";

const items = [
  ["/detect", "Analyze", Camera],
  ["/anemia", "Anemia / FAMACHA", HeartPulse],
  ["/records", "Herd Records", Database],
  ["/system", "System", Settings],
] as const;

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">G</div>
        <div>
          <div className="brand-name">GoatVision</div>
          <div className="brand-subtitle">Livestock Intelligence</div>
        </div>
      </div>

      <nav className="nav" aria-label="Main navigation">
        {items.map(([href, label, Icon]) => {
          const active =
            pathname === href ||
            pathname.startsWith(`${href}/`);

          return (
            <Link
              key={href}
              href={href}
              className={`nav-link ${active ? "active" : ""}`}
            >
              <Icon size={19} strokeWidth={1.9} />
              <span>{label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <span className="status-dot" />
        AI services connected
      </div>
    </aside>
  );
}
