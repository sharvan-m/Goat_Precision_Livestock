import Link from "next/link";
import { getJSON } from "@/lib/api";

function displayId(id: string) {
  const m = id.match(/(\d+)$/);
  return m ? m[1].padStart(2, "0") : id;
}

export default async function RecordsPage() {
  const data = await getJSON<any>("/api/records").catch(() => ({ count: 0, unique_goats: 0, rows: [] }));
  const rows = data.rows || [];
  const columns = rows.length ? Object.keys(rows[0]).slice(0, 9) : [];

  return <>
    <div className="page-top">
      <div>
        <div className="eyebrow">HERD RECORDS</div>
        <h1>Everything you have measured.</h1>
        <p className="page-copy">The same records created by the weight-analysis pipeline are available here for review.</p>
      </div>
    </div>

    <div className="stats">
      <div className="stat"><div className="stat-label">Records</div><div className="stat-value">{data.count}</div></div>
      <div className="stat"><div className="stat-label">Unique goats</div><div className="stat-value">{data.unique_goats}</div></div>
      <div className="stat"><div className="stat-label">Data source</div><div className="stat-value">CSV</div></div>
      <div className="stat"><div className="stat-label">Export</div><div className="stat-value">CSV / Excel</div></div>
    </div>

    {rows.length === 0 ? (
      <div className="card empty">No records yet. <Link href="/analyze" style={{color:"var(--green)",fontWeight:800}}>Analyze a goat →</Link></div>
    ) : (
      <div className="table-wrap">
        <table>
          <thead><tr>{columns.map(c => <th key={c}>{c.replaceAll("_", " ")}</th>)}</tr></thead>
          <tbody>
            {rows.slice().reverse().map((row: any, i: number) => (
              <tr key={i}>{columns.map(c => <td key={c}>{c === "goat_id" ? <b>{displayId(String(row[c]))}</b> : String(row[c])}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    )}
  </>;
}
