import { BarChart3, Scale } from "lucide-react";
import { getJSON } from "@/lib/api";
import type { Analytics } from "@/lib/types";

export default async function AnalyticsPage() {
  const a = await getJSON<Analytics>("/api/analytics").catch(() => ({records:0,unique_goats:0,average_weight_kg:null,median_weight_kg:null,view_breakdown:{},weights:[]}));
  const max = Math.max(...(a.weights.length ? a.weights : [1]));
  return <>
    <div className="page-top"><div><div className="eyebrow">INSIGHTS</div><h1>Understand the measurements.</h1><p className="page-copy">A lightweight analytics layer on top of the records already produced by the backend.</p></div></div>
    <div className="stats"><div className="stat"><div className="stat-label">Records</div><div className="stat-value">{a.records}</div></div><div className="stat"><div className="stat-label">Unique goats</div><div className="stat-value">{a.unique_goats}</div></div><div className="stat"><div className="stat-label">Average predicted weight</div><div className="stat-value">{a.average_weight_kg == null ? "—" : `${a.average_weight_kg} kg`}</div></div><div className="stat"><div className="stat-label">Median predicted weight</div><div className="stat-value">{a.median_weight_kg == null ? "—" : `${a.median_weight_kg} kg`}</div></div></div>
    <div className="grid-2">
      <div className="card"><h2><Scale size={18} style={{verticalAlign:"-3px"}}/> Weight distribution</h2>{a.weights.length===0?<div className="empty">No weights available yet.</div>:<div style={{display:"flex",alignItems:"end",height:260,gap:5}}>{a.weights.map((w,i)=><div key={i} title={`${w} kg`} style={{flex:1,maxWidth:20,height:`${Math.max(7,(w/max)*100)}%`,background:"linear-gradient(180deg,var(--green),#64a778)",borderRadius:"5px 5px 2px 2px"}}/>)}</div>}</div>
      <div className="card"><h2><BarChart3 size={18} style={{verticalAlign:"-3px"}}/> View quality</h2>{Object.entries(a.view_breakdown).length===0?<div className="empty">No view records yet.</div>:Object.entries(a.view_breakdown).map(([k,v])=><div key={k} style={{marginBottom:18}}><div style={{display:"flex",justifyContent:"space-between",fontSize:12,fontWeight:750}}><span>{k}</span><span>{v}</span></div><div className="progress" style={{marginTop:7}}><div style={{width:`${Math.min(100,(v/a.records)*100)}%`}}/></div></div>)}</div>
    </div>
    <p className="footer-note">Analytics reflects logged model outputs, not independently validated biological measurements.</p>
  </>;
}
