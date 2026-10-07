GoatVision — restore Anemia / FAMACHA connection

This patch restores the existing Anemia / FAMACHA page and reconnects it to the sidebar without changing the goat detection/annotation pipeline.

Files:
1. Sidebar.tsx -> replace frontend/components/Sidebar.tsx
2. anemia_page.tsx -> rename to frontend/app/anemia/page.tsx
3. globals_additions.css -> append to frontend/app/globals.css

The page calls:
GET  /api/anemia/status
POST /api/anemia/analyze

No backend changes are included. Existing backend anemia routes should remain intact.
