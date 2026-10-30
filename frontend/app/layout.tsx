import "./globals.css";
import Sidebar from "@/components/Sidebar";
import { AnalysisProvider } from "@/lib/AnalysisContext";

export const metadata = {
  title: "GoatVision - Smart Goat Monitoring",
  description: "Computer-vision goat monitoring and weight intelligence.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AnalysisProvider>
          <div className="app-shell">
            <Sidebar />
            <main className="main-content">{children}</main>
          </div>
        </AnalysisProvider>
      </body>
    </html>
  );
}
