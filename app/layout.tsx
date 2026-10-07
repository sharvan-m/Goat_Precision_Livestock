import "./globals.css";
import { Sidebar } from "@/components/Sidebar";

export const metadata = {
  title: "GoatVision — Smart Goat Monitoring",
  description: "Computer-vision goat monitoring and weight intelligence.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="app-shell">
          <Sidebar />
          <main className="main-content">{children}</main>
        </div>
      </body>
    </html>
  );
}
