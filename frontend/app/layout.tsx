import type { Metadata } from "next";
import { Sora, Instrument_Serif, Fira_Code } from "next/font/google";
import "./globals.css";
import LenisProvider from "@/components/layout/LenisProvider";
import AuthInitializer from "@/components/layout/AuthInitializer";
import { cn } from "@/lib/utils";
import { Analytics } from "@vercel/analytics/next";

const sora = Sora({
  subsets: ["latin"],
  variable: "--font-sora",
  display: "swap",
});

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  style: "italic",
  variable: "--font-instrument-serif",
  display: "swap",
});

const firaCode = Fira_Code({
  subsets: ["latin"],
  variable: "--font-fira-code",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Neuro",
  description: "Understanding how AI revolutionizes the world.",
};

import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import PageTransition from "@/components/layout/PageTransition";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={cn(
          sora.variable,
          instrumentSerif.variable,
          firaCode.variable,
          "antialiased bg-primary text-foreground min-h-screen"
        )}
      >
        <AuthInitializer />
        <LenisProvider>
          <PageTransition>
            <div className="noise-overlay" />
            <Navbar />
            <main>{children}</main>
            <Footer />
          </PageTransition>
        </LenisProvider>
        <Analytics />
      </body>
    </html>
  );
}
