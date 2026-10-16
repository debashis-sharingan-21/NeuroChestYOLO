import { ReactNode } from "react";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-primary pt-24 md:pt-32 pb-20 px-4 md:px-8 lg:px-20">
      <div className="max-w-7xl mx-auto">
        {children}
      </div>
    </div>
  );
}
