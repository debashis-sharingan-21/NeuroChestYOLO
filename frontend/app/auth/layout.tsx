import { ReactNode } from "react";

export default function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col bg-primary p-6 ">
      {/* Spacer to push content below the fixed navbar */}
      {/* <div className="h- w-full shrink-0" /> */}
      
      <div className="flex-1 flex items-center justify-center mt-14 pb-4">
        <div className="w-full max-w-md bg-surface/50 backdrop-blur-xl p-8 rounded-container border border-accent/20">
          {children}
        </div>
      </div>
    </div>
  );
}
