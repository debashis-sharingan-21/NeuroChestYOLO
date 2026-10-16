import Hero from "@/components/sections/Hero";
import Features from "@/components/sections/Features";
import Philosophy from "@/components/sections/Philosophy";
import Protocol from "@/components/sections/Protocol";

export default function Home() {
  return (
    <div className="flex flex-col">
      <Hero />
      <Features />
      <Philosophy />
      <Protocol />
    </div>
  );
}
