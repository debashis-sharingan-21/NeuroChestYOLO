import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const imagesDir = path.join(process.cwd(), "public", "testing_images");
    
    if (!fs.existsSync(imagesDir)) {
      return NextResponse.json({ images: [] });
    }

    const files = fs.readdirSync(imagesDir);
    const imageFiles = files.filter(file => 
      [".png", ".jpg", ".jpeg", ".webp"].includes(path.extname(file).toLowerCase())
    );

    return NextResponse.json({ images: imageFiles });
  } catch (error) {
    console.error("Error listing sample images:", error);
    return NextResponse.json({ error: "Failed to list images" }, { status: 500 });
  }
}
