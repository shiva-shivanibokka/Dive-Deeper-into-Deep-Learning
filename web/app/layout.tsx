import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Deep Learning Playground",
  description: "Eight deep-learning architectures, trained in PyTorch, running live in your browser.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
