/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // onnxruntime-web ships its own wasm; don't let webpack try to bundle node-only bits
  webpack: (config) => {
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false, crypto: false };
    // transformers.js ships node-only backends; keep them out of the browser bundle
    config.resolve.alias = { ...config.resolve.alias, "onnxruntime-node": false, sharp: false };
    return config;
  },
};

export default nextConfig;
