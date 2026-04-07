import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/:path*`,
      },
    ]
  },
  images: {
    remotePatterns: [
      // Allow data URIs for local Grad-CAM heatmaps
      // Add S3/R2 bucket domain here when deploying to production
    ],
  },
}

export default nextConfig
