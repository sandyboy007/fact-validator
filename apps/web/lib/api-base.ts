const DEFAULT_LOCAL_API_BASE_URL = "http://127.0.0.1:8000";

function isLocalHost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

export function getApiBaseUrl() {
  const configuredBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/$/, "");
  }

  if (typeof window === "undefined") {
    return DEFAULT_LOCAL_API_BASE_URL;
  }

  const { hostname, origin } = window.location;
  if (isLocalHost(hostname)) {
    return DEFAULT_LOCAL_API_BASE_URL;
  }

  return origin;
}