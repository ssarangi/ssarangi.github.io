/**
 * Site-specific switches that sit outside the AstroPaper theme config.
 */
export const extras = {
  /** Giscus comments (GitHub Discussions). OFF by default. See README. */
  giscus: {
    enabled: false,
    repo: "ssarangi/ssarangi.github.io",
    repoId: "",
    category: "Comments",
    categoryId: "",
  },
  /**
   * Privacy-friendly analytics. None by default. Set one of:
   *   { provider: "plausible", domain: "example.com" }
   *   { provider: "goatcounter", code: "mysite" }   // https://mysite.goatcounter.com
   */
  analytics: null as
    | null
    | { provider: "plausible"; domain: string }
    | { provider: "goatcounter"; code: string },
};
