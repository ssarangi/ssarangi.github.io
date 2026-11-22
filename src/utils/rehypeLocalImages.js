import { visit } from "unist-util-visit";
import sharp from "sharp";
import path from "node:path";

/**
 * For <img src="/img/..."> that points at a file in /public, add intrinsic
 * width/height (prevents layout shift) plus lazy loading + async decoding.
 * Keeps the legacy stable /img/... URLs while avoiding re-encoding ~230 MB.
 */
export function rehypeLocalImages() {
  const cache = new Map();
  return async tree => {
    const jobs = [];
    visit(tree, "element", node => {
      if (node.tagName !== "img") return;
      const src = node.properties?.src;
      if (
        typeof src !== "string" ||
        !src.startsWith("/") ||
        src.startsWith("//")
      )
        return;
      if (/\.(mp4|webm|mov)$/i.test(src)) {
        // Legacy posts embedded video with image syntax: ![](x.mp4)
        node.tagName = "video";
        node.properties = {
          src,
          controls: true,
          preload: "none",
          playsInline: true,
          style: "max-width:100%;height:auto",
          "aria-label": node.properties.alt || undefined,
        };
        return;
      }
      node.properties.loading = "lazy";
      node.properties.decoding = "async";
      if (node.properties.width && node.properties.height) return;
      jobs.push(
        (async () => {
          const file = path.join(process.cwd(), "public", decodeURI(src));
          if (!cache.has(file)) {
            cache.set(
              file,
              sharp(file)
                .metadata()
                .then(m => ({
                  width: m.width,
                  height: m.pageHeight ?? m.height,
                }))
                .catch(() => null)
            );
          }
          const dim = await cache.get(file);
          if (dim?.width && dim?.height) {
            node.properties.width = dim.width;
            node.properties.height = dim.height;
          } else {
            throw new Error(
              `rehypeLocalImages: missing or unreadable image ${src}`
            );
          }
        })()
      );
    });
    await Promise.all(jobs);
  };
}
