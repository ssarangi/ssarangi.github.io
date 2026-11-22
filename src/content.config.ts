import { defineCollection } from "astro:content";
import { z } from "astro/zod";
import { glob } from "astro/loaders";
import config from "@/config";

export const BLOG_PATH = "src/content/posts";

const posts = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: `./${BLOG_PATH}` }),
  schema: ({ image }) =>
    z.object({
      author: z.string().default(config.site.author),
      pubDate: z.coerce.date(),
      updatedDate: z.coerce.date().optional().nullable(),
      title: z.string(),
      description: z.string(),
      tags: z.array(z.string()).default(["others"]),
      draft: z.boolean().default(false),
      /** Pinned on the home page "Featured" section. */
      featured: z.boolean().default(false),
      /** Old/low-signal posts: kept (and routable) but listed only on /archives. */
      archived: z.boolean().default(false),
      heroImage: image().optional(),
      heroImageAlt: z.string().default(""),
      ogImage: image().or(z.string()).optional(),
      canonicalURL: z.url().optional(),
      /** Series name; posts sharing it get prev/next navigation. */
      series: z.string().optional(),
      seriesOrder: z.number().int().positive().optional(),
      /** Set when the post also exists elsewhere (e.g. the Medium cross-post). */
      crosspostURL: z.url().optional(),
      hideEditPost: z.boolean().optional(),
      hideToc: z.boolean().optional(),
      // An empty value means "not set", so the site timezone applies.
      timezone: z
        .string()
        .optional()
        .transform(tz => tz || undefined),
    }),
});

const pages = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: "./src/content/pages" }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    ogImage: z.string().optional(),
    canonicalURL: z.url().optional(),
  }),
});

export const collections = { posts, pages };
