import type { CollectionEntry } from "astro:content";
import { postFilter } from "./postFilter";

/**
 * Returns posts that are eligible to be shown to users, sorted by “last updated”
 * descending (uses `updatedDate` when present, otherwise `pubDate`).
 *
 * Note: filtering respects drafts and scheduled posts via `postFilter()`.
 */
export function getSortedPosts(posts: CollectionEntry<"posts">[]) {
  return posts
    .filter(postFilter)
    .sort(
      (a, b) =>
        Math.floor(
          new Date(b.data.updatedDate ?? b.data.pubDate).getTime() / 1000
        ) -
        Math.floor(
          new Date(a.data.updatedDate ?? a.data.pubDate).getTime() / 1000
        )
    );
}

/** Posts shown on the home page, /posts and the RSS feed (archived ones excluded). */
export function getListedPosts(posts: CollectionEntry<"posts">[]) {
  return getSortedPosts(posts).filter(({ data }) => !data.archived);
}
