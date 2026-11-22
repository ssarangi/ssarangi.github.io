import getReadingTime from "reading-time";
import { toString } from "mdast-util-to-string";

/** Adds `minutesRead` to the frontmatter exposed via render().remarkPluginFrontmatter */
export function remarkReadingTime() {
  return (tree, { data }) => {
    // Code and images shouldn't count as prose.
    const text = toString(tree, { includeImageAlt: false });
    data.astro.frontmatter.minutesRead = Math.max(
      1,
      Math.ceil(getReadingTime(text).minutes)
    );
  };
}
