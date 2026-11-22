import rss from "@astrojs/rss";
import { getCollection, render } from "astro:content";
import { experimental_AstroContainer as AstroContainer } from "astro/container";
import { loadRenderers } from "astro:container";
import { getContainerRenderer as mdxRenderer } from "@astrojs/mdx/container-renderer";
import { getListedPosts } from "@/utils/getSortedPosts";
import { getPostUrl } from "@/utils/getPostPaths";
import config from "@/config";

export async function GET() {
  const posts = await getCollection("posts");
  const sorted = getListedPosts(posts);
  const renderers = await loadRenderers([mdxRenderer()]);
  const container = await AstroContainer.create({ renderers });
  const origin = new URL(config.site.url).origin;

  const items = await Promise.all(
    sorted.map(async post => {
      const { Content } = await render(post);
      const html = await container.renderToString(Content);
      return {
        link: getPostUrl(post.id, post.filePath, config.site.lang),
        title: post.data.title,
        description: post.data.description,
        pubDate: post.data.pubDate,
        categories: post.data.tags,
        // Full content, with root-relative URLs made absolute for feed readers.
        content: html.replace(/(src|href)="\//g, `$1="${origin}/`),
      };
    })
  );

  return rss({
    title: config.site.title,
    description: config.site.description,
    site: config.site.url,
    items,
  });
}
