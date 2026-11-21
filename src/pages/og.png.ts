import type { APIRoute } from "astro";
import { renderOg } from "@/utils/renderOg";
import config from "@/config";

export const GET: APIRoute = () =>
  renderOg({
    title: config.site.title,
    subtitle: "Systems engineer writing about AI inference infrastructure",
    left: "",
    right: new URL(config.site.url).hostname,
  });
