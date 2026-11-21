import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://ssarangi.github.io/",
    title: "Satyajit Sarangi",
    description:
      "Systems engineer writing about AI inference infrastructure: inference engines, tool-call parsing, LoRA, constrained decoding, and compilers.",
    author: "Satyajit Sarangi",
    profile: "https://github.com/ssarangi",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "UTC",
    dir: "ltr",
  },
  posts: {
    perPage: 10,
    perIndex: 6,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: true,
    showArchives: true,
    showBackButton: true,
    editPost: {
      enabled: true,
      url: "https://github.com/ssarangi/ssarangi.github.io/edit/master/",
    },
    search: "pagefind",
    toc: true,
  },
  socials: [
    { name: "github", url: "https://github.com/ssarangi" },
    { name: "linkedin", url: "https://www.linkedin.com/in/satyajitsarangi/" },
    { name: "x", url: "https://x.com/ssarangi" },
    { name: "mail", url: "mailto:satyajit.sarangi@gmail.com" },
  ],
  shareLinks: [
    { name: "x", url: "https://x.com/intent/post?url=" },
    { name: "linkedin", url: "https://www.linkedin.com/sharing/share-offsite/?url=" },
    { name: "mail", url: "mailto:?subject=See%20this%20post&body=" },
  ],
});