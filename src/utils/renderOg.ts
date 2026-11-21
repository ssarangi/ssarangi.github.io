import fs from "node:fs/promises";
import path from "node:path";
import satori from "satori";
import sharp from "sharp";

const font = (weight: 400 | 700) =>
  fs.readFile(
    path.join(
      process.cwd(),
      `node_modules/@fontsource/inter/files/inter-latin-${weight}-normal.woff`
    )
  );

type OgArgs = { title: string; subtitle?: string; left: string; right: string };

/** Minimal typographic OG card: near-white, one accent rule. */
export async function renderOg({ title, subtitle, left, right }: OgArgs) {
  const [regular, bold] = await Promise.all([font(400), font(700)]);
  const svg = await satori(
    {
      key: null,
      type: "div",
      props: {
        style: {
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background: "#fcfcfb",
          color: "#1b1b1d",
          padding: "72px 80px",
          fontFamily: "Inter",
          borderTop: "14px solid #1f5fd1",
        },
        children: [
          {
            type: "div",
            props: {
              style: { display: "flex", flexDirection: "column" },
              children: [
                {
                  type: "div",
                  props: {
                    style: {
                      fontSize: title.length > 60 ? 56 : 68,
                      fontWeight: 700,
                      lineHeight: 1.12,
                      letterSpacing: "-0.02em",
                      maxHeight: 360,
                      overflow: "hidden",
                    },
                    children: title,
                  },
                },
                ...(subtitle
                  ? [
                      {
                        type: "div",
                        props: {
                          style: {
                            fontSize: 30,
                            marginTop: 28,
                            color: "#5f6368",
                            lineHeight: 1.35,
                          },
                          children: subtitle,
                        },
                      },
                    ]
                  : []),
              ],
            },
          },
          {
            type: "div",
            props: {
              style: {
                display: "flex",
                justifyContent: "space-between",
                fontSize: 28,
                color: "#5f6368",
              },
              children: [
                { type: "span", props: { children: left } },
                {
                  type: "span",
                  props: {
                    style: { fontWeight: 700, color: "#1b1b1d" },
                    children: right,
                  },
                },
              ],
            },
          },
        ],
      },
    },
    {
      width: 1200,
      height: 630,
      fonts: [
        { name: "Inter", data: regular, weight: 400, style: "normal" },
        { name: "Inter", data: bold, weight: 700, style: "normal" },
      ],
    }
  );
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return new Response(new Uint8Array(png), {
    headers: { "Content-Type": "image/png" },
  });
}
