---
title: vidsticker
emoji: ✂️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Turn a video into a transparent animated sticker
---

# vidsticker

Drop in a video, get back an animated sticker with the background cut away —
GIF, WebP or APNG, with real transparency.

The matte is a hybrid: a chroma key gives pixel-accurate edges but keeps
anything in the background that isn't the screen colour (light rays, flares,
sparkle overlays), while a neural matte knows what the subject is but has soft
edges and leaves a screen-coloured fringe. Taking the per-pixel minimum of the
two keeps the strengths of both.

Source and full write-up: <https://github.com/analogik95/Background-remover>

## Notes for this Space

- Free CPU Spaces have 2 vCPU, so matting runs at roughly a second a frame — a
  6 second clip takes about two minutes. Progress streams live while it works.
- The Space sleeps when idle; the first request after a nap pays a cold start.
- Two conversions run at a time. Anything past that queues.
