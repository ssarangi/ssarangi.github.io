---
layout: post
title: OpenStreetMap rendering and simple Routing.
excerpt: This post is about a small experiment of loading Openstreetmap files and rendering them with both matplotlib and vispy. Also tried a simple routing with dijkstra and A*.
tags: [python, openstreetmap, vispy, dijkstra, A*]
modified: 2016-03-25
comments: true
---

# Introduction:

OpenStreetMaps are becoming more and more popular with open data and truly revolutionizing the maps industry. There are a lot of players in this field but one of the most prominent has been MapBox.
Mapbox has editors to edit and upload OSM files and their rendering is just too beautiful.

# Structure:

OSM typically uses 2 file formats. XML or PBF. PBF is short for Google's protocol buffers which is just a compressed format. XML on the other hand is quite verbose.

## Matplotlib Rendering
{: .center}
![Matplotlib](https://raw.githubusercontent.com/ssarangi/osmpy/master/shortest_path.png)

## Vispy Rendering
{: .center}
![Matplotlib](https://raw.githubusercontent.com/ssarangi/osmpy/master/vispy_rendering.png)
