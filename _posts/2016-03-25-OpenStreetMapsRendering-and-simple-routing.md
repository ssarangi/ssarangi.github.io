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

## XML format:
OSM uses Nodes to represent latitudes and longitudes. These nodes are then combined to form complex structures. For example, a collection of nodes form the buildings or highways or roads etc.
So the structure is pretty simple to understand.

Each OSM file starts with a Bounding box.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="CGImap 0.4.0 (23328 thorn-02.openstreetmap.org)" copyright="OpenStreetMap and contributors" attribution="http://www.openstreetmap.org/copyright" license="http://opendatacommons.org/licenses/odbl/1-0/">
 <bounds minlat="37.4081400" minlon="-121.8716900" maxlat="37.4099500" maxlon="-121.8684400"/>
 ```

 

## Matplotlib Rendering
{: .center}
![Matplotlib](https://raw.githubusercontent.com/ssarangi/osmpy/master/shortest_path.png)

## Vispy Rendering
{: .center}
![Matplotlib](https://raw.githubusercontent.com/ssarangi/osmpy/master/vispy_rendering.png)
