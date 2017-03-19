---
layout: post
title: OpenStreetMap rendering and simple Routing.
excerpt: This post is about a small experiment of loading Openstreetmap files and rendering them with both matplotlib and vispy. Also tried a simple routing with dijkstra and A*.
tags: [python, openstreetmap, vispy, dijkstra, A*]
modified: 2016-03-25
comments: true
crosspost_to_medium: true
---
[Github](https://github.com/ssarangi/osmpy)


Thanks to [Pranabesh](prnbs.github.io) for working with me on this.

# Introduction:

OpenStreetMaps are becoming more and more popular with open data and truly revolutionizing the maps industry. There are a lot of players in this field but one of the most prominent has been MapBox.
Mapbox has editors to edit and upload OSM files and their rendering is just too beautiful.

# Structure:

OSM typically uses 2 file formats. XML or PBF. PBF is short for Google's protocol buffers which is just a compressed format. XML on the other hand is quite verbose.

## XML format:
OSM uses Nodes to represent latitudes and longitudes. These nodes are then combined to form complex structures. For example, a collection of nodes form the buildings or highways or roads etc.
So the structure is pretty simple to understand.

Each OSM file starts with a Bounding box.

~~~xml
<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="CGImap 0.4.0 (23328 thorn-02.openstreetmap.org)" copyright="OpenStreetMap and contributors" attribution="http://www.openstreetmap.org/copyright" license="http://opendatacommons.org/licenses/odbl/1-0/">
<bounds minlat="37.4081400" minlon="-121.8716900" maxlat="37.4099500" maxlon="-121.8684400"/>
~~~

The next step are the nodes. Every node has a unique id which is what it is referenced by along with whether its visible on the map or not. The rest of the tags are almost self-explanatory. lat, lon and id are what we will use here.

~~~xml
<node id="65425741" visible="true" version="4" changeset="3424990" timestamp="2009-12-22T06:37:55Z" user="woodpeck_fixbot" uid="147510" lat="37.4169957" lon="-121.8541592"/>
<node id="65443553" visible="true" version="6" changeset="6857224" timestamp="2011-01-04T03:51:20Z" user="mk408" uid="201724" lat="37.4129151" lon="-121.8624849"/>
<node id="65452440" visible="true" version="6" changeset="25041843" timestamp="2014-08-27T00:25:00Z" user="StellanL" uid="28775" lat="37.4100723" lon="-121.8683465"/>
<node id="65506168" visible="true" version="4" changeset="2817765" timestamp="2009-10-11T18:55:54Z" user="woodpeck_fixbot" uid="147510" lat="37.4076559" lon="-121.8691881"/>
<node id="65511824" visible="true" version="4" changeset="3140985" timestamp="2009-11-17T13:34:37Z" user="woodpeck_fixbot" uid="147510" lat="37.4084658" lon="-121.8697810"/>
<node id="65588013" visible="true" version="7" changeset="32500664" timestamp="2015-07-08T17:31:12Z" user="StellanL" uid="28775" lat="37.4091332" lon="-121.8702756">
<tag k="highway" v="traffic_signals"/>
</node>
~~~

A way generally represents an object, which could be a highway, residential roads, primary roads, secondary roads etc. A way would use a bunch of node id's to define its structure. The tag element identifies what the way represents.
It might also contain auxiliary data called Tiger Data which is data released by the US Census Bureau.

~~~xml
<way id="8940804" visible="true" version="8" changeset="7051091" timestamp="2011-01-22T14:48:02Z" user="mk408" uid="201724">
  <nd ref="65511824"/>
  <nd ref="65511826"/>
  <nd ref="65511829"/>
  <nd ref="65511832"/>
  <nd ref="65511834"/>
  <nd ref="65511837"/>
  <nd ref="65511838"/>
  <nd ref="313453779"/>
  <nd ref="313453780"/>
  <nd ref="65511840"/>
  <nd ref="65511841"/>
  <tag k="highway" v="tertiary"/>
  <tag k="name" v="Junewood Avenue"/>
  <tag k="tiger:cfcc" v="A41"/>
  <tag k="tiger:county" v="Santa Clara, CA"/>
  <tag k="tiger:name_base" v="Junewood"/>
  <tag k="tiger:name_type" v="Ave"/>
  <tag k="tiger:separated" v="no"/>
  <tag k="tiger:source" v="tiger_import_dch_v0.6_20070809"/>
  <tag k="tiger:tlid" v="122962140:122962213:122962214:122962219:122962228:122962234"/>
  <tag k="tiger:upload_uuid" v="bulk_upload.pl-9f300d22-5de3-4867-bd5e-8c2a200c22ad"/>
  <tag k="tiger:zip_left" v="95132"/>
  <tag k="tiger:zip_right" v="95132"/>
 </way>
~~~

With this information we start the process of rendering the maps. A graph is created for the roads which is then used to render.
Used NetworkX to create the graph. Rendering to Matplotlib was done with rules to render different kinds of structures.

~~~python
class MatplotLibMap:
    renderingRules = {
        'primary': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           =  (0.933, 0.51, 0.933),  #'#ee82ee',
                zorder          = 400,
        ),
        'primary_link': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = (0.85, 0.44, 0.84), # '#da70d6',
                zorder          = 300,
        ),
        'secondary': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = (0.85, 0.75, 0.85), # '#d8bfd8',
                zorder          = 200,
        ),
        'secondary_link': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = (0.85, 0.75, 0.85), # '#d8bfd8',
                zorder          = 200,
        ),
        'tertiary': dict(
                linestyle       = '-',
                linewidth       = 4,
                color           = (1.0, 0.0, 0.0), #(0.0,0.0,0.7),
                zorder          = 100,
        ),
        'tertiary_link': dict(
                linestyle       = '-',
                linewidth       = 4,
                color           = (1.0, 0.0, 0.0), #(0.0,0.0,0.7),
                zorder          = 100,
        ),
        'residential': dict(
                linestyle       = '-',
                linewidth       = 1,
                color           = (1.0, 1.0, 0.0), #(0.1,0.1,0.1),
                zorder          = 50,
        ),
        'unclassified': dict(
                linestyle       = ':',
                linewidth       = 1,
                color           = (0.5,0.5,0.5),
                zorder          = 10,
        ),
        'calculated_path': dict(
                linestyle       = '-',
                linewidth       = 4,
                color           = (1.0,0.0,0.0),
                zorder          = 2000,
        ),
        'correct_path': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = (0.6,0.8,0.0),
                zorder          = 1900,
        ),
        'default': dict(
                linestyle       = '-',
                linewidth       = 3,
                color           = (1.0, 0.48, 0.0),
                zorder          = 500,
                ),

        'other': dict(
                linestyle       = '-',
                linewidth       = 3,
                color           = (0.6, 0.6, 0.6),
                zorder          = 500,
                ),
        }
~~~

The render method is also not very complicated.

~~~python
def render(self, axes, plot_nodes=False):
    plt.sca(axes)
    for idx, nodeID in enumerate(self._osm.ways.keys()):
        wayTags = self._osm.ways[nodeID].tags
        wayType = None
        if 'highway' in wayTags.keys():
            wayType = wayTags['highway']

        if wayType in [
                       'primary',
                       'primary_link',
                       'unclassified',
                       'secondary',
                       'secondary_link',
                       'tertiary',
                       'tertiary_link',
                       'residential',
                       'trunk',
                       'trunk_link',
                       'motorway',
                       'motorway_link'
                        ]:
            oldX = None
            oldY = None

            if wayType in list(MatplotLibMap.renderingRules.keys()):
                thisRendering = MatplotLibMap.renderingRules[wayType]
            else:
                thisRendering = MatplotLibMap.renderingRules['default']

            for nCnt, nID in enumerate(self._osm.ways[nodeID].nds):
                y = float(self._osm.nodes[nID].lat)
                x = float(self._osm.nodes[nID].lon)

                self._node_map[(x, y)] = nID

                if oldX is None:
                    pass
                else:
                    plt.plot([oldX,x],[oldY,y],
                            marker          = '',
                            linestyle       = thisRendering['linestyle'],
                            linewidth       = thisRendering['linewidth'],
                            color           = thisRendering['color'],
                            solid_capstyle  = 'round',
                            solid_joinstyle = 'round',
                            zorder          = thisRendering['zorder'],
                            picker=2
                    )

                    if plot_nodes == True and (nCnt == 0 or nCnt == len(self._osm.ways[nodeID].nds) - 1):
                        plt.plot(x, y,'ro', zorder=5)

                oldX = x
                oldY = y

    self._fig.canvas.mpl_connect('pick_event', self.__onclick__)
    plt.draw()
~~~

## Matplotlib Rendering

{: .center}
![Matplotlib](https://raw.githubusercontent.com/ssarangi/osmpy/master/shortest_path.png)

## Vispy Rendering

{: .center}
![Matplotlib](https://raw.githubusercontent.com/ssarangi/osmpy/master/vispy_rendering.png)
