"""
Plotting Utilities for Bibliometric and Network Analysis

This script provides reusable plotting functions used throughout the analysis
notebook to visualize publication distributions and co-authorship networks.
Extracting these here keeps the notebook focused on analysis while avoiding
duplicated plotting code across cells.

Key functionalities include:
- Choropleth maps of publication counts, with optional zoom into a region.
- Bump charts comparing a country-level metric (e.g. centrality) between two years.
- Co-authorship networks drawn on top of a world map, positioned by country.

Dependencies:
- pandas
- numpy
- matplotlib
- mpl_toolkits (axes_grid1)

Usage:
1. Import the functions into your notebook or script.
2. Prepare the inputs (see each function's docstring for the expected shape).
3. Call the plotting function; each one displays the figure via `plt.show()`.

Example:
    from scripts import visualization

    visualization.plot_choropleth(world_plt, has_published, no_published)
    visualization.plot_bump_chart(deg_2015, deg_2024, country_gr, income_colors)
    visualization.plot_network_on_map(G_lic, world, income_colors, edge_scale=0.05)

Authors: P. Sitthirat et al
Version: 1.0
License: MIT License
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_choropleth(world_plt, has_published, no_published, xlim=None, ylim=None, title=None, figsize=None):
    """
    Plot a choropleth map of publication counts (log-scaled) across countries.

    Countries with zero publications are shaded grey; countries with at least
    one are colored by their 'Log Publication Count'. When `xlim`/`ylim` are
    omitted, the full world map is drawn with a colorbar legend; when given,
    the map is zoomed to that region without a colorbar (for regional insets).

    Parameters:
    - world_plt (geopandas.GeoDataFrame): Base country boundaries to draw.
    - has_published (geopandas.GeoDataFrame): Subset of the merged map/publication
      data with a 'Log Publication Count' column, restricted to countries with
      at least one publication.
    - no_published (geopandas.GeoDataFrame): Subset of the merged map/publication
      data restricted to countries with zero publications.
    - xlim (tuple, optional): (min, max) longitude to zoom into. Default None (full map).
    - ylim (tuple, optional): (min, max) latitude to zoom into. Default None (full map).
    - title (str, optional): Title to display above the map. Default None.
    - figsize (tuple, optional): Figure size. Defaults to (15, 10) for the full
      map and (15, 9) for zoomed regions.

    Returns:
    - None. Displays the plot.

    Example:
        >>> plot_choropleth(world_plt, has_published, no_published)
        >>> plot_choropleth(world_plt, has_published, no_published,
        ...                  xlim=(-100, -60), ylim=(10, 30), title="Caribbean Region")
    """

    is_full_map = xlim is None and ylim is None
    if figsize is None:
        figsize = (15, 10) if is_full_map else (15, 9)

    fig, ax = plt.subplots(figsize=figsize)
    world_plt.plot(ax=ax, edgecolor='black', color='white')
    no_published.plot(ax=ax, color='Grey', edgecolor='black')
    has_published.plot(
        column='Log Publication Count',
        cmap='RdYlGn',
        linewidth=0.8,
        edgecolor='black',
        ax=ax
    )

    if is_full_map:
        # Colorbar range must reflect both plotted subsets (zero-publication
        # countries anchor the bottom of the scale at 0).
        combined_log = pd.concat([has_published['Log Publication Count'], no_published['Log Publication Count']])
        norm = plt.Normalize(vmin=combined_log.min(), vmax=combined_log.max())
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
        sm._A = []

        cax = inset_axes(ax, width="30%", height="3%", loc='lower right', borderpad=3)
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')

        custom_vals = [10, 100, 1000]
        log_ticks = [np.log1p(v) for v in custom_vals]
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([str(v) for v in custom_vals])
        cbar.set_label("Number of Publications")
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if title:
        ax.set_title(title)

    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_bump_chart(scores_before, scores_after, country_gr, income_colors,
                     top_n=15, year_before=2015, year_after=2024,
                     display_gap_after=15, divider_y=16, ylim=(40, 0)):
    """
    Plot a bump chart comparing country ranks on a metric between two years.

    Countries ranked in the top `top_n` by either year's score are kept,
    ranked (rank 1 = highest score), and connected with a line from their
    `year_before` rank to their `year_after` rank, colored by income group.

    Parameters:
    - scores_before (dict): Mapping of country code -> metric score for `year_before`
      (e.g. from `nx.degree_centrality(G)`).
    - scores_after (dict): Mapping of country code -> metric score for `year_after`.
    - country_gr (pd.DataFrame): Country reference table with 'Code', 'Economy',
      and 'Income group' columns (see `data/world-map/country_group.csv`).
    - income_colors (dict): Mapping of income-group label -> color, including
      an 'Other' fallback for unmapped groups.
    - top_n (int, optional): Number of top-ranked countries to keep per year. Default 15.
    - year_before (int or str, optional): Axis label for the earlier year. Default 2015.
    - year_after (int or str, optional): Axis label for the later year. Default 2024.
    - display_gap_after (int, optional): Display ranks beyond this value are
      pushed down by one row to leave a visual gap between the two rank groups.
      Default 15.
    - divider_y (int, optional): Row at which to draw the horizontal divider line. Default 16.
    - ylim (tuple, optional): Y-axis limits (inverted so rank 1 is at the top). Default (40, 0).

    Returns:
    - None. Displays the plot.

    Example:
        >>> deg_2015 = nx.degree_centrality(G_2020)
        >>> deg_2024 = nx.degree_centrality(G_2024)
        >>> plot_bump_chart(deg_2015, deg_2024, country_gr, income_colors)
    """

    df_before = pd.DataFrame(scores_before.items(), columns=['Country', 'Score_before'])
    df_after = pd.DataFrame(scores_after.items(), columns=['Country', 'Score_after'])
    df_merged = pd.merge(df_before, df_after, on='Country', how='outer').fillna(0)

    # Compute ranks (lower rank = more central)
    df_merged['Rank_before_label'] = df_merged['Score_before'].rank(ascending=False, method='min').astype(int)
    df_merged['Rank_before'] = df_merged['Rank_before_label']
    df_merged['Rank_after_label'] = df_merged['Score_after'].rank(ascending=False, method='min').astype(int)
    df_merged['Rank_after'] = df_merged['Rank_after_label']

    # Top-N countries by either year
    top_before = df_merged.nsmallest(top_n, 'Rank_before')['Country']
    top_after = df_merged.nsmallest(top_n, 'Rank_after')['Country']
    top_countries = pd.Series(list(set(top_before).union(set(top_after))))

    # Filter and attach country metadata
    df_bump = df_merged[df_merged['Country'].isin(top_countries)].reset_index(drop=True)
    df_bump = df_bump.merge(country_gr[['Code', 'Economy', 'Income group']], left_on='Country', right_on='Code')
    df_bump = df_bump.sort_values(by='Rank_after').reset_index(drop=True)
    df_bump['Rank_before_display'] = df_bump['Rank_before'].rank(method='first').astype(int)
    df_bump['Rank_after_display'] = df_bump['Rank_after'].rank(method='first').astype(int)
    df_bump.loc[df_bump['Rank_before_display'] > display_gap_after, 'Rank_before_display'] += 1
    df_bump.loc[df_bump['Rank_after_display'] > display_gap_after, 'Rank_after_display'] += 1

    fig, ax = plt.subplots(figsize=(4, 15))

    for _, row in df_bump.iterrows():
        x = [0, 1]
        y = [row['Rank_before_display'], row['Rank_after_display']]
        color = income_colors.get(row['Income group'], income_colors['Other'])

        ax.plot(x, y, color=color, linewidth=3, alpha=1, marker='o', markersize=10)

        ax.text(-0.10, y[0], row['Rank_before_label'], ha='right', va='center', fontsize=9, color=color)
        ax.text(1.10, y[1], f"{row['Rank_after_label']} {row['Economy']}", ha='left', va='center', fontsize=9, color=color)

    ax.axhline(y=divider_y, color='gray', linestyle='--', linewidth=1)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([str(year_before), str(year_after)], fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_network_on_map(G, world, income_colors, edge_scale=1, node_scale=1, title=None):
    """
    Plot a co-authorship network overlaid on a world map, positioning each
    country node at its representative point.

    Parameters:
    - G (networkx.Graph): Co-authorship graph with country-code nodes. Each
      node should carry a 'size' attribute (see `network.network_coauthorship`)
      and an 'Income group' attribute for coloring.
    - world (geopandas.GeoDataFrame): Country boundaries; must include an
      'ADM0_A3' column (matched against graph node ids) and 'geometry'.
    - income_colors (dict): Mapping of income-group label -> color, used with
      a 'gray' fallback for missing/unmapped groups.
    - edge_scale (float, optional): Multiplier applied to edge weight to get
      line width. Default 1.
    - node_scale (float, optional): Multiplier applied to node 'size' to get
      marker size. Default 1.
    - title (str, optional): Plot title. Default None.

    Returns:
    - None. Displays the plot.

    Example:
        >>> plot_network_on_map(G_lic, world, income_colors, edge_scale=0.05, node_scale=1)
    """

    node_countries = list(G.nodes)
    world_nodes = world[world['ADM0_A3'].isin(node_countries)].copy()
    world_nodes['coords'] = world_nodes['geometry'].representative_point()
    positions = {row['ADM0_A3']: (row['coords'].x, row['coords'].y) for _, row in world_nodes.iterrows()}

    fig, ax = plt.subplots(figsize=(15, 10))
    world.plot(ax=ax, color='lightgray', edgecolor='white')

    for u, v, data in G.edges(data=True):
        if u in positions and v in positions:
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            ax.plot([x0, x1], [y0, y1],
                    color='gray',
                    linewidth=data.get('weight', 1) * edge_scale,
                    alpha=0.6,
                    zorder=1)

    for node in G.nodes:
        if node in positions:
            x, y = positions[node]
            size = G.nodes[node].get('size', 1) * node_scale
            income = G.nodes[node].get('Income group', 'Other')
            color = income_colors.get(income, 'gray')
            ax.scatter(x, y,
                       s=size,
                       color=color,
                       edgecolor='black',
                       linewidth=0.5,
                       zorder=2)

    if title:
        plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
