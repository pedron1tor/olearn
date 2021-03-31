import matplotlib.pyplot as plt
from bokeh.layouts import row
from bokeh.io import output_file, show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.plotting import figure, output_file, show
import pandas as pd
import math

# code for embedded_ bokeh heatmap
class GetCorr():
    def __init__(self, df, symbol, selector):
        self.df = df
        self.symbol = symbol
        self.selector = selector

    def GetHeatmap(self):
        correlation = self.df.corr()
        correlation.index.name = 'Yaxis'
        correlation.columns.name = 'Xaxis'
        correlation = correlation.stack().rename("value").reset_index()
        output_file('templates/myPlot.html')

        # You can use your own palette here
        colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']

        # Had a specific mapper to map color with value
        mapper = LinearColorMapper(
            palette=colors, low=correlation.value.min(), high=correlation.value.max())
        # Define a figure
        if self.selector>= 200:
            p = figure(
                plot_width=3000,
                plot_height=3000,
                title=f"{self.symbol} variable correlation",
                x_range=list(correlation.Xaxis.drop_duplicates()),
                y_range=list(correlation.Yaxis.drop_duplicates()),

                tools="hover,save",
                tooltips=[('Variables', '@Xaxis, @Yaxis'), ('Correlation Coefficient', '@value')],
                x_axis_location="above")
            p.xaxis.major_label_orientation = math.pi / 2
            # Create rectangle for heatmap
            p.rect(
                x="Xaxis",
                y="Yaxis",
                width=1,
                height=1,
                source=ColumnDataSource(correlation),
                line_color=None,
                fill_color=transform('value', mapper))
            # Add legend
            color_bar = ColorBar(
                color_mapper=mapper,
                location=(0, 0),
                ticker=BasicTicker(desired_num_ticks=len(colors)))

            p.add_layout(color_bar, 'right')
            show(p)
        if 150>=self.selector>=100:
            p = figure(
                plot_width=1500,
                plot_height=1500,
                title=f"{self.symbol} variable correlation",
                x_range=list(correlation.Xaxis.drop_duplicates()),
                y_range=list(correlation.Yaxis.drop_duplicates()),

                toolbar_location='above',
                tools="hover, save",
                tooltips=[('Variables', '@Xaxis, @Yaxis'), ('Correlation Coefficient', '@value')],
                x_axis_location="above")
            p.xaxis.major_label_orientation = math.pi / 2
            # Create rectangle for heatmap
            p.rect(
                x="Xaxis",
                y="Yaxis",
                width=1,
                height=1,
                source=ColumnDataSource(correlation),
                line_color=None,
                fill_color=transform('value', mapper))
            # Add legend
            color_bar = ColorBar(
                color_mapper=mapper,
                location=(0, 0),
                ticker=BasicTicker(desired_num_ticks=len(colors)))

            p.add_layout(color_bar, 'right')
            show(p)
        if 100>=self.selector>50:
            p = figure(
                plot_width=700,
                plot_height=700,
                title=f"{self.symbol} variable correlation",
                x_range=list(correlation.Xaxis.drop_duplicates()),
                y_range=list(correlation.Yaxis.drop_duplicates()),

                toolbar_location='above',
                tools="hover, save",
                tooltips=[('Variables', '@Xaxis, @Yaxis'), ('Correlation Coefficient', '@value')],
                x_axis_location="above")
            p.xaxis.major_label_orientation = math.pi / 2
            # Create rectangle for heatmap
            p.rect(
                x="Xaxis",
                y="Yaxis",
                width=1,
                height=1,
                source=ColumnDataSource(correlation),
                line_color=None,
                fill_color=transform('value', mapper))
            # Add legend
            color_bar = ColorBar(
                color_mapper=mapper,
                location=(0, 0),
                ticker=BasicTicker(desired_num_ticks=len(colors)))

            p.add_layout(color_bar, 'right')
            show(p)
        if 50>= self.selector:
            p = figure(
                plot_width=700,
                plot_height=700,
                title=f"{self.symbol} variable correlation",
                x_range=list(correlation.Xaxis.drop_duplicates()),
                y_range=list(correlation.Yaxis.drop_duplicates()),

                toolbar_location='above',
                tools="hover, save",
                tooltips=[('Variables', '@Xaxis, @Yaxis'), ('Correlation Coefficient', '@value')],
                x_axis_location="above")
            p.xaxis.major_label_orientation = math.pi / 2
            # Create rectangle for heatmap
            p.rect(
                x="Xaxis",
                y="Yaxis",
                width=1,
                height=1,
                source=ColumnDataSource(correlation),
                line_color=None,
                fill_color=transform('value', mapper))
            # Add legend
            color_bar = ColorBar(
                color_mapper=mapper,
                location=(0, 0),
                ticker=BasicTicker(desired_num_ticks=len(colors)))

            p.add_layout(color_bar, 'right')
            show(p)