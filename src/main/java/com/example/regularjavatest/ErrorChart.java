package com.example.regularjavatest;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;

public class ErrorChart extends ApplicationFrame {

    public ErrorChart(String title, XYSeries errorSeries) {
        super(title);
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(errorSeries);
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Stuff I suppose",  // chart title
                "Iteration",        // x axis label
                "Yeah lol",            // y axis label
                dataset,            // data
                PlotOrientation.VERTICAL,
                true,               // include legend
                true,               // tooltips
                false               // urls
        );
        ChartPanel chartPanel = new ChartPanel(chart);
        chart.getXYPlot().getDomainAxis().setLabel("Generation");
        chart.getXYPlot().getRangeAxis().setLabel("Error");
        chartPanel.setVisible(true);

        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        setContentPane(chartPanel);

    }
}
