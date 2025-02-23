import { ApexOptions } from "apexcharts";
import React from "react";
import ReactApexChart from "react-apexcharts";

export default function LineChart(props: { labels: string[]; series: number[] }) {
  const options: ApexOptions = {
    chart: {
      fontFamily: "Outfit, sans-serif",
      height: 310,
      type: "line",
      toolbar: { show: false },
    },
    stroke: {
      curve: "smooth",
      width: 5,
    },
    fill: {
      type: "gradient",
      gradient: {
        shadeIntensity: 1,
        gradientToColors: ["#FF0000"], // Red at high probabilities
        inverseColors: true,
        opacityFrom: 1,
        opacityTo: 1,
        stops: [0, 50, 100],
        colorStops: [
          { offset: 0, color: "#00FF00", opacity: 1 },  // Green at low probabilities
          { offset: 50, color: "#FFFF00", opacity: 1 }, // Yellow in the middle
          { offset: 100, color: "#FF0000", opacity: 1 }, // Red at high probabilities
        ],
      },
    },
    markers: {
      size: 5,
      strokeColors: "#fff",
      strokeWidth: 2,
      hover: { size: 7 },
    },
    grid: {
      borderColor: "#E5E7EB",
      strokeDashArray: 5,
    },
    dataLabels: { enabled: false },
    tooltip: {
      enabled: true,
      theme: "dark",  // Matches dark mode
      style: {
        fontSize: "12px",
        fontFamily: "Outfit, sans-serif",
      },
      y: {
        formatter: (value: number) => `${(value * 100).toFixed(1)}%`, // Example: Show as percentage
      },
    },
    xaxis: {
      categories: props.labels,
      labels: { style: { fontSize: "12px", colors: "#FFFFFF" } },
      title: { text: "Lap #", style: { color: "#FFFFFF" } },
    },
    yaxis: {
      labels: { style: { fontSize: "12px", colors: "#FFFFFF" } },
      title: { text: "Probability of Crash", style: { color: "#FFFFFF" } },
    },
  };
  
  return (
    <div className="p-4 bg-gray-800 shadow-xl text-white rounded-2xl">
      <h2 className="text-3xl font-semibold pt-2 text-center ">Risk Factor</h2>
      <div className="flex justify-center">
        <ReactApexChart
          options={options}
          series={[{ name: "Data", data: props.series }]} // ✅ Fix: Convert `series` to an object array
          type="line"
          width={500}
          height={310}
        />
      </div>
    </div>
  );
}
