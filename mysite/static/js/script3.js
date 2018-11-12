var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 960 - margin.left - margin.right, 
    height = 500 - margin.top - margin.bottom;

var numberOfPoints = 100;
var pointRadius = 9;




var confusionMatrix2 = gConfusionMatrix2;


function createConfusionMatrix4(options) {
        var width = 200;
        var height = 200;
        var container = options.container;
        var legend = options.legend;


        var xExtent = d3.extent(confusionMatrix2, function(d) { return d[0] });
    var yExtent = d3.extent(confusionMatrix2, function(d) { return d[1] });
    var xRange = xExtent[1] - xExtent[0];
    var yRange = yExtent[1] - yExtent[0];




        var svg = d3.select(container).append("svg")
      .attr("width", width + margin.left + margin.right)
     .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


      var x = d3.scale.linear()
      .domain([xExtent[0] - xRange*0.1, xExtent[1] + xRange*0.1])
    .range([0, width]);

   var y = d3.scale.linear()
   .domain([yExtent[0] - yRange*0.1, yExtent[1] + yRange*0.1])
    .range([height, 0]);

   var color = d3.scale.category10();

   var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

   var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left");


    svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .style("text-anchor", "end")
      .text("Sepal Width (cm)");

  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("class", "label")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Sepal Length (cm)")



      svg.selectAll(".dot")
      .data(confusionMatrix2)
      .enter().append("circle")
      .attr("id",function(d,i) {return "dot_" + i;}) // added
      .attr("class", "dot")
      .attr("r", 3.5)
      .attr("cx", function(d) { return x(d[0]); })
      .attr("cy", function(d) { return y(d[1]); })
      .style("fill", function(d) { return color(d[2]); });





}