var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 960 - margin.left - margin.right, 
    height = 500 - margin.top - margin.bottom;

var numberOfPoints = 100;
var pointRadius = 9;


var confusionMatrix = gConfusionMatrix;

function createConfusionMatrix3(options) {
    var width = 200;
    var height = 200;
    var container = options.container;
    var legend = options.legend;


    var xExtent = d3.extent(confusionMatrix, function(d) { return d[0] });
    var yExtent = d3.extent(confusionMatrix, function(d) { return d[1] });
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


    // Lasso functions to execute while lassoing
    var lasso_start = function() {
        lasso.items()
        .attr("r",3.5) // reset size
        .style("fill",null) // clear all of the fills
        .classed({"not_possible":true,"selected":false}); // style as not possible
    };

    var lasso_draw = function() {
        // Style the possible dots
        lasso.items().filter(function(d) {return d.possible===true})
        .classed({"not_possible":false,"possible":true});

        // Style the not possible dot
        lasso.items().filter(function(d) {return d.possible===false})
        .classed({"not_possible":true,"possible":false});
    };

    var lasso_end = function() {
        // Reset the color of all dots
        lasso.items()
        .style("fill", function(d) { return color(d[2]); });

        // Style the selected dots
        lasso.items().filter(function(d,j) {
        //document.write(i);
        if(d.selected===true) document.write(j);
         for (var i = d.length - 1; i >= 0; i--) {
             d[0];
             if(d.selected===true) document.write(j+" - ");
         }
         return d.selected===true;
        })
        .classed({"not_possible":false,"possible":false})
        .attr("r",7);

        // Reset the style of the not selected dots
        lasso.items().filter(function(d) {return d.selected===false})
        .classed({"not_possible":false,"possible":false})
        .attr("r",3.5);
    };

    // Create the area where the lasso event can be triggered
    var lasso_area = svg.append("rect")
                      .attr("width",width)
                      .attr("height",height)
                      .style("opacity",0);

    // Define the lasso
    var lasso = d3.lasso()
        .closePathDistance(75) // max distance for the lasso loop to be closed
        .closePathSelect(true) // can items be selected by closing the path?
        .hoverSelect(true) // can items by selected by hovering over them?
        .area(lasso_area) // area where the lasso can be started
        .on("start",lasso_start) // lasso start function
        .on("draw",lasso_draw) // lasso draw function
        .on("end",lasso_end); // lasso end function

    // Init the lasso on the svg:g that contains the dots
    svg.call(lasso);


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
        .data(confusionMatrix)
        .enter().append("circle")
        .attr("id",function(d,i) {return "dot_" + i;}) // added
        .attr("class", "dot")
        .attr("r", 3.5)
        .attr("cx", function(d) { return x(d[0]); })
        .attr("cy", function(d) { return y(d[1]); })
        .style("fill", function(d) { return color(d[2]); });

    lasso.items(d3.selectAll(".dot"));

}