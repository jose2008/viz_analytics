function matrix10(options){
	container = options.container;
	data = options.data;
	var width  = 300;
var height = 300;

var colores_g = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6", "#dd4477", "#66aa00", "#b82e2e", "#316395", "#994499", "#22aa99", "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#651067", "#329262", "#5574a6", "#3b3eac"];


//create svg element
var svg = d3.select(container)
	.append("svg")
	.data(data)
	.attr("width",width)
	.attr("height",height);
	//append circle
	document.write("before...")
	for(i=0; i<data.length; i++){
		for(j=0; j<data.length; j++){
			if(data[i][j] == 1) c = 1;
			else c=0
			document.write(data.length)
			svg.append("rect")
			.attr("x",1*j)
			.attr("y",1*i)
			.attr("width",5)
			.attr("height",5)
			.attr("fill", colores_g[c] );	
		}
	}
}
			



