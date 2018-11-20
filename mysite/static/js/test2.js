
var matrixParams1 = { container : '#container1',
                      legend    : '#legend1',
                      model     :  model_kmean,
                      count     : 0

                    };

var matrixParams2 = { 	container : '#container2',
                    	legend    : '#legend2',
                    	model     : model_birch,
                    	count     : 1000
                    };

var matrixParams3 = { 	container : '#container3',
                        legend    : '#legend3',
                        model 	  : model_som,
                        count     : 2000
                    };

var matrixParams4 = { 	container : '#container4',
                       	legend    : '#legend4',
                       	model     : model_voting,
                       	count     : 3000
                    };


var confusionMatrix = model_kmean;
var confusionMatrix2 = model_birch;
var confusionMatrix3 = model_som;
var confusionMatrix4 = model_voting;

//print(confusionMatrix);

createGraficaModel(matrixParams1);
createGraficaModel(matrixParams2);
createGraficaModel(matrixParams3);
createGraficaModel(matrixParams4);



//createConfusionMatrix2( matrixParams1 );

//createConfusionMatrix5( matrixParams2 );
//createConfusionMatrix3( matrixParams3 );
//createConfusionMatrix4( matrixParams4 );
//createConfusionMatrix5( matrixParams1 );